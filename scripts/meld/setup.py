#!/usr/bin/env python
# encoding: utf-8
"""
Set up a MELD (Modeling Employing Limited Data) H/T-REMD run to compute the 
protein stability difference between a wild-type and mutant proteins.

The script expects a single template PDB file inside a ``TEMPLATES``
directory, named as ``<protein>_<mutation>_meld.pdb`` (only the first
four characters of the filename are used as the protein name, and characters
5 through -9 are used as the mutation label -- see :func:`find_pdb_file`).

It also expects the following auxiliary files in the working directory:
    - ``<protein_name>_rg.dat``: a single radius-of-gyration value used to
      size the cartesian restraints.
    - ``<protein_name>_<mutation>_distogram.dat``: pairwise distance restraints.
    - ``<protein_name>_<mutation>_distogram_mutation.dat``: pairwise distance restraints.

The system is constructed using:
    - Amber ff14SB side-chain force field
    - GB-Neck2 implicit solvent
    - 30 replica Hamiltonian exchange
    - Cartesian confinement restraints
    - MELD distance restraints from native distogram

Running this script builds the MELD system, attaches cartesian and
distogram restraints, configures the replica-exchange ladder, and
writes everything to the MELD data store (``Data/`` by default).
"""

import os

import numpy as np
from openmm import unit as u

import meld
from meld import comm, remd, vault
from meld.remd import adaptor, ladder
from meld.system.scalers import LinearRamp

# --------------------------------------------------------------------------
# Replica exchange / simulation constants
# --------------------------------------------------------------------------
N_REPLICAS = 30          # number of replicas in the temperature/alpha ladder
N_STEPS = 500_000        # number of steps, in units of the exchange period
BLOCK_SIZE = 100         # number of blocks used for trajectory storage

# Cartesian restraint force constant (kJ/mol/nm^2)
SPRING_CONSTANT = 250.0

# Distance restraint parameters
DISTANCE_TOLERANCE = 0.3   # nm; defines r1/r4 offsets from the target distance
DISTANCE_INNER_TOLERANCE = 0.2  # nm; defines r2/r3 offsets from the target distance
DISTANCE_FORCE_CONSTANT = 250.0  # kJ/mol/nm^2

# Fraction of restraints in a group that must be active (distogram groups)
DISTOGRAM_ACTIVE_FRACTION = 0.8

TEMPLATES_DIR = "TEMPLATES"


def count_residues(pdb_filepath):
    """Count the number of unique residues in a PDB file.

    A residue is identified by its chain identifier + residue sequence
    number, taken directly from columns 22-26 of each ATOM/HETATM record.
    """
    residues = set()
    with open(pdb_filepath, "r") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                residue_id = line[21:26]  # chain identifier + residue sequence number
                residues.add(residue_id)
    return len(residues)


def find_pdb_file():
    """Return the path to the first .pdb file found in TEMPLATES/, or None."""
    if os.path.isdir(TEMPLATES_DIR):
        for file in os.listdir(TEMPLATES_DIR):
            if file.lower().endswith(".pdb"):
                return os.path.join(TEMPLATES_DIR, file)
    return None


def distogram(filename, s, scaler):
    """Build distance-restraint groups from a distogram file.

    The file is a whitespace-separated table with columns
    ``i name_i j name_j distance``, with blank lines separating restraint
    groups. Each group is wrapped in a restraint group that enforces 80% of
    its restraints to be satisfied.
    """
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(
                s.restraints.create_restraint_group(
                    rest_group, int(len(rest_group) * DISTOGRAM_ACTIVE_FRACTION)
                )
            )
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])
            name_i = cols[1]
            j = int(cols[2])
            name_j = cols[3]
            dist = float(cols[4])
            atom_1_index = s.index.atom(i - 1, name_i)
            atom_2_index = s.index.atom(j - 1, name_j)
            rest = s.restraints.create_restraint(
                "distance",
                scaler,
                LinearRamp(0.0, 100.0, 0.0, 1.0),
                r1=(dist - DISTANCE_TOLERANCE) * u.nanometer,
                r2=(dist - DISTANCE_INNER_TOLERANCE) * u.nanometer,
                r3=(dist + DISTANCE_INNER_TOLERANCE) * u.nanometer,
                r4=(dist + DISTANCE_TOLERANCE) * u.nanometer,
                k=DISTANCE_FORCE_CONSTANT * u.kilojoule_per_mole / u.nanometer**2,
                atom1=atom_1_index,
                atom2=atom_2_index,
            )
            rest_group.append(rest)
    dists.append(
        s.restraints.create_restraint_group(
            rest_group, int(len(rest_group) * DISTOGRAM_ACTIVE_FRACTION)
        )
    )
    return dists


def cartesian_wild(s, scaler, residues, delta, k=SPRING_CONSTANT):
    """Restrain the CA atoms of the wild-type residues near the origin."""
    cart = []
    backbone = ["CA"]
    for i in residues:
        for b in backbone:
            atom_index = s.index.atom(i, b)
            rest = s.restraints.create_restraint(
                "cartesian",
                scaler,
                LinearRamp(0.0, 100.0, 0.0, 1.0),
                atom_index=atom_index,
                x=0 * u.nanometer,
                y=0 * u.nanometer,
                z=0 * u.nanometer,
                delta=delta * u.nanometer,
                force_const=k * u.kilojoules_per_mole / (u.nanometer * u.nanometer),
            )
            cart.append(rest)
    return cart


def cartesian_mutant(s, scaler, residues, delta, k=SPRING_CONSTANT):
    """Restrain the CA atoms of the mutant residues near (100, 0, 0) nm.

    Offsetting the mutant residues far from the wild-type residues keeps the
    two copies of the system from interacting with one another.
    """
    cart = []
    backbone = ["CA"]
    for i in residues:
        for b in backbone:
            atom_index = s.index.atom(i, b)
            rest = s.restraints.create_restraint(
                "cartesian",
                scaler,
                LinearRamp(0.0, 100.0, 0.0, 1.0),
                atom_index=atom_index,
                x=100 * u.nanometer,
                y=0 * u.nanometer,
                z=0 * u.nanometer,
                delta=delta * u.nanometer,
                force_const=k * u.kilojoules_per_mole / (u.nanometer * u.nanometer),
            )
            cart.append(rest)
    return cart


def gen_state(s, index):
    """Generate a starting state template with alpha set by replica index."""
    state = s.get_state_template()
    state.alpha = index / (N_REPLICAS - 1.0)
    return state


def setup_system():
    """Build the MELD system, attach restraints, and populate the data store.

    Returns
    -------
    int
        The number of atoms in the built system.
    """
    pdb_filepath = find_pdb_file()
    if not pdb_filepath:
        print(f'No PDB files found in the "{TEMPLATES_DIR}" directory.')
        return None

    protein_name = os.path.basename(pdb_filepath)[:-4][:4]
    mutation = os.path.basename(pdb_filepath)[5:-9]
    num_residues = count_residues(pdb_filepath)
    print(
        f"Protein: {protein_name}, Number of residues: {num_residues}, "
        f"Mutation: {mutation}"
    )

    rg = np.loadtxt(f"{protein_name}_rg.dat")
    radius = 5 * rg

    wild_residues = list(range(0, num_residues - 1))
    mutant_residues = list(range(num_residues, 2 * num_residues - 1))

    # Build the system from the template PDB.
    p = meld.AmberSubSystemFromPdbFile(pdb_filepath)
    b = meld.AmberOptions(
        forcefield="ff14sbside",
        implicit_solvent_model="gbNeck2",
        remove_com=False,
        use_big_timestep=False,
        use_bigger_timestep=True,  # MD timestep (4.5 fs)
        cutoff=1.8 * u.nanometers,
        enable_amap=False,
        amap_beta_bias=1.0,
    )

    bb = meld.AmberSystemBuilder(b)
    s = bb.build_system([p]).finalize()
    s.temperature_scaler = meld.GeometricTemperatureScaler(
        0, 0.9, 300.0 * u.kelvin, 500.0 * u.kelvin
    )

    cartesian_wild_scaler = s.restraints.create_scaler("constant")
    cartesian_mutant_scaler = s.restraints.create_scaler("constant")

    distogram_scaler = s.restraints.create_scaler(
        "nonlinear", alpha_min=0.1, alpha_max=1, factor=4, strength_at_alpha_max=0.00001
    )
    distogram_mutation_scaler = s.restraints.create_scaler(
        "nonlinear", alpha_min=0.1, alpha_max=1, factor=4, strength_at_alpha_max=0.00001
    )

    print("All Scalers Loaded")

    cartesian_wild_rest = cartesian_wild(s, cartesian_wild_scaler, wild_residues, radius)
    s.restraints.add_as_always_active_list(cartesian_wild_rest)

    cartesian_mutant_rest = cartesian_mutant(
        s, cartesian_mutant_scaler, mutant_residues, radius
    )
    s.restraints.add_as_always_active_list(cartesian_mutant_rest)

    print("Cartesian Loaded")

    distogram_restraints = distogram(
        f"{protein_name}_{mutation}_distogram.dat", s, scaler=distogram_scaler
    )
    distogram_mutation_restraints = distogram(
        f"{protein_name}_{mutation}_distogram_mutation.dat",
        s,
        scaler=distogram_mutation_scaler,
    )

    s.restraints.add_selectively_active_collection(distogram_restraints, 1)
    s.restraints.add_selectively_active_collection(distogram_mutation_restraints, 1)

    print("Distograms Loaded")
    print("All Restraints Loaded")

    options = meld.RunOptions(
        timesteps=14445,  # number of MD steps per exchange
        minimize_steps=20000,
    )

    store = vault.DataStore(
        gen_state(s, 0), N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE
    )
    store.initialize(mode="w")
    store.save_system(s)
    store.save_run_options(options)

    # Set up replica exchange (temperature ladder, adaptor, communicator).
    l = ladder.NearestNeighborLadder(n_trials=2500)
    policy_1 = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy_1)
    remd_runner = remd.leader.LeaderReplicaExchangeRunner(
        N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a
    )
    store.save_remd_runner(remd_runner)

    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS, timeout=60000)
    store.save_communicator(c)

    states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)
    store.save_data_store()

    return s.n_atoms


if __name__ == "__main__":
    setup_system()