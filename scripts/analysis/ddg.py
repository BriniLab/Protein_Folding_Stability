"""
Trajectory analysis for predicting protein folding stability difference (ddG) upon mutation using MELD simulations.

Simulation setup
-----------------
Wild-type and mutant are simulated in a single system. The topology is therefore composed of two consecutive halves along the residue index: the first half is wild-type, the second half is the mutant.

The reference native structure contains two copies (same total residue count as the simulation topology), so each half is superposed onto its own corresponding residues in the reference.

Analysis
--------
1. Load the trajectory and a reference native structure.
2. For each frame, compute the CA-RMSD of each protein to the native structure.
   A protein is assigned as "folded" if its RMSD is below `FOLDED_CUTOFF` (4 Å).
3. Select frames where only one the two proteins is folded (valid frames).
4. Compute ddG from the relative frequency of "wild folded" vs. "mutant
   folded" among those frames, as
   ddG = -k*T * log(n_wild_folded / n_mutant_folded).

   Sign convention: 
   ddG < 0 means the wild-type was folded in more often than the mutant therefore the mutation is destabilizing.
   ddG > 0 means the mutant was folded more often than the wild-type therefore the mutation is stabilizing. 
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import mdtraj


logger = logging.getLogger(__name__)

BOLTZMANN_CONSTANT = 0.0019872041   # kcal / (mol K)
TEMPERATURE = 298.0                 # K
FOLDED_CUTOFF = 0.4                 # nm (= 4 Angstrom); RMSD below this counts as "folded"
EQUILIBRATION_FRAMES = 3000         # leading frames discarded before analysis


# --------------------------------------------------------------------------
# Topology helpers
# --------------------------------------------------------------------------

def count_residues(pdb_filepath: Union[str, Path], protein_only: bool = True) -> int:
    """
    Count residues in a PDB file

    Parameters
    ----------
    pdb_filepath : Path to a PDB file.
    protein_only : If True (default), count only protein residues, ignoring
        any solvent, ions, or ligands. `analyze_trajectory` assumes the
        topology contains only the wild-type and mutant protein copies.
    """
    top = mdtraj.load(str(pdb_filepath)).topology
    if protein_only:
        return sum(1 for res in top.residues if res.is_protein)
    return top.n_residues


# --------------------------------------------------------------------------
# RMSD
# --------------------------------------------------------------------------

def compute_rmsd(
    traj: mdtraj.Trajectory,
    reference: mdtraj.Trajectory,
    traj_selection: str,
    ref_selection: Optional[str] = None,
) -> np.ndarray:
    """
    CA-RMSD (nm) of `traj` relative to `reference`, after superposing on
    the selected CA atoms.

    Parameters
    ----------
    traj, reference : mdtraj trajectories.
    traj_selection : mdtraj atom-selection string picking out the residues
        of interest in `traj`.
    ref_selection : mdtraj atom-selection string picking out the
        corresponding residues in `reference`. Defaults to `traj_selection`
    """
    if ref_selection is None:
        ref_selection = traj_selection

    traj_atoms = traj.topology.select(f"name CA and ({traj_selection})")
    ref_atoms = reference.topology.select(f"name CA and ({ref_selection})")

    if len(traj_atoms) == 0 or len(ref_atoms) == 0:
        raise ValueError(
            f"Selection matched no CA atoms (traj: {len(traj_atoms)}, "
            f"reference: {len(ref_atoms)}). Check traj_selection/ref_selection."
        )
    if len(traj_atoms) != len(ref_atoms):
        raise ValueError(
            f"traj_selection and ref_selection pick a different number of CA "
            f"atoms ({len(traj_atoms)} vs {len(ref_atoms)}); RMSD requires a "
            "1:1 residue correspondence between them."
        )

    traj_region = traj.atom_slice(traj_atoms)
    reference_region = reference.atom_slice(ref_atoms)

    traj_region.superpose(reference_region, 0)
    return mdtraj.rmsd(traj_region, reference_region, 0)


# --------------------------------------------------------------------------
# ddG estimation
# --------------------------------------------------------------------------

def compute_ddg(
    rmsd_wild: np.ndarray,
    rmsd_mutant: np.ndarray,
    folded_cutoff: float = FOLDED_CUTOFF,
    boltzmann_constant: float = BOLTZMANN_CONSTANT,
    temperature: float = TEMPERATURE,
):
    """
    Compute ddG = -k*T * log(n_wild_folded / n_mutant_folded) from frames
    where exactly one of the wild-type and mutant proteins is folded at a
    time.

    Sign convention: ddG < 0 -> wild-type folded more often than the
    mutant -> mutation is destabilizing. ddG > 0 -> mutant folded more
    often than the wild-type -> mutation is stabilizing.

    Returns
    -------
    valid_frames : np.ndarray
        Indices (into the input arrays) of frames where wild-type and
        mutant disagree on folding state (one folded, the other not).
    running_ddg : np.ndarray
        ddG computed cumulatively as valid frames accumulate, in the
        same order as `valid_frames`. Useful for checking convergence;
        entries are NaN until both wild-type and mutant have each been
        observed folded at least once.
    ddg : float or None
        Final ddG (kcal/mol) using all valid frames, 
        or None if wild-type or mutant is never observed folded in a valid frame (insufficient sampling to compute ddG).
    """
    if rmsd_wild.shape != rmsd_mutant.shape:
        raise ValueError("rmsd_wild and rmsd_mutant must have the same shape.")

    wild_folded = rmsd_wild < folded_cutoff
    mutant_folded = rmsd_mutant < folded_cutoff

    # Frames where wild-type and mutant disagree on folding state.
    valid_frames = np.where(wild_folded != mutant_folded)[0]

    if len(valid_frames) == 0:
        return valid_frames, np.array([]), None

    wild_folded_count = np.cumsum(wild_folded[valid_frames])
    mutant_folded_count = np.cumsum(mutant_folded[valid_frames])

    running_ddg = np.full(len(valid_frames), np.nan)
    both_observed = (wild_folded_count > 0) & (mutant_folded_count > 0)
    running_ddg[both_observed] = (
        -boltzmann_constant * temperature
        * np.log(wild_folded_count[both_observed] / mutant_folded_count[both_observed])
    )

    total_wild_folded = wild_folded_count[-1]
    total_mutant_folded = mutant_folded_count[-1]
    if total_wild_folded > 0 and total_mutant_folded > 0:
        ddg = -boltzmann_constant * temperature * np.log(
            total_wild_folded / total_mutant_folded
        )
    else:
        logger.warning(
            "Cannot compute ddG: wild-type folded in %d valid frames, "
            "mutant folded in %d. At least one of each is required.",
            total_wild_folded, total_mutant_folded,
        )
        ddg = None

    return valid_frames, running_ddg, ddg


# --------------------------------------------------------------------------
# Trajectory analysis
# --------------------------------------------------------------------------

@dataclass
class TrajectoryAnalysisResult:
    rmsd_wild: np.ndarray
    rmsd_mutant: np.ndarray
    valid_frames: np.ndarray
    running_ddg: np.ndarray
    ddg: Optional[float]
    n_frames: int
    n_valid_frames: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def analyze_trajectory(
    traj_file: Union[str, Path],
    topology_file: Union[str, Path],
    reference_file: Union[str, Path],
    folded_cutoff: float = FOLDED_CUTOFF,
    equilibration_frames: int = EQUILIBRATION_FRAMES,
) -> TrajectoryAnalysisResult:
    """
    Run the full ddG analysis on one trajectory.

    Assumes the system's protein residues are split into two equal halves
    by sequential (mdtraj) residue index: wild-type (first half) and
    mutant (second half), and that `reference_file` contains both copies
    (i.e. the same total residue count as the simulation topology, e.g.
    a MELD-prepared '<protein>_meld.pdb' reference rather than a plain
    single-chain PDB). Each half is superposed onto its own corresponding
    residues in the reference, using the reference's matching numbering.

    Parameters
    ----------
    traj_file : Path to the trajectory file (e.g. .dcd).
    topology_file : Path to the topology matching the trajectory (e.g. .pdb).
    reference_file : Path to the reference/native structure. Must contain
        both the wild-type and mutant halves, matching the trajectory's
        residue numbering (e.g. a '<protein>_meld.pdb' file, not the plain
        wild-type monomer PDB).
    folded_cutoff : RMSD (nm) below which a half counts as folded.
    equilibration_frames : Leading frames discarded before analysis.

    Returns
    -------
    TrajectoryAnalysisResult
    """
    traj_file, topology_file, reference_file = (
        Path(traj_file), Path(topology_file), Path(reference_file)
    )

    reference = mdtraj.load(str(reference_file), top=str(reference_file))
    full_traj = mdtraj.load(str(traj_file), top=str(topology_file))

    if equilibration_frames >= full_traj.n_frames:
        raise ValueError(
            f"equilibration_frames ({equilibration_frames}) >= total frames "
            f"({full_traj.n_frames}); no frames left to analyze."
        )
    traj = full_traj[equilibration_frames:]

    n_residues = count_residues(topology_file)
    if n_residues % 2 != 0:
        logger.warning(
            "Topology has an odd number of protein residues (%d); the last "
            "residue will be dropped from the mutant half.", n_residues
        )
    midpoint = n_residues // 2

    n_reference_residues = count_residues(reference_file)
    if n_reference_residues != n_residues:
        raise ValueError(
            f"Reference structure has {n_reference_residues} residues but "
            f"the topology has {n_residues} residues. The reference must "
            "contain both the wild-type and mutant halves (e.g. a "
            "'<protein>_meld.pdb' file), matching the trajectory's residue "
            "numbering -- not a plain single-chain wild-type PDB."
        )

    wild_selection = f"resid 0 to {midpoint - 1}"
    mutant_selection = f"resid {midpoint} to {n_residues - 1}"

    # Each half is superposed onto its own matching residues in the
    # reference (ref_selection defaults to traj_selection), since the
    # reference structure contains both copies with matching numbering.
    rmsd_wild = compute_rmsd(traj, reference, wild_selection)
    rmsd_mutant = compute_rmsd(traj, reference, mutant_selection)

    valid_frames, running_ddg, ddg = compute_ddg(
        rmsd_wild, rmsd_mutant, folded_cutoff
    )

    logger.info(
        "%s: %d/%d frames valid, ddG = %s kcal/mol",
        traj_file.name, len(valid_frames), traj.n_frames,
        f"{ddg:.2f}" if ddg is not None else "undetermined",
    )

    return TrajectoryAnalysisResult(
        rmsd_wild=rmsd_wild,
        rmsd_mutant=rmsd_mutant,
        valid_frames=valid_frames,
        running_ddg=running_ddg,
        ddg=ddg,
        n_frames=traj.n_frames,
        n_valid_frames=len(valid_frames),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traj_file", type=Path)
    parser.add_argument("topology_file", type=Path)
    parser.add_argument("reference_file", type=Path)
    parser.add_argument("--folded-cutoff", type=float, default=FOLDED_CUTOFF)
    parser.add_argument("--equilibration-frames", type=int, default=EQUILIBRATION_FRAMES)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    result = analyze_trajectory(
        traj_file=args.traj_file,
        topology_file=args.topology_file,
        reference_file=args.reference_file,
        folded_cutoff=args.folded_cutoff,
        equilibration_frames=args.equilibration_frames,
    )
    if result.ddg is not None:
        effect = "destabilizing" if result.ddg < 0 else "stabilizing"
        print(f"ddG = {result.ddg:.3f} kcal/mol ({effect})")
        print("(ddG < 0: mutation destabilizing; ddG > 0: mutation stabilizing)")
    else:
        print("ddG could not be determined (insufficient valid frames).")
    print(f"Valid frames: {result.n_valid_frames}/{result.n_frames}")


if __name__ == "__main__":
    main()