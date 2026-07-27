# Protein Folding Stability

This repository contains the source code accompanying the manuscript:

> **Predicting the Effect of Amino Acid Mutations on Protein Stability Using MELD-Accelerated Molecular Dynamics Simulations**

The repository provides the scripts used to set up MELD simulations and analyze the resulting trajectories to estimate protein stability changes upon mutation (ΔΔG).

---

# Repository Structure

```text
.
├── README.md
├── scripts/
│   ├── meld/
│   │   └── setup.py                 # Sets up and launches MELD simulations
│   └── analysis/
│       └── ddg.py                   # Computes ΔΔG from MELD trajectories
└── meld/                            # MELD software distribution used in this study
```

---

# Contents

## scripts/

This directory contains the MELD scripts developed for this work.

### scripts/meld/setup.py

Sets up and launches MELD simulations.

The script:

- prepares the MELD simulation environment
- loads the unfolded starting structures
- applies AlphaFold-derived distance restraints
- applies radius of gyration restraints
- generates the files required to start a MELD simulation

The script expects the corresponding protein structures and restraint files from the accompanying data repository found in zotero at https://zenodo.org/records/21512306.

---

### scripts/analysis/ddg.py

Computes mutation-induced changes in protein stability (ΔΔG) from MELD simulation trajectories.

The analysis estimates the folded populations of the wild-type and mutant proteins and computes the corresponding free-energy difference.

Run from the command line as:

```bash
python ddg.py path_to_trajectory0.dcd path_to_topology0.pdb path_to_reference.pdb
```

where:

- `trajectory0.dcd` is the trajectory of the lowest-temperature replica
- `topology0.pdb` is the corresponding topology
- `reference.pdb` is the reference folded structure

The resulting ΔΔG values correspond to those reported in the accompanying manuscript.

---

## meld/

This directory contains the complete source distribution of the MELD software used in this study.

The original MELD project is available at:

https://github.com/maccallumlab/meld

---

# Workflow

1. Obtain the protein structures and restraint files from the accompanying data repository at https://zenodo.org/records/21512306.
2. Use `scripts/meld/setup.py` to prepare a MELD simulation.
3. Run the simulation using MELD.
4. Analyze the resulting trajectory with `scripts/analysis/ddg.py` to compute ΔΔG.

---

# Citation

If you use this software, please cite the accompanying manuscript:

> **Predicting the Effect of Amino Acid Mutations on Protein Stability Using MELD-Accelerated Molecular Dynamics Simulations**

and MELD:

MacCallum JL, Perez A, Dill KA. *Determining protein structures by combining semireliable data with atomistic physical models by Bayesian inference*. Proceedings of the National Academy of Sciences. 2015.

---

# Contact

Please contact **Emiliano Brini** (exbsch@rit.edu) with questions regarding the software or simulations.