Physics-Informed Boltzmann Kinetic Learning for Subgrid-Scale Turbulence Modeling (PINO-LBM)
![Conference](https://img.shields.io/badge/Conference-FMFP_2026-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch) ![Physics-Informed](https://img.shields.io/badge/Physics--Informed-Losses-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

This repository contains the complete PyTorch implementation of the unified physics-informed neural-kinetic framework presented in our paper:

"Physics-Informed Boltzmann Kinetic Learning for Subgrid-Scale Turbulence Modeling in the Lattice Boltzmann Framework" Presented at the Proceedings of the 13th International and 53rd National Conference on Fluid Mechanics and Fluid Power (FMFP), December 10-12, 2026, NIT Rourkela, India.
Paper ID: FMFP2026-XX-YYY

Authors
Anjan Mahapatra¹ (anjanmahapatra10@gmail.com)
Nikhil Raj² (nikhilraj24@iitk.ac.in)
¹ Department of Mechanical Engineering, Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram 600127, India
² Department of Mechanical Engineering, Indian Institute of Technology (IIT), Kanpur 208016, India

Abstract
Numerical simulations of turbulent flows at high Reynolds numbers remain computationally prohibitive due to the enormous degrees of freedom required. While machine learning has emerged as a transformative tool, existing approaches within the Lattice Boltzmann Method (LBM) remain fragmented across three disjoint paradigms: collision operator learning, subgrid-scale stress prediction, and neural operator-based fast-forward prediction.

This paper presents a unified physics-informed neural-kinetic framework integrating these paradigms within a single LBM architecture. The framework employs a U-shaped Neural Operator (UNO) with spectral convolutions and physics-informed losses for moment-matching, equivariance, and subgrid-scale (SGS) energy transfer.

Our complete PyTorch implementation is provided here. Numerical validation on homogeneous isotropic turbulence demonstrates superior performance over the classical Smagorinsky model. The neural operator achieves 1.5% relative error in density prediction, accurately captures the $k^{-5/3}$ Kolmogorov scaling, reproduces anomalous scaling (intermittency), and preserves physical backscatter with 15-20% negative energy transfer events.

Methodology
1. Lattice Boltzmann Method (LBM) Preliminaries
The mesoscopic Lattice Boltzmann Method characterizes the fluid system through particle distribution functions $f_i(\mathbf{x}, t)$ representing the probability of finding particles at position $\mathbf{x}$ and time $t$ moving with discrete velocity $\mathbf{c}_i$.

For the D2Q9 velocity stencil, the discrete Boltzmann evolution equation is governed by:

$$f_i(\mathbf{x} + \mathbf{c}_i\Delta t, t + \Delta t) = f_i(\mathbf{x}, t) + \Omega_i(\mathbf{x}, t)$$

where the collision operator $\Omega_i$ models the local redistribution of particles. The macroscopic density $\rho$ and momentum $\rho \mathbf{u}$ are obtained as the zeroth and first moments:

$$\rho = \sum_{i} f_i, \quad \rho\mathbf{u} = \sum_{i} f_i \mathbf{c}_i$$

2. U-Shaped Neural Operator (UNO) Architecture
Instead of evaluating intermediate collision and streaming steps explicitly, our model learns a continuous, discretization-invariant mapping:

$$\mathcal{G} : {f_i(\cdot, t_0)}{i=0}^8 \mapsto {f_i(\cdot, t_0 + n\Delta t)}{i=0}^8$$

This maps the initial distribution functions directly to a future state $n$ steps ahead, bypassing the fundamental time-step constraints of conventional LBM.

UNO Block Structure: Features 3 encoder-decoder blocks with skip connections. Each block combines local convolutions with a spectral convolution layer that captures long-range dependencies by transforming features to Fourier space, applying learned complex weights, and transforming back.
Discretization Invariance: The architecture enables zero-shot super-resolution/inference on finer grids without retraining.
3. Physics-Informed Loss Functions
The model enforces physical constraints during training by optimizing a composite loss function:

$$\mathcal{L} = \mathcal{L}{\text{data}} + \lambda{\text{mom}}\mathcal{L}{\text{mom}} + \lambda{\text{equiv}}\mathcal{L}{\text{equiv}} + \lambda{\Pi}\mathcal{L}_{\Pi}$$

Loss Component	Mathematical Formulation	Physical Significance
Data Fidelity Loss	$\mathcal{L}{\text{data}} = \frac{1}{9}\sum{i=0}^{8} | f_{i, \text{pred}} - f_{i, \text{true}} |^2$	Minimizes deviation from the true distribution functions.
Moment-Matching Loss	$\mathcal{L}{\text{mom}} = |\rho{\text{pred}} - \rho_{\text{true}}|^2 + |(\rho\mathbf{u}){\text{pred}} - (\rho\mathbf{u}){\text{true}}|^2$	Enforces the fundamental conservation of mass and momentum.
Equivariance Loss	$\mathcal{L}_{\text{equiv}} = |\mathcal{G}(\mathbf{R} \cdot f) - \mathbf{R} \cdot \mathcal{G}(f)|^2$	Guarantees physical consistency under dihedral symmetries (90° rotations and reflections $\mathbf{R}$).
SGS Energy Transfer	$\mathcal{L}{\Pi} = |\Pi{\text{pred}} - \Pi_{\text{true}}|^2, \quad \text{where } \Pi = -\tau_{ij}S_{ij}$	Captures energy exchange between resolved and unresolved scales, including backscatter.
Key Results & Discussion
Density Prediction Accuracy: Relative error is approximately 1.5% with absolute error on the order of $10^{-3}$.
Kolmogorov Spectral Scaling: Accurately reproduces the theoretical $k^{-5/3}$ scaling in the inertial range of the energy spectrum, avoiding the excessive dissipation of traditional eddy-viscosity models.
Intermittency & Structure Functions: Correctly reproduces non-Gaussian statistics and anomalous scaling of higher-order (1st, 2nd, and 3rd order) velocity structure functions: $$S^p(r) = \langle|\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})|^p\rangle$$
Physical Backscatter Preservation: Successfully captures the negative tail of the SGS energy transfer distribution, correctly predicting 15-20% negative energy transfer events (backscatter), which are fundamentally absent in classical Smagorinsky formulations.
Repository Structure
├── data/                  # LBM simulations & dataset generation
├── models/
│   ├── uno.py             # U-shaped Neural Operator (UNO) model definition
│   └── spectral_layers.py # Fourier/Spectral convolution layers
├── utils/
│   ├── lbm_solver.py      # Reference D2Q9 LBM simulator
│   └── physics_losses.py  # Physics-informed loss functions (Moment, Equivariance, SGS)
├── train.py               # Training loop with Adam optimizer and LR scheduling
├── evaluate.py            # Evaluation suite, error analysis, and visualization
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
Getting Started
Installation
Clone this repository:
git clone https://github.com/yourusername/PINO-LBM.git
cd PINO-LBM
Set up your python environment and install requirements:
pip install -r requirements.txt
Requirements
Python $\ge 3.8$
PyTorch $\ge 1.10$
NumPy, SciPy, Matplotlib
Workflow
1. Generate LBM Datasets
Generate physical reference data of 2D homogeneous isotropic turbulence using the D2Q9 solver with random forcing:

python utils/lbm_solver.py --grid_size 64 --num_snapshots 1200 --save_path data/
2. Model Training
Train the physics-informed neural operator for 30 epochs with default parameters and scheduler:

python train.py --epochs 30 --lr 1e-3 --lambda_mom 1.0 --lambda_equiv 0.1 --lambda_sgs 0.1
3. Evaluation
Generate figures (energy spectrum, error maps, structure functions, vorticity fields) and compare results against DNS and Smagorinsky models:

python evaluate.py --checkpoint checkpoints/best_model.pt
Citation
If you use this framework or implementation in your work, please cite our FMFP 2026 paper:

@inproceedings{mahapatra2026physics,
  title={Physics-Informed Boltzmann Kinetic Learning for Subgrid-Scale Turbulence Modeling in the Lattice Boltzmann Framework},
  author={Mahapatra, Anjan and Raj, Nikhil},
  booktitle={Proceedings of the 13th International and 53rd National Conference on Fluid Mechanics and Fluid Power (FMFP)},
  number={FMFP2026-XX-YYY},
  year={2026},
  address={NIT Rourkela, India}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.
