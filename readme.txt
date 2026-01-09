# RT-PINNs
## A Physics-Informed Neural Network for Rayleigh-Taylor Instability and Compressible Turbulent Mixing

## Overview
This repository contains a `PyTorch` implementation of a Physics-Informed Neural Network (PINN) designed to simulate the 2-D compressible Rayleigh-Taylor Instability (RTI).

Unlike traditional CFD solvers (e.g., Finite Difference/Volume) that rely on a mesh, this project utilizes a deep neural network to approximate the solution to the Navier-Stokes equations directly. The network is trained without labeled data, relying solely on a composite loss function derived from:
* **Governing Equations:** Compressible Euler/Navier-Stokes equations.
* **Initial Conditions:** A perturbed interface separating heavy and light fluids.
* **Boundary Conditions:** Periodic (y-axis) and No-Slip/Adiabatic walls (x-axis).

## Repository Structure
* `RTI_PINNs.ipynb`: The main Jupyter notebook containing the full pipeline (setup, training, analysis).
* `checkpoints/`: Directory where model weights are saved during training.
* `training_log.csv`: Log file tracking loss convergence (PDE, IC, BC) over epochs.
* `Figures/`: Generated plots for flow fields, mixing width, and loss history.

## Getting Started

### Prerequisites
The code is self-contained in a Jupyter Notebook. It depends on standard scientific Python libraries:
* `torch` (PyTorch)
* `numpy`
* `matplotlib`
* `tqdm` (for progress bars)

### How to Run
The entire workflow is contained within `RTI_PINNs.ipynb`.

1.  **Launch Jupyter:** Open the notebook in your preferred environment.
2.  **Configuration:**
    * The code automatically detects CUDA devices.
    * **Recommendation:** For practical training times (~100k epochs), use a GPU-accelerated environment.
3.  **Execution:** Run the cells sequentially. The notebook is structured as follows:
    * Environment Setup & Dependencies
    * PINN Architecture Definition
    * Physics Implementation (Loss Functions)
    * Main Training Loop (Inviscid)
    * Analysis & Figure Generation
    * Secondary Training Loop (Viscous/Navier-Stokes)

### Computational Resources
**Note:** Training a PINN for turbulent mixing is computationally intensive.
* **Laptop (CPU):** Valid for debugging and checking code syntax (very slow for training).
* **HPC (GPU):** Highly recommended. For Princeton users, use the Jupyter interactive sessions on:
    * [`myadroit.princeton.edu`](https://myadroit.princeton.edu)
    * [`mytiger.princeton.edu`](https://mytiger.princeton.edu)

## Methodology & Limitations
This implementation explores the "spectral bias" of neural networks in fluid dynamics. While the PINN successfully captures the linear growth phase and mean flow stratification, it acts as a low-pass filter, effectively smoothing out high-frequency turbulent structures. The resulting flow resembles a RANS (Reynolds-Averaged Navier-Stokes) solution.

## Acknowledgments
Comparison data generated using the [`pyranda`](https://github.com/LLNL/pyranda) high-order finite-difference code.
