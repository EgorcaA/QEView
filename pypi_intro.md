<!-- ABOUT THE PROJECT -->

## Overview

This tool is designed to streamline the analysis and visualization of results obtained from **Quantum Espresso** simulations, including:

* *Band Structure*
* *Density of States (DOS, pDOS)*
* *Wannier projection using Wannier90*

By simplifying the interpretation of complex data, this package provides an efficient and user-friendly approach while remaining flexible for customization.

# Supported Systems
The package supports:

- **2D and 3D materials**
- **Ferromagnetic (FM) and Paramagnetic (PM) configurations**

# Key Features
* *Visualization*: Generate clear and insightful visual representations of your simulation results.
* *Predefined Visualization Methods*: A simple and intuitive interface for efficient workflow.
* *Wannier90 Hamiltonian Loading*: Enables *band structure interpolation and plotting* for enhanced analysis.

The package can proceed both 2D and 3D ferromagnetic(FM) and paramagnetic(PM) cases.

Features:
* *Visualization*: Generate clear and informative visualizations to better understand your simulation results.
* *Ready-to-use visualization methods*: Simple and intuitive interface for efficient workflow.
* Wannier90 hamiltonian loading for BS interpolation and plotting


API documentation 

Explore the `user guide` to quickly get up to speed with the tool.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Install

* qeschema
  ```sh
  pip install qeview
  ```


<!-- USAGE EXAMPLES -->
## Usage

Define you data document using:
```python
from qeview.qe_analyse_FM import qe_analyse_FM
import qeview.wannier_loader as wnldr 

calc = qe_analyse_FM('./', 'FeCl2')
```
Now you can access basic plots and properties
```python
calc.get_qe_kpathBS()

calc.plot_FullDOS(efrom=-10, eto=10)
calc.plot_pDOS('1', efrom=-10, eto=10, yfrom=-10)
calc.plot_BS(efrom=-5, eto=5)
  ```
