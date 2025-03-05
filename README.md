<!-- ABOUT THE PROJECT -->
## About The Project

This tool is designed to help you analyze and visualize the results obtained from Quantum Espresso simulations (Band structure, density of states DOS, pDOS) and Wannier projection using wannier90. It simplifies the process of interpreting complex data, remaining accessibly simple to adjust the code for your needs. 

The package can proceed both 2D and 3D ferromagnetic(FM) and paramagnetic(PM) cases.

Features:
* *Visualization*: Generate clear and informative visualizations to better understand your simulation results.
* *Ready-to-use visualization methods*: Simple and intuitive interface for efficient workflow.
* Wannier90 hamiltonian loading for BS interpolation and plotting

[API documentation](https://qeview.readthedocs.io/en/latest/)

Explore the `user guide` to quickly get up to speed with the tool.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Install

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

<img src="pics/spinDOS.png" alt="spinDOS_pic" width="200"/>
<img src="pics/spinBS.png" alt="spinBS_pic" width="200"/>
<img src="pics/interpolated_bs.png" alt="interpolated_bs" width="200"/>
<img src="pics/pDOS.png" alt="pDOS" width="200"/>

and also plot 2D fermi surfaces

<img src="pics/2D_band_plot.png" alt="2D_band_plot" width="400"/>

and 3D fermi surfaces

<img src="pics/3D_band_plot.png" alt="3D_band_plot" width="400"/>


<!-- CONTACT -->
## Contact

Egor Agapov -  agapov.em@phystech.edu
Project Link: [https://github.com/EgorcaA/qe_helper](https://github.com/EgorcaA/qe_helper)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

