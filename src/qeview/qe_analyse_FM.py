
from .qe_base import qe_analyse_base # Relative import

import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from  tqdm import tqdm
import os
import re

from .wannier_loader import Wannier_loader_FM

Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr


class qe_analyse_FM(qe_analyse_base):
    """
    Class for analyzing Quantum Espresso (QE) data for ferromagnetic (FM) systems.
    This class provides methods to read, process, and plot Density of States (DOS), projected Density of States (pDOS), 
    and band structure data from Quantum Espresso calculations. 
    It also includes functionality to interface with Wannier90 for band structure calculations.
    """
    
    def __init__(self, dir, name):
        super().__init__( dir, name)

        self.efermi = None

        self.eDOS = None
        self.dosup = None
        self.dosdn = None

        self.ePDOS = None
        self.pdos_up = None
        self.pdos_dn = None

        self.nbandsDFT = None
        self.BS_wannier_up = None
        self.BS_wannier_dn = None

    def get_full_DOS(self, filename='dos.dat', qe_dir='qe'):
        """
        Reads the Density of States (DOS) from a file and stores it in the instance variables:
            - eDOS: A list of energy values.
            - dosup: A list of DOS values for spin-up electrons.
            - dosdn: A list of DOS values for spin-down electrons.
            - efermi: The Fermi energy level.

        
        Parameters
        ----------
        filename : str, optional
            The name of the DOS file (default is 'dos.dat').
        qe_dir : str, optional
            The directory where the QE data is located (default is 'qe').

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the DOS file is not found.
        IOError
            If there is an error reading the DOS file.
        
        """
        self.eDOS = []
        self.dosup = []
        self.dosdn = []
        self.efermi = 0
        try:
            file_path = os.path.join(self.directory, qe_dir, filename)
            with open(file_path) as f:
                line = f.readline()
                self.efermi = float(re.search(r"EFermi =\s*(-?\d+\.\d*)\s*eV", line).group(1))
                for line in f:
                    if not line.strip():
                        continue
                    energy, edosup, edosdn, *_ = line.split()
                    self.eDOS.append(float(energy))
                    self.dosup.append(float(edosup))
                    self.dosdn.append(float(edosdn))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The DOS file '{file_path}' was not found: {e}") from e
        except IOError as e:
            raise IOError(f"Failed to read DOS from file '{file_path}': {e}") from e
        
        print(f'efermi {self.efermi:.2f}')
        self.eDOS = np.array(self.eDOS)
        self.dosup = np.array(self.dosup)
        self.dosdn = np.array(self.dosdn)


    

    def plot_FullDOS(self, efrom=-5, eto=5, yto=10, 
                     saveQ=False, picname='DOS.png', title='DOS', width=7, height=7./1.6, qe_dir='qe'):
        """
        Plots the Density of States (DOS) for a given energy range.

        Parameters
        ----------
        efrom : float, optional
            The starting energy value for the plot (default is -5).
        eto : float, optional
            The ending energy value for the plot (default is 5).
        yto : float, optional
            The maximum absolute value for the y-axis (default is 10).
        saveQ : bool, optional
            If True, the plot will be saved as an image file (default is False).
        picname : str, optional
            The name of the image file to save the plot (default is 'DOS.png').
        title : str, optional
            The title of the plot (default is 'DOS').
        width : float, optional
            The width of the plot (default is 7).
        height : float, optional
            The height of the plot (default is 7/1.6).
        qe_dir : str, optional
            The directory where Quantum Espresso files are located (default is 'qe').

        Returns
        -------
        None

        Notes
        -----
        - If the DOS data is not initialized, the method will call `get_full_DOS` to initialize it.
        - The x-axis represents the energy relative to the Fermi energy (E - E_f) in eV.
        - The y-axis represents the density of states in arbitrary units.
        """
          
        
        if self.eDOS is None or self.dosup is None or self.dosdn is None:
            print('Energies and DOS were not initialized. I run get_full_DOS')
            self.get_full_DOS(filename='dos.dat', qe_dir=qe_dir)
            
        fig, dd = plt.subplots() 
        
        dd.plot(self.eDOS - self.efermi, self.dosup, 
                    label="DOS up", color='red', linewidth=0.5)

        dd.plot(self.eDOS - self.efermi, -self.dosdn, 
                    label="DOS dn", color='blue', linewidth=0.5)

        plt.fill_between(
                x= self.eDOS-self.efermi, 
                y1=self.dosup,
                y2=-self.dosdn,
                color= "grey",
                alpha= 0.1)

        # locator = AutoMinorLocator()
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        dd.yaxis.set_major_locator(MultipleLocator(2))
        dd.xaxis.set_minor_locator(MultipleLocator(1))
        dd.xaxis.set_major_locator(MultipleLocator(2))

        dd.set_ylabel('Density of states')  # Add an x-label to the axes.
        dd.set_xlabel(r'$E-E_f$ [eV]')  # Add a y-label to the axes.
        dd.set_title(title)
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        
        dd.vlines(0, ymin=-yto*1.1, ymax=yto*1.1, colors='black', ls='-', alpha= 1.0, linewidth=1.0)
        
        min_lim = np.min(self.eDOS - self.efermi)
        max_lim = np.max(self.eDOS - self.efermi)
        dd.hlines(0, xmin=min_lim*1.1, xmax=max_lim*1.1, colors='black', ls='-', alpha= 1.0, linewidth=1.0)

        fig.set_figwidth(width)    
        fig.set_figheight(height)
        dd.set_ylim((-yto, yto))
        dd.set_xlim((efrom, eto))
        
        if saveQ:
            pic_path = os.path.join(self.directory, picname)
            plt.savefig(pic_path, dpi=200, bbox_inches='tight')

        plt.show()


    def get_pDOS(self, qe_dir='qe'):
        """
        Reads projected density of states (pDOS) data from files and organizes it by spin and orbital type.
        This method reads pDOS data from files in the specified directory, processes the data, and stores it in dictionaries
        for spin-up and spin-down states, categorized by orbital type (s, p, d).
        
        Parameters
        ----------
        qe_dir : str, optional
            The directory where the QE data is located (default is 'qe').

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the pDOS file is not found.
        IOError
            If there is an error reading the pDOS file.
        """
        def read_pdos(file, i, qe_dir='qe'):
            
            file_path = os.path.join(self.directory, qe_dir, str(file))
            data = np.loadtxt(file_path, skiprows=1)

            # Extract energy column (first column)
            e = data[:, 0]

            # Sum the selected PDOS columns
            pdos = data[:, i] + data[:, i+2]

            return e, pdos

        def list_pdos_files(path):
            """
            Generator function that lists PDOS (Projected Density of States) files in the given directory.
            Args:
                path (str): The directory path where PDOS files are located.
            Yields:
                tuple: A tuple containing the filename and a tuple of matched groups from the regex pattern.
                       The matched groups include:
                       - Atom number (str)
                       - Atom symbol (str)
                       - Wavefunction number (str)
                       - Wavefunction symbol (str)
            Raises:
                FileNotFoundError: If a file matching the pattern is not found.
            """
            found = False  # Flag to track if any matching file is found
            for f in os.listdir(path):
                
                if f.startswith( self.name + '.pdos_atm'):
                    match = re.search(
                        r"pdos_atm#(\d+)\((\w+)\)\_wfc#(\d+)\((\w+)\)", f)
                    if match:
                        found = True  # Set flag to True if a match is found
                        yield f, match.groups()
                    
            if not found:
                raise FileNotFoundError('No pdos files were found')

        self.pdos_up = {"s": dict(), "p": dict(), "d": dict()}
        self.pdos_dn = {"s": dict(), "p": dict(), "d": dict()}
        
        qqe_dir = os.path.join(self.directory, qe_dir)
        for file, info in list_pdos_files(qqe_dir):
            print(file)
            atom_number,  _, _, orbital_type = info
            
            self.ePDOS, pdos_up = read_pdos(file, 1, qe_dir)#spinup
            self.pdos_up[orbital_type].update({atom_number: pdos_up})

            _, pdos_dn = read_pdos(file, 2, qe_dir)#spindown
            self.pdos_dn[orbital_type].update({atom_number: pdos_dn})
            

    def plot_pDOS(self, element="1", efermi=None, efrom=-15, eto=15, yfrom=-10, yto=10,
                  saveQ=False, picname='pDOS.png', title='pDOS', width=7, height=7./1.6, qe_dir='qe'):
        
        """
        Plots the projected Density of States (pDOS) for a given element and energy range.

        Parameters
        ----------
        element : str, optional
            The element number to plot the pDOS for (default is "1").
        efermi : float, optional
            The Fermi energy level (default is None).
        efrom : float, optional
            The starting energy value for the plot (default is -15).
        eto : float, optional
            The ending energy value for the plot (default is 15).
        yfrom : float, optional
            The starting y-axis value for the plot (default is -10).
        yto : float, optional
            The ending y-axis value for the plot (default is 10).
        saveQ : bool, optional
            If True, the plot will be saved to a file (default is False).
        picname : str, optional
            The name of the file to save the plot (default is 'pDOS.png').
        title : str, optional
            The title of the plot (default is 'pDOS').
        width : float, optional
            The width of the plot (default is 7).
        height : float, optional
            The height of the plot (default is 7/1.6).
        qe_dir : str, optional
            The directory where the QE data is located (default is 'qe').

        Returns
        -------
        None

        Raises
        ------
        Exception
            If `efermi` is not defined and `get_full_DOS` has not been run.
        Exception
            If no pDOS files are found for the specified element.

        """
        
        if self.ePDOS is None or self.pdos_up is None or self.pdos_dn is None:
            print('Energies and pDOS were not initialized. I run get_pDOS')
            self.get_pDOS(qe_dir)
        if efermi is None:
            if self.efermi is not None:
                efermi = self.efermi
            else:
                raise Exception('efermi is not defined. Run get_full_DOS(filename=dos.dat)')
        if title == 'pDOS':
            title = element +" pDOS"

        fig, dd = plt.subplots()  

        ########################### UP spin
        atom_pdos = {"s": None, "p": None, "d": None}
        atom_tdos = np.zeros((len(self.pdos_up['s']['1'])))
        
        found_element = False
        for orbital_type in atom_pdos.keys():
            if str(element) in self.pdos_up[orbital_type].keys():
                atom_pdos[orbital_type] = self.pdos_up[orbital_type][str(element)]
                atom_tdos += self.pdos_up[orbital_type][str(element)]
                found_element = True
        if not found_element:
            raise Exception('No pdos files were found for this element')

        atom_pdos_index = self.ePDOS - efermi

        dd.plot(self.ePDOS-efermi, atom_tdos, color='green', label='TDOS '+element, linewidth=0.8, linestyle='dashed') 

        if atom_pdos['s'] is not None:
            dd.plot(atom_pdos_index, atom_pdos['s'], 
                    label="s DOS", color='c', linewidth=0.5)

        if atom_pdos['p'] is not None:
            dd.plot(atom_pdos_index, atom_pdos['p'], 
                    label="p DOS", color='red', linewidth=0.5)

        if atom_pdos['d'] is not None:
            dd.plot(atom_pdos_index, atom_pdos['d'], 
                    label="d DOS", color='blue', linewidth=0.5)
        plt.fill_between(
                x= self.ePDOS-efermi, 
                y1=atom_tdos, 
                # where= (-1 < t)&(t < 1),
                color= "grey",
                alpha= 0.1)


        ########################### DOWN spin
        atom_pdos = {"s": None, "p": None, "d": None}
        atom_tdos = np.zeros((len(self.pdos_dn['s']['1'])))
        
        for orbital_type in atom_pdos.keys():
            if str(element) in self.pdos_dn[orbital_type].keys():
                atom_pdos[orbital_type] = self.pdos_dn[orbital_type][str(element)]
                atom_tdos += self.pdos_dn[orbital_type][str(element)]


        dd.plot(self.ePDOS-efermi, -atom_tdos, color='green', label='TDOS '+element, linewidth=0.8, linestyle='dashed') 
        
        if atom_pdos['s'] is not None:
            dd.plot(atom_pdos_index, -atom_pdos['s'], 
                    label="s DOS", color='c', linewidth=0.5)

        if atom_pdos['p'] is not None:
            dd.plot(atom_pdos_index, -atom_pdos['p'], 
                    label="p DOS", color='red', linewidth=0.5)

        if atom_pdos['d'] is not None:
            dd.plot(atom_pdos_index, -atom_pdos['d'], 
                    label="d DOS", color='blue', linewidth=0.5)
            
        plt.fill_between(
                x= self.ePDOS-efermi, 
                y1=-atom_tdos, 
                color= "grey",
                alpha= 0.1)


        locator = AutoMinorLocator()
        dd.yaxis.set_minor_locator(locator)
        dd.xaxis.set_minor_locator(locator)

        dd.set_ylabel('Density of states')  # Add an x-label to the axes.
        dd.set_xlabel(r'$E-E_f$ [eV]')  # Add a y-label to the axes.
        dd.set_title(title)
        dd.legend(loc='upper right')  # Add a legend.
        
        dd.vlines(0, ymin=0, ymax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        
        fig.set_figwidth(width)     
        fig.set_figheight(height)
        dd.set_ylim((yfrom, yto))
        dd.set_xlim((efrom, eto))
        
        if saveQ:
            pic_path = os.path.join(self.directory, element+'_'+picname)
            plt.savefig(pic_path, dpi=200, bbox_inches='tight')


        plt.show()



    def get_band_structure(self, bands_up_name=None, bands_dn_name=None, qe_dir='qe'):
        """
        Retrieve the band structure from specified or fallback files.
        This method attempts to load the band structure data from user-specified
        filenames or from a set of fallback filenames. The method searches for the
        files in the specified Quantum Espresso (QE) directory.
     
        Parameters
        ----------
        bands_up_name : str, optional
            The name of the spin up band structure file (default is None).
        bands_dn_name : str, optional
            The name of the spin down band structure file (default is None).
        qe_dir : str, optional
            The directory where the QE data is located (default is 'qe').

        Returns
        -------
        None
            This method updates the instance variables `hDFT_up`, `hDFT_dn` and `nbandsDFT` 
            with the band structure data and the number of bands, respectively.

        Raises
        ------
        FileNotFoundError
            If the band structure file is not found.

        """
        
        potential_files = [
            (bands_up_name, bands_dn_name),  # User-specified filenames
            ('bands1.dat.gnu', 'bands2.dat.gnu'),  # First fallback
            ('bands_up.dat.gnu', 'bands_dn.dat.gnu')  # Second fallback
        ]

        for up_name, dn_name in potential_files:
            if up_name and dn_name:  # Ensure both filenames are available
                file_path_up = os.path.join(self.directory, qe_dir, up_name)
                file_path_dn = os.path.join(self.directory, qe_dir, dn_name)

                try:
                    self.hDFT_up = self.get_spin_BS(file_path_up)
                    self.hDFT_dn = self.get_spin_BS(file_path_dn)
                    self.nbandsDFT = self.hDFT_up.shape[0]
                    return  # Exit once successful
                except FileNotFoundError:
                    continue  # Try the next fallback

        # If all fallbacks fail, raise an error
        raise FileNotFoundError("No valid band structure files found in 'qe' directory.")



    def print_bands_range(self, band_from=None, band_to=None, efermi=None):
        """
        This method prints the Fermi energy and the energy range for each band in the specified range. 
        The energy range is printed both in absolute terms and relative to the Fermi energy.

        Parameters
        ----------
        band_from : int, optional
            The starting band index (inclusive) (defaults to 0).
        band_to : int, optional
            The ending band index (exclusive) (defaults to the total number of bands if not provided).
        efermi : float, optional
                The Fermi energy level (default is None).
        Returns
        -------
        None

        Raises
        ------
        Exception
            If no bands were parsed.
        Exception
            If efermi is not defined in class and not provided.
        
        """
        if self.nbandsDFT is None:
            raise Exception('No bands were parsed. Run get_band_structure()')
        if efermi is None:
            if self.efermi is not None:
                efermi = self.efermi
            else:
                raise Exception('efermi is not defined. Run get_full_DOS(filename=dos.dat)')

        if band_from is None:
            band_from = 0
        if band_to is None:
            band_to = self.nbandsDFT

        print(f'efermi {efermi:.2f}')
        print("-------------SPIN UP---------------")
        for band_num in range(band_from,band_to):
            print(f'band {band_num+1} eV from  {min(self.hDFT_up[band_num, : ,1]) :.2f} to  {max(self.hDFT_up[band_num, : ,1]) :.2f} \
                eV-eF from  {min(self.hDFT_up[band_num, : ,1]) -efermi :.2f} to  {max(self.hDFT_up[band_num, : ,1]) - efermi:.2f}' )
        print("-------------SPIN DN---------------")
        for band_num in range(band_from,band_to):
            print(f'band {band_num+1} eV from  {min(self.hDFT_dn[band_num, : ,1]) :.2f} to  {max(self.hDFT_dn[band_num, : ,1]) :.2f} \
                eV-eF from  {min(self.hDFT_dn[band_num, : ,1]) - efermi :.2f} to  {max(self.hDFT_dn[band_num, : ,1]) - efermi:.2f}' )


    def plot_BS(self, efrom=-15, eto=15, efermi=None,
                saveQ=False, picname='BS.png', title='BS', width=7, height=7/1.6):
        """
        Plots the band structure (BS) of the material with spin-up and spin-down components.

        Parameters
        -----------
        efrom : float, optional
            The lower energy limit for the plot. Default is -15.
        eto : float, optional
            The upper energy limit for the plot. Default is 15.
        efermi : float, optional
            The Fermi energy level. If not provided, it will use self.efermi.
        saveQ : bool, optional
            If True, the plot will be saved as an image file. Default is False.
        picname : str, optional
            The name of the image file to save the plot. Default is 'BS.png'.
        title : str, optional
            The title of the plot. Default is 'BS'.
        width : float, optional
            The width of the plot in inches. Default is 7.
        height : float, optional
            The height of the plot in inches. Default is 7/1.6.

        Raises
        -------
        Exception
            If no bands were parsed, or if high symmetry points were not parsed, or if efermi is not defined.
        
        Notes
        ------
        - Ensure that `get_band_structure()` and `get_sym_points()` are run before calling this method.
        - Ensure that `get_full_DOS(filename=dos.dat)` is run if efermi is not provided.

        """
        
        if self.nbandsDFT is None:
            raise Exception('No bands were parsed. Run get_band_structure()')
        if self.HighSymPointsNames is None:
            raise Exception('High Symmetry Points were not parsed from band.in file. Run get_sym_points()')
        if efermi is None:
            if self.efermi is not None:
                efermi = self.efermi
            else:
                raise Exception('efermi is not defined. Run get_full_DOS(filename=dos.dat)')

        fig, dd = plt.subplots() 
        
        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists
        assert len(normal_ticks) > 1, "not enough High Sym Points"
        assert len(normal_ticks) == len(label_ticks), "Length of High Sym Points distanclistes and labels lists should be the same"

        if self.nbandsDFT is not None:
            for band in range(self.nbandsDFT):
                for spin, data, color, label in zip(
                    ["up", "down"], 
                    [self.hDFT_up, self.hDFT_dn], 
                    ["red", "blue"], 
                    ["up", "down"]
                ):
                    assert len(data[band, :, 0]) == len(data[band, :, 1]), \
                        f'len of qe kpath {data[band, :, 0]} is not equal to calculated BS {data[band, :, 1]}'

                    dd.plot(
                        data[band, :, 0],
                        data[band, :, 1] - efermi,
                        color=color,
                        linewidth=0.7,
                        alpha=1.0,
                        label=label if band == 0 else None,  # Avoid redundant labels
                    )


        dd.set_ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        plt.xticks(normal_ticks, label_ticks)
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        dd.axhline(y=0, ls='--', color='k')
        dd.set_title(title)

        plt.grid(axis='x')
        plt.xlim(normal_ticks[0], normal_ticks[-1])
        plt.ylim(efrom, eto)

        fig.set_figwidth(width)    
        fig.set_figheight(height)  

        if saveQ:
            pic_path = os.path.join(self.directory, picname)
            plt.savefig(pic_path, dpi=200, bbox_inches='tight')

        plt.show()

            
    # Wannier90 interface 
    def load_wannier(self, kpath_filename, kpaths_dir='kpaths',
                     hr_up_name=None, hr_dn_name=None, wannier_dir='wannier'):
        """
        Load Wannier data and k-path information.

        Parameters
        -----------
        kpath_filename : str
            The filename of the k-path file to be loaded.
        kpaths_dir : str, optional
            The directory where the k-path files are located. Default is 'kpaths'.
        hr_up_name : str, optional
            The filename of the Wannier Hamiltonian for the spin-up component. Default is None.
        hr_dn_name : str, optional
            The filename of the Wannier Hamiltonian for the spin-down component. Default is None.
        wannier_dir : str, optional
            The directory where the Wannier files are located. Default is 'wannier'.
        
        Returns
        --------
        None

        """
        
        self.wannier = Wannier_loader_FM(hr_up_name, hr_dn_name, wannier_dir)

        os.makedirs(os.path.abspath(kpaths_dir), exist_ok=True)
        kpath_file = os.path.join(os.path.abspath(kpaths_dir), kpath_filename)

        self.wannier.load_kpath(kpath_file)
        self.BS_wannier_up = self.wannier.get_wannier_BS(spin=0)
        self.BS_wannier_dn = self.wannier.get_wannier_BS(spin=1)
        


    def plot_wannier_BS(self, efrom=-15, eto=15, saveQ=False, picname='BS_wannier.png', title='BS Wannier', width=7, height=7/1.6):
        """
        Plots the Wannier band structure.

        Parameters
        ----------
        efrom : float, optional
            The lower energy limit for the plot (default is -15).
        eto : float, optional
            The upper energy limit for the plot (default is 15).
        saveQ : bool, optional
            If True, the plot will be saved to a file (default is False).
        picname : str, optional
            The name of the file to save the plot (default is 'BS_wannier.png').
        title : str, optional
            The title of the plot (default is 'BS Wannier').
        width : float, optional
            The width of the plot in inches (default is 7).
        height : float, optional
            The height of the plot in inches (default is 7/1.6).

        Returns
        -------
        None

        Raises
        ------
        Exception
            If no bands were parsed from QE or Wannier.
            If `efermi` is not defined.
            If High Symmetry Points were not parsed from band.in file.
        """
        
        if self.BS_wannier_up is None and self.BS_wannier_dn is None and self.nbandsDFT is None:
            raise Exception('No bands were parsed from qe or wannier.')
        if self.efermi is None:
            raise Exception('efermi is not defined. Run get_full_DOS(filename=dos.dat)')
        if self.HighSymPointsNames is None:
            raise Exception('High Symmetry Points were not parsed from band.in file. Run get_sym_points()')
        

        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists
        assert len(normal_ticks) > 1, "not enough High Sym Points"
        assert len(normal_ticks) == len(label_ticks), "Length of High Sym Points distanclistes and labels lists should be the same"

        fig, dd = plt.subplots() 

        if self.nbandsDFT is not None:
            for band in range(self.nbandsDFT):
                for spin, data, color, label in zip(
                    ["up", "down"], 
                    [self.hDFT_up, self.hDFT_dn], 
                    ["red", "blue"], 
                    ["up", "down"]
                ):
                    assert len(data[band, :, 0]) == len(data[band, :, 1]), \
                        f'len of qe kpath {data[band, :, 0]} is not equal to calculated BS {data[band, :, 1]}'

                    dd.plot(
                        data[band, :, 0],
                        data[band, :, 1] - self.efermi,
                        color=color,
                        linewidth=0.7,
                        alpha=1.0,
                        label=label if band == 0 else None,  # Avoid redundant labels
                    )

        nwa = self.BS_wannier_dn.shape[1]
        assert nwa > 0, "no wannier bands"

        if self.BS_wannier_up is not None and self.BS_wannier_dn is not None:
            for band in range(nwa):
                for spin, data, color, label in zip(
                    ["up", "down"], 
                    [self.BS_wannier_up, self.BS_wannier_dn], 
                    ["red", "blue"], 
                    ["Wannier up", "Wannier down"]
                ):
                    assert len(self.wannier.kpath_dists_qe) == len(data[ :, band]), \
                        f'len of wannier kpath {len(self.wannier.kpath_dists_qe)} is not equal to calculated BS {len(data[ :, band])}'
                    dd.plot(
                        self.wannier.kpath_dists_qe,
                        data[ :, band] - self.efermi,
                        color=color,
                        linewidth=3,
                        alpha=0.3,
                        label=label if band == 0 else None,  # Avoid redundant labels
                    )
        


        dd.set_ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        plt.xticks(normal_ticks, label_ticks)
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(axis='x')
        dd.axhline(y=0, ls='--', color='k')
        plt.xlim(normal_ticks[0], normal_ticks[-1])
        plt.ylim(efrom, eto)

        dd.set_title(title)

        fig.set_figwidth(width)   
        fig.set_figheight(height) 

        if saveQ:
            pic_path = os.path.join(self.directory, picname)
            plt.savefig(pic_path, dpi=200, bbox_inches='tight')

        plt.show()
