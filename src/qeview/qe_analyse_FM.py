
from .qe_base import qe_analyse_base # Relative import

import numpy as np
import numpy.linalg as LA
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from  tqdm import tqdm
import os
import re

from .wannier_loader import Wannier_loader_FM

Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)



class qe_analyse_FM(qe_analyse_base):
    '''
    Class for analyzing Quantum Espresso (QE) data with a focus on ferromagnetic (FM) systems.
    This class inherits from `qe_analyse_base` and provides methods to read and analyze density of states (DOS),
    projected density of states (pDOS), and band structure (BS) data from QE calculations. It also includes
    methods to plot these data.
    Methods:
        __init__(self, dir, name):
            Initializes the qe_analyse_FM instance with the specified directory and name.
        get_full_DOS(self):
            Raises IOError if the DOS file does not exist or cannot be opened.
        get_hr(self):
            Reads the spin-polarized band structure data from files and stores it in the instance variables.
            If the files cannot be found or opened, an error message is printed.
        plot_FullDOS(self, efrom=-5, eto=5, saveQ=False, picname='DOS'):
            Plots the full density of states (DOS) for spin-up and spin-down electrons.
            Optionally saves the plot to a file.
        get_pDOS(self):
            Reads the projected density of states (pDOS) data from files and stores it in the instance variables.
        plot_pDOS(self, element="1", efrom=None, eto=None, yfrom=None, yto=None):
            Plots the projected density of states (pDOS) for a specified element.
            Optionally sets the energy and y-axis limits for the plot.
        print_bands_range(self, band_from=None, band_to=None):
            Prints the energy range of the specified bands for spin-up and spin-down electrons.
        plot_BS(self, efrom=None, eto=None):
            Plots the band structure (BS) for spin-up and spin-down electrons.
            Optionally sets the energy limits for the plot.
        load_wannier(self, kpath_filename='kpath_qe2.dat', wannier_hr=''):
            Loads the Wannier90 data and k-path information for band structure calculations.
        plot_wannier_BS(self, efrom=None, eto=None):
            Plots the Wannier-interpolated band structure (BS) for spin-up and spin-down electrons.
            Optionally sets the energy limits for the plot.
    '''
    
    def __init__(self, dir, name):
        super().__init__( dir, name)


    def get_full_DOS(self):
        """
        Reads the density of states (DOS) data from a file and stores it in the instance variables.
        This method initializes the following instance variables:
        - eDOS: A list of energy values.
        - dosup: A list of DOS values for spin-up electrons.
        - dosdn: A list of DOS values for spin-down electrons.
        - efermi: The Fermi energy level.
        The method attempts to read the DOS data from a file located at `self.directory + "qe/dos.dat"`.
        If the file is successfully read, the energy values, spin-up DOS values, and spin-down DOS values
        are stored in the respective instance variables as numpy arrays. The Fermi energy level is also
        extracted from the file.
        If the file cannot be found or opened, an error message is printed.
        Raises:
            IOError: If the DOS file does not exist or cannot be opened.
        """
        
        self.eDOS = []
        self.dosup = []
        self.dosdn = []
        self.efermi = 0
        try:
            with open(self.directory + "qe/dos.dat") as f:
                line = f.readline()
                self.efermi = float(re.search(r"EFermi =\s*(-?\d+\.\d*)\s*eV", line).group(1))
                for line in f:
                    if not line.strip():
                        continue
                    energy, edosup, edosdn, *_ = line.split()
                    self.eDOS.append(float(energy))
                    self.dosup.append(float(edosup))
                    self.dosdn.append(float(edosdn))
        except IOError:
            print("Error: DOS file does not appear to exist.")
        print(f'efermi {self.efermi:.2f}')
        self.eDOS = np.array(self.eDOS)
        self.dosup = np.array(self.dosup)
        self.dosdn = np.array(self.dosdn)



    def get_hr(self):
        """
        Retrieves and processes spin band structure data from specified files.

        This method attempts to read spin band structure data from two files
        ('bands1.dat.gnu' and 'bands2.dat.gnu') in the 'qe' subdirectory of the
        specified directory. If an error occurs (e.g., files not found), it falls
        back to reading from 'bands_up.dat.gnu' and 'bands_dn.dat.gnu'.

        Attributes:
            hDFT_up (numpy.ndarray): Spin-up band structure data.
            hDFT_dn (numpy.ndarray): Spin-down band structure data.
            nbandsDFT (int): Number of bands in the DFT calculation.

        Raises:
            Exception: If both sets of files are not found or cannot be read.
        """
        try:
            self.hDFT_up = self.get_spin_BS(self.directory +'qe/bands1.dat.gnu')
            self.hDFT_dn = self.get_spin_BS(self.directory +'qe/bands2.dat.gnu')
            self.nbandsDFT = self.hDFT_up.shape[0]
        except Exception as e:
            self.hDFT_up = self.get_spin_BS(self.directory +'qe/bands_up.dat.gnu')
            self.hDFT_dn = self.get_spin_BS(self.directory +'qe/bands_dn.dat.gnu')
            self.nbandsDFT = self.hDFT_up.shape[0]


    def plot_FullDOS(self, efrom=-5, eto=5, saveQ=False, picname='DOS.png'):
        """
        Plots the full Density of States (DOS) with spin polarization.
        Parameters:
        -----------
        efrom : float, optional
            The lower bound of the energy range to plot (default is -5).
        eto : float, optional
            The upper bound of the energy range to plot (default is 5).
        saveQ : bool, optional
            If True, the plot will be saved as an image file (default is False).
        picname : str, optional
            The name of the file to save the plot if saveQ is True (default is 'DOS').
        
        Notes:
        ------
        - The plot displays the spin-polarized Density of States (DOS).
        - The x-axis represents the energy relative to the Fermi energy (E - E_f) in eV.
        - The y-axis represents the density of states.
        - A vertical dashed line is drawn at E - E_f = 0.
        - The plot can be saved as an image file if saveQ is set to True.
        """
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
        dd.set_title("Spinpolarized DOS")
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        
        dd.vlines(0, ymin=-30, ymax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        dd.hlines(0, xmin=-30, xmax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        
        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        dd.set_ylim((-10, 10))
        dd.set_xlim((efrom, eto))
        
        if saveQ:
            plt.savefig('./'+ picname, dpi=200, bbox_inches='tight')
        plt.show()


    def get_pDOS(self):
        """
        Reads projected density of states (pDOS) data from files and organizes it by spin and orbital type.
        This method reads pDOS data from files in the specified directory, processes the data, and stores it in dictionaries
        for spin-up and spin-down states, categorized by orbital type (s, p, d).
        The method performs the following steps:
        1. Defines a helper function `read_pdos` to read pDOS data from a file and sum the relevant columns.
        2. Defines a helper function `list_pdos_files` to list all pDOS files in the specified directory that match the naming pattern.
        3. Initializes dictionaries `pdos_up` and `pdos_dn` to store pDOS data for spin-up and spin-down states, respectively.
        4. Iterates over the pDOS files, reads the data using `read_pdos`, and updates the dictionaries with the processed data.
        Raises:
            FileNotFoundError: If a file does not match the expected naming pattern.
        Attributes:
            pdos_up (dict): Dictionary to store spin-up pDOS data, categorized by orbital type.
            pdos_dn (dict): Dictionary to store spin-down pDOS data, categorized by orbital type.
            ePDOS (pd.Series): Energy values corresponding to the pDOS data.
        """
        
        def read_pdos(file, i):
            """
            Reads projected density of states (PDOS) data from a specified file.

            Args:
                file (str): The name of the file containing the PDOS data.
                i (int): The index of the column to be used for PDOS calculation.

            Returns:
                tuple: A tuple containing:
                - e (pd.Series): The energy values.
                - pdos (pd.Series): The summed PDOS values from the specified columns.
            """
            df = pd.read_csv(self.directory +'qe/'+ str(file), sep='\s+', skiprows=[0], header=None)
            e, pdos = df.iloc[:, 0], df.iloc[:, [i,i+2]].sum(axis=1)
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
            for f in os.listdir(path):
                
                if f.startswith( self.name + '.pdos_atm'):
                    match = re.search(
                        r"pdos_atm#(\d+)\((\w+)\)\_wfc#(\d+)\((\w+)\)", f)
                    if not match:
                        raise FileNotFoundError
                    yield f, match.groups()

        self.pdos_up = {"s": dict(), "p": dict(), "d": dict()}
        self.pdos_dn = {"s": dict(), "p": dict(), "d": dict()}
        for file, info in list_pdos_files(self.directory + 'qe/'):
            atom_number,  _, _, orbital_type = info
            
            self.ePDOS, pdos_up = read_pdos(file, 1)#spinup
            self.pdos_up[orbital_type].update({atom_number: pdos_up})

            _, pdos_dn = read_pdos(file, 2)#spindown
            self.pdos_dn[orbital_type].update({atom_number: pdos_dn})


    def plot_pDOS(self, element="1", efrom=None, eto=None, yfrom=None, yto=None):
        """
        Plots the projected Density of States (pDOS) for a given element.
        Parameters:
        -----------
        element : str, optional
            The element for which the pDOS is to be plotted. Default is "1".
        efrom : float, optional
            The lower bound of the energy range for the plot. Default is -15.
        eto : float, optional
            The upper bound of the energy range for the plot. Default is 15.
        yfrom : float, optional
            The lower bound of the y-axis for the plot. Default is -10.
        yto : float, optional
            The upper bound of the y-axis for the plot. Default is -yfrom.
        """
        if efrom is None:
            efrom = -15
        if eto is None:
            eto =15
        if yfrom is None:
            yfrom = -10
        if yto is None:
            yto = -yfrom

        fig, dd = plt.subplots()  

        ########################### UP spin
        atom_pdos = {"s": None, "p": None, "d": None}
        atom_tdos = np.zeros((len(self.pdos_up['s']['1'])))
        
        for orbital_type in atom_pdos.keys():
            if str(element) in self.pdos_up[orbital_type].keys():
                atom_pdos[orbital_type] = self.pdos_up[orbital_type][str(element)]
                atom_tdos += self.pdos_up[orbital_type][str(element)]

        atom_pdos = pd.DataFrame(atom_pdos)
        atom_pdos.index = self.ePDOS -self.efermi

        dd.plot(self.ePDOS-self.efermi, atom_tdos, color='green', label='TDOS '+element, linewidth=0.8, linestyle='dashed') 

        if atom_pdos['s'][0] is not None:
            dd.plot(atom_pdos.index, atom_pdos['s'], 
                    label="s DOS", color='c', linewidth=0.5)

        if atom_pdos['p'][0] is not None:
            dd.plot(atom_pdos.index, atom_pdos['p'], 
                    label="p DOS", color='red', linewidth=0.5)

        if atom_pdos['d'][0] is not None:
            dd.plot(atom_pdos.index, atom_pdos['d'], 
                    label="d DOS", color='blue', linewidth=0.5)
        plt.fill_between(
                x= self.ePDOS-self.efermi, 
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

        atom_pdos = pd.DataFrame(atom_pdos)
        atom_pdos.index = self.ePDOS -self.efermi

        dd.plot(self.ePDOS-self.efermi, -atom_tdos, color='green', label='TDOS '+element, linewidth=0.8, linestyle='dashed') 
        
        if atom_pdos['s'][0] is not None:
            dd.plot(atom_pdos.index, -atom_pdos['s'], 
                    label="s DOS", color='c', linewidth=0.5)

        if atom_pdos['p'][0] is not None:
            dd.plot(atom_pdos.index, -atom_pdos['p'], 
                    label="p DOS", color='red', linewidth=0.5)

        if atom_pdos['d'][0] is not None:
            dd.plot(atom_pdos.index, -atom_pdos['d'], 
                    label="d DOS", color='blue', linewidth=0.5)
            
        plt.fill_between(
                x= self.ePDOS-self.efermi, 
                y1=-atom_tdos, 
                color= "grey",
                alpha= 0.1)


        locator = AutoMinorLocator()
        dd.yaxis.set_minor_locator(locator)
        dd.xaxis.set_minor_locator(locator)

        dd.set_ylabel('Density of states')  # Add an x-label to the axes.
        dd.set_xlabel(r'$E-E_f$ [eV]')  # Add a y-label to the axes.
        dd.set_title(element +" pDOS")
        dd.legend()  # Add a legend.
        
        dd.vlines(0, ymin=0, ymax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        dd.set_ylim((yfrom, yto))
        dd.set_xlim((efrom, eto))
        # plt.savefig('./pics/'+ element+'_DOS.png', dpi=200)
        # plt.savefig('./2pub/pics/pDOS.png', dpi=200, bbox_inches='tight')

        plt.show()


    def print_bands_range(self, band_from=None, band_to=None):
        """
        This method prints the Fermi energy and the energy range for each band in the specified range. 
        The energy range is printed both in absolute terms and relative to the Fermi energy.

        Parameters:
        band_from (int, optional): The starting band index (inclusive). Defaults to 0 if not provided.
        band_to (int, optional): The ending band index (exclusive). Defaults to the total number of bands if not provided.
        """
        if band_from is None:
            band_from = 0
        if band_to is None:
            band_to = self.nbandsDFT

        print(f'efermi {self.efermi:.2f}')
        print("-------------SPIN UP---------------")
        for band_num in range(band_from,band_to):
            print(f'band {band_num+1} eV from  {min(self.hDFT_up[band_num, : ,1]) :.2f} to  {max(self.hDFT_up[band_num, : ,1]) :.2f} \
                eV-eF from  {min(self.hDFT_up[band_num, : ,1]) -self.efermi :.2f} to  {max(self.hDFT_up[band_num, : ,1]) - self.efermi:.2f}' )
        print("-------------SPIN DN---------------")
        for band_num in range(band_from,band_to):
            print(f'band {band_num+1} eV from  {min(self.hDFT_dn[band_num, : ,1]) :.2f} to  {max(self.hDFT_dn[band_num, : ,1]) :.2f} \
                eV-eF from  {min(self.hDFT_dn[band_num, : ,1]) - self.efermi :.2f} to  {max(self.hDFT_dn[band_num, : ,1]) - self.efermi:.2f}' )



    def plot_BS(self, efrom=None, eto=None):
        """
        Plots the band structure with spin-up and spin-down components.
        Parameters:
        -----------
        efrom : float, optional
            The lower limit of the energy range to plot. Default is -15.
        eto : float, optional
            The upper limit of the energy range to plot. Default is 15.
        Notes:
        ------
        - The function uses matplotlib to create the plot.
        - The x-axis represents the high symmetry points in the Brillouin zone.
        - The y-axis represents the energy relative to the Fermi level (E - $E_f$).
        - Spin-up bands are plotted in red and spin-down bands are plotted in blue.
        - The function automatically adjusts the figure size and adds grid lines and labels.
        - The Fermi level is indicated by a dashed horizontal line at y=0.
        """
        if efrom is None:
            efrom = -15
        if eto is None:
            eto =15
        
        fig, dd = plt.subplots() 
        
        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists
        
        for band in range(self.nbandsDFT):
            if band == 0:
                dd.plot(self.hDFT_up[band, : ,0], 
                        self.hDFT_up[band, : , 1] - self.efermi, label='up', color='red', linewidth=0.7,
                            alpha=1.0)

                dd.plot(self.hDFT_dn[band, : ,0], 
                        self.hDFT_dn[band, : , 1] - self.efermi, label='down', color='blue', linewidth=0.7,
                            alpha=1.0)
            else:
                dd.plot(self.hDFT_up[band, : ,0], 
                        self.hDFT_up[band, : , 1] - self.efermi,  color='red', linewidth=0.7,
                        alpha=1.0)

                dd.plot(self.hDFT_dn[band, : ,0], 
                        self.hDFT_dn[band, : , 1] - self.efermi,  color='blue', linewidth=0.7,
                        alpha=1.0)


        dd.set_ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        plt.xticks(normal_ticks, label_ticks)
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(axis='x')
        dd.axhline(y=0, ls='--', color='k')
        plt.xlim(normal_ticks[0], normal_ticks[-1])
        plt.ylim(efrom, eto)

        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        #plt.savefig('./2pub/pics/BS.png', dpi=200, bbox_inches='tight')

        plt.show()

            
    # Wannier90 interface 
    def load_wannier(self, kpath_filename='kpath_qe2.dat', wannier_hr=''):
        """
        Load Wannier data and calculate band structures for spin up and spin down.

        Parameters:
        kpath_filename (str): The filename of the k-path data. Default is 'kpath_qe2.dat'.
        wannier_hr (str): The filename of the Wannier Hamiltonian. Default is an empty string 
        (files should be named hrup.dat and hrdn.dat in wannier folder).

        Returns:
        None
        """
        self.wannier = Wannier_loader_FM(self.directory, '')
        self.wannier.load_kpath('./kpaths/'+ kpath_filename)
        self.BS_wannier_dn = self.wannier.get_wannier_BS(spin=1)
        self.BS_wannier_up = self.wannier.get_wannier_BS(spin=0)


    def plot_wannier_BS(self, efrom=None, eto=None):
        """
        Plots the Wannier band structure along with the DFT band structure.
        
        Parameters:
        efrom (float, optional): The lower limit of the energy range to plot. Defaults to -15.
        eto (float, optional): The upper limit of the energy range to plot. Defaults to 15.
        
        This function plots the band structure for both spin-up and spin-down states using data from
        Wannier and DFT calculations. The plot includes labels for high symmetry points and a legend
        indicating the spin direction. The Fermi level is set to zero on the y-axis.
        The plot is displayed using matplotlib and can be saved by uncommenting the savefig line.
        """
        if efrom is None:
            efrom = -15
        if eto is None:
            eto =15

        nwa = self.BS_wannier_dn.shape[1]

        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists

        fig, dd = plt.subplots()  # Create a figure containing a single axes.
        for band in range(self.nbandsDFT):
            if band == 0:
                dd.plot(self.hDFT_up[band, : ,0], 
                        self.hDFT_up[band, : , 1] - self.efermi, label='up', color='red', linewidth=0.7,
                            alpha=1.0)

                dd.plot(self.hDFT_dn[band, : ,0], 
                        self.hDFT_dn[band, : , 1] - self.efermi, label='down', color='blue', linewidth=0.7,
                            alpha=1.0)
            else:
                dd.plot(self.hDFT_up[band, : ,0], 
                        self.hDFT_up[band, : , 1] - self.efermi,  color='red', linewidth=0.7,
                        alpha=1.0)

                dd.plot(self.hDFT_dn[band, : ,0], 
                        self.hDFT_dn[band, : , 1] - self.efermi,  color='blue', linewidth=0.7,
                        alpha=1.0)


        for band in range(nwa):
            if band == 0:
                
                dd.plot(self.wannier.kpath_dists_qe,
                        self.BS_wannier_up[ : , band] - self.efermi , label='up', color='r', alpha=0.5, linewidth=3)

                dd.plot(self.wannier.kpath_dists_qe,
                        self.BS_wannier_dn[ : , band] - self.efermi , label='down', color='b', alpha=0.5, linewidth=3)
                
            else:
                
                dd.plot(self.wannier.kpath_dists_qe,
                        self.BS_wannier_up[ : , band] - self.efermi , color='r', alpha=0.3, linewidth=3)

                dd.plot(self.wannier.kpath_dists_qe,
                        self.BS_wannier_dn[ : , band] - self.efermi ,  color='b', alpha=0.3, linewidth=3)


        dd.set_ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        plt.xticks(normal_ticks, label_ticks)
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(axis='x')
        dd.axhline(y=0, ls='--', color='k')
        plt.xlim(normal_ticks[0], normal_ticks[-1])
        plt.ylim(efrom, eto)

        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        # plt.savefig('./2pub/pics/BS_wannier.png', dpi=200, bbox_inches='tight')

        plt.show()
