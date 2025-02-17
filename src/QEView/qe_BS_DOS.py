import numpy as np
import pandas as pd
import numpy.linalg as LA
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import qeschema
import math
import pickle 
from  tqdm import tqdm
import os
import re

import wannier_loader

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



class qe_analyse_spinpolarized():
    '''
    Class for analyzing spin-polarized data from Quantum Espresso calculations.
    Attributes:
        directory (str): Directory containing the Quantum Espresso output files.
        name (str): Name of the material/system being analyzed.
        eDOS (numpy.ndarray): Energy values for the density of states (DOS).
        dosup (numpy.ndarray): DOS values for spin-up electrons.
        dosdn (numpy.ndarray): DOS values for spin-down electrons.
        efermi (float): Fermi energy level.
        acell (numpy.ndarray): Real-space lattice vectors.
        bcell (numpy.ndarray): Reciprocal-space lattice vectors.
        pos (tuple): Atomic positions in Cartesian and fractional coordinates.
        HighSymPointsNames (list): Names of high symmetry points.
        HighSymPointsDists (list): Distances between high symmetry points.
        HighSymPointsCoords (list): Coordinates of high symmetry points.
        hDFT_up (numpy.ndarray): Spin-up band structure data.
        hDFT_dn (numpy.ndarray): Spin-down band structure data.
        nbandsDFT (int): Number of bands in the DFT calculation.
        pdos_up (dict): Projected DOS for spin-up electrons.
        pdos_dn (dict): Projected DOS for spin-down electrons.
        ePDOS (numpy.ndarray): Energy values for the projected DOS.
        wannier (Wannier_loader_FM): Wannier90 interface for the system.
        BS_wannier_up (numpy.ndarray): Wannier90 band structure for spin-up electrons.
        BS_wannier_dn (numpy.ndarray): Wannier90 band structure for spin-down electrons.
    Methods:
        __init__(dir, name): Initializes the analysis with the given directory and name.
        get_full_DOS(): Reads and stores the full density of states (DOS) data.
        get_crystell_str(): Reads and calculates crystal structure information.
        get_sym_points(): Reads and stores high symmetry points from the band structure input file.
        get_spin_BS(path): Reads spin-polarized band structure data from a file.
        plot_FullDOS(saveQ=False, picname='DOS'): Plots the full density of states (DOS).
        get_pDOS(): Reads and stores the projected density of states (pDOS) data.
        plot_pDOS(element="1", efrom=None, eto=None, yfrom=None, yto=None): Plots the projected density of states (pDOS) for a given element.
        print_bands_range(band_from=None, band_to=None): Prints the energy range of specified bands.
        get_hr(): Reads and stores the spin-polarized band structure data.
        plot_BS(efrom=None, eto=None): Plots the band structure.
        get_qe_kpathBS(printQ=False): Generates and writes the k-path for Quantum Espresso band structure calculations.
        get_integer_kpath(N_points_direction=10, num_points_betweens=5, filename='kpath_integer.dat'): Generates and writes an integer k-path for Wannier90 calculations.
        load_wannier(): Loads Wannier90 data and calculates the band structure.
        plot_wannier_BS(efrom=None, eto=None): Plots the Wannier90 band structure.
    '''

    def __init__(self, dir, name):
        self.directory = dir # './'
        self.name = name # 'CrTe2'

        self.get_full_DOS()
        self.get_crystell_str()
        self.get_hr()
        self.get_sym_points()

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


    def get_crystell_str(self):
        """
        Reads the crystal structure information from a Quantum Espresso output file and calculates
        various properties of the crystal lattice.

        This function performs the following steps:
        1. Reads the cell parameters from the 'data-file-schema.xml' file.
        2. Calculates the unit cell volume.
        3. Computes the reciprocal-space vectors.
        4. Prints the lattice constant (alat) and the reciprocal-space vectors in both Cartesian coordinates
           and in units of 2π/alat.
        5. Prints the real-space vectors in both Cartesian coordinates and in units of alat.
        6. Retrieves and prints the atomic positions in both Cartesian coordinates (in units of alat) and
           fractional coordinates.

        Raises:
            IOError: If the 'data-file-schema.xml' file does not exist in the specified directory.

        Prints:
            - Unit cell volume in Angstrom^3.
            - Reciprocal-space vectors in Angstrom^-1 and in units of 2π/alat.
            - Real-space vectors in Angstrom and in units of alat.
            - Atomic positions in Cartesian coordinates (in units of alat) and fractional coordinates.
        """
        pw_document = qeschema.PwDocument()
        try:
            with open(self.directory+ "qe/data-file-schema.xml") as fin:
                pass
        except IOError:
            print("Error: data-file-schema.xml file does not appear to exist.")

        pw_document.read(self.directory+ "qe/data-file-schema.xml")
        acell = np.array(pw_document.get_cell_parameters())*Bohr2Ang
        self.alat = pw_document.to_dict()['qes:espresso']['input']['atomic_structure']['@alat']*Bohr2Ang
        V = LA.det(acell)
        print(f'Unit Cell Volume:   {V:.4f}  (Ang^3)')
        b1 = 2*np.pi*np.cross(acell[1], acell[2])/V
        b2 = 2*np.pi*np.cross(acell[2], acell[0])/V
        b3 = 2*np.pi*np.cross(acell[0], acell[1])/V
        self.bcell = np.array([b1, b2, b3])
        self.acell = acell
        # print('Reciprocal-Space Vectors (Ang^-1)')
        # with printoptions(precision=10, suppress=True):
        #     print(b)
        print(f'alat {self.alat:.4f}')
        print('Reciprocal-Space Vectors cart (Ang^-1)')
        with printoptions(precision=10, suppress=True):
            print(self.bcell)

        print('Reciprocal-Space Vectors cart (2 pi / alat)')
        with printoptions(precision=10, suppress=True):
            print(self.bcell/ (2*np.pi/self.alat))


        print('Real-Space Vectors cart (Ang)')
        with printoptions(precision=10, suppress=True):
            print(acell)
        print('Real-Space Vectors cart (alat)')
        with printoptions(precision=10, suppress=True):
            print(acell/self.alat)

        print('\n\n positions cart (alat)')
        self.pos = pw_document.get_atomic_positions()
        with printoptions(precision=10, suppress=True):
            print(self.pos[0])
            print(np.array(self.pos[1])*Bohr2Ang/self.alat)

        print('positions (frac or crystal)')
        with printoptions(precision=10, suppress=True):
            print(  np.array(self.pos[1])*Bohr2Ang @ LA.inv(acell) )


    def get_sym_points(self):
        """
        Reads high symmetry points from the 'band.in' file and calculates their distances and coordinates.
        Populates the HighSymPointsNames, HighSymPointsDists, and HighSymPointsCoords attributes.
        """
        self.HighSymPointsNames = []
        self.HighSymPointsDists = []
        self.HighSymPointsCoords = []
        
        try:
            with open(self.directory + "qe/band.in") as fin:
                # Skip lines until 'K_POINTS' is found
                while True:
                    file_row = fin.readline()
                    if file_row.split()[0] == 'K_POINTS':
                        break
                
                n_strings = int(fin.readline())
                k_string = fin.readline().split()
                Letter_prev = k_string[5]
                dist = 0.0
                k_prev = np.array(list(map(float, k_string[:3])))
                
                self.HighSymPointsNames.append(Letter_prev)
                self.HighSymPointsDists.append(dist)
                self.HighSymPointsCoords.append(k_prev)

                for _ in range(n_strings - 1):
                    line = fin.readline()
                    k_string = line.split()
                    Letter_new = k_string[5]
                    k_new = np.array(list(map(float, k_string[:3])))
                    delta_k = k_new - k_prev
                    dist += LA.norm(self.bcell.T @ delta_k) / (2. * np.pi / self.alat)
                    k_prev = k_new
                    
                    self.HighSymPointsNames.append(Letter_new)
                    self.HighSymPointsDists.append(dist)
                    self.HighSymPointsCoords.append(k_prev)
        
        except IOError:
            print("Error: band.in file does not appear to exist.")


    def get_qe_kpathBS(self, filename="kpath_qe2.txt", saveQ=True, points_per_unit=10):
        """
        Generate a k-path with points between high-symmetry points, proportional to their distances.
        Writes to a file and returns the k-path as a list of formatted strings.
        
        :param filename: Name of the output file
        :param points_per_unit: Number of points per unit distance
        """
        kpath = []
        kpath_coords = []
        kpath_dists = []
        
        for i in range(len(self.HighSymPointsNames) - 1):
            name1, name2 = self.HighSymPointsNames[i], self.HighSymPointsNames[i+1]
            coord1, coord2 = np.array(self.HighSymPointsCoords[i]), np.array(self.HighSymPointsCoords[i+1])
            delta_k = self.HighSymPointsCoords[i+1] - self.HighSymPointsCoords[i]
            dist = LA.norm(self.bcell.T@delta_k) / (2.*np.pi / self.alat)
            
            num_points = max(int(dist * points_per_unit), 2)  # Ensure at least 2 points per segment
            segment_points = np.linspace(coord1, coord2, num_points, endpoint=False)
            dists_segment = np.linspace(self.HighSymPointsDists[i], self.HighSymPointsDists[i+1], num_points, endpoint=False)

            for j, point in enumerate(segment_points):
                if j == 0 and name1:  # Label only the first point of a segment
                    kpath.append(f"{name1} {point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {dists_segment[j]:.8f}")
                else:
                    kpath.append(f". {point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {dists_segment[j]:.8f}")
                kpath_coords.append(point)
                kpath_dists.append(dists_segment[j])
                print(kpath[-1])

        # Add the last high-symmetry point
        kpath.append(f"{self.HighSymPointsNames[-1]} {self.HighSymPointsCoords[-1][0]:.8f} {self.HighSymPointsCoords[-1][1]:.8f} {self.HighSymPointsCoords[-1][2]:.8f} {self.HighSymPointsDists[-1]:.8f}")
        kpath_coords.append(self.HighSymPointsCoords[-1])
        kpath_dists.append(self.HighSymPointsDists[-1])
        print(kpath[-1])
        
        # Write to file
        if saveQ:
            with open('kpaths/' + filename, "w") as f:
                f.write("\n".join(kpath))
        
        return np.array(kpath_coords), np.array(kpath_dists)



    def get_integer_kpath(self, N_points_direction=10, num_points_betweens=5, 
                        filename='kpath_integer_new.dat', saveQ=False):
        """
        Generates a k-path with integer coordinates for high symmetry points.
        Parameters:
        -----------
        N_points_direction : int, optional
            The number of points in each direction (default is 10).
        num_points_betweens : int or list of int, optional
            The number of points between each pair of high symmetry points. If an integer is provided, 
            the same number of points is used between all pairs. If a list is provided, it should have 
            length NHSP-1, where NHSP is the number of high symmetry points (default is 5).
        filename : str, optional
            The name of the file to save the k-path (default is 'kpath_integer.dat').
        Returns:
        --------
        kpath_return : list of numpy arrays
            List of k-points with integer coordinates.
        kpath_draw_path_return : list of floats
            List of distances corresponding to each k-point in units 2 pi / alat.
        Notes:
        ------
        The function prints the k-path lines and the high symmetry points names during execution.
        """
        
        NHSP = len(self.HighSymPointsCoords)
        kpath_return = []
        kpath_draw_path_return = []
        kpath_lines = []

        Letter_prev = self.HighSymPointsNames[0]
        dist = 0.0
        k_prev = self.HighSymPointsCoords[0]
        
        for HSP_ind in range(1, NHSP):
            Letter_new = self.HighSymPointsNames[HSP_ind]
            k_new = self.HighSymPointsCoords[HSP_ind]
            delta_k = k_new - k_prev
            
            num_points_between = num_points_betweens if isinstance(num_points_betweens, int) else num_points_betweens[HSP_ind-1]
            
            for point in range(num_points_between + (HSP_ind == NHSP-1)):
                k_to_write = k_prev + delta_k / num_points_between * point
                k_to_write = np.array(list(map(int, k_to_write * N_points_direction)))
                
                Letter_to_write = Letter_prev if point == 0 else (Letter_new if HSP_ind == NHSP-1 and point == num_points_between else '.')
                
                kpath_lines.append(f'{Letter_to_write} {k_to_write[0]:.0f} {k_to_write[1]:.0f} {k_to_write[2]:.0f} \t {dist :.8f}')
                kpath_return.append(k_to_write)
                kpath_draw_path_return.append(dist )
                print(kpath_lines[-1])
                dist += LA.norm(self.bcell.T @ delta_k / num_points_between)/ (2.*np.pi / self.alat)
            
            k_prev = k_new[:]
            Letter_prev = Letter_new
        
        if saveQ:
            with open("./kpaths/" + filename, "w") as fout:
                fout.writelines(kpath_lines)
                
        return np.array(kpath_return), np.array(kpath_draw_path_return)
    

    @staticmethod
    def get_spin_BS(path):
        hr_fact_data = []
        with open(path) as f:
                band = 0
                hr_fact_data.append([])

                for line in f:
                    
                    if line == ' \n':
                        hr_fact_data[-1] = np.array(hr_fact_data[-1])
                        hr_fact_data.append([])
                        band+=1
                    else:
                        hr_string = line.split()
                        hr_fact_data[-1].append(np.array([
                            float(hr_string[0]), float(hr_string[1]), 
                        ]))
                        
        hr_fact_data = np.array(hr_fact_data[:-1])  
        return hr_fact_data
    

    def plot_FullDOS(self, saveQ=False, picname='DOS'):
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
        dd.set_xlim((-5, 5))
        
        if saveQ:
            plt.savefig('./'+ picname, dpi=200, bbox_inches='tight')
        plt.show()


    def get_pDOS(self):
        
        def read_pdos(file, i):
            df = pd.read_csv(self.directory +'qe/'+ str(file), sep='\s+', skiprows=[0], header=None)
            e, pdos = df.iloc[:, 0], df.iloc[:, [i,i+2]].sum(axis=1)
            return e, pdos

        def list_pdos_files(path):
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
        if efrom is None:
            efrom = -15
        if eto is None:
            eto =15
        if yfrom is None:
            yfrom = -10
        if yto is None:
            yto = -yfrom
        # plt.figure(figsize= (40, 20))
        fig, dd = plt.subplots()  # Create a figure containing a single axes.

        ########################### UP spin
        # print(self.pdos_up)
        atom_pdos = {"s": None, "p": None, "d": None}
        atom_tdos = np.zeros((len(self.pdos_up['s']['1'])))
        
        for orbital_type in atom_pdos.keys():
            if str(element) in self.pdos_up[orbital_type].keys():
                atom_pdos[orbital_type] = self.pdos_up[orbital_type][str(element)]
                atom_tdos += self.pdos_up[orbital_type][str(element)]

        atom_pdos = pd.DataFrame(atom_pdos)
        atom_pdos.index = self.ePDOS -self.efermi

        dd.plot(self.ePDOS-self.efermi, atom_tdos, color='green', label='TDOS '+element, linewidth=0.8, linestyle='dashed') 

        # for orbital_type in atom_pdos.keys():
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
        
        # for orbital_type in atom_pdos.keys():
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
                # where= (-1 < t)&(t < 1),
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
        # fig.set_figwidth(12)     #  ширина и
        # fig.set_figheight(6)    #  высота "Figure"
        # dd.set_ylim((-10, 10))
        # dd.set_xlim((-7, 3))
        # plt.savefig(element+'_DOS.png', dpi=1000)
        # plt.savefig('./pics/'+ element+'_DOS.png', dpi=200)
        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        dd.set_ylim((yfrom, yto))
        dd.set_xlim((efrom, eto))
        # plt.savefig('./2pub/pics/pDOS.png', dpi=200, bbox_inches='tight')

        plt.show()


    def print_bands_range(self, band_from=None, band_to=None):
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


    def get_hr(self):
        try:
            self.hDFT_up = self.get_spin_BS(self.directory +'qe/bands1.dat.gnu')
            self.hDFT_dn = self.get_spin_BS(self.directory +'qe/bands2.dat.gnu')
            self.nbandsDFT = self.hDFT_up.shape[0]
        except Exception as e:
            self.hDFT_up = self.get_spin_BS(self.directory +'qe/bands_up.dat.gnu')
            self.hDFT_dn = self.get_spin_BS(self.directory +'qe/bands_dn.dat.gnu')
            self.nbandsDFT = self.hDFT_up.shape[0]


    def plot_BS(self, efrom=None, eto=None):
        if efrom is None:
            efrom = -15
        if eto is None:
            eto =15
        
        fig, dd = plt.subplots() 
        
        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists
        # print(normal_ticks)
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
        # dd.set_xlabel('rho')  # Add a y-label to the axes.
        # dd.set_title("pk/p from density")
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



    # def get_integer_kpath(self, N_points_direction=10, num_points_betweens=5, filename='kpath_integer.dat'):
        
    #     # N_points = 10
    #     kmax = self.hDFT_up[0, -1 ,0]
    #     qe2wan =  self.HighSymPointsDists[-1]/kmax
        
    #     NHSP = len(self.HighSymPointsCoords)
    #     # num_points_betweens = [12, 4 ,8] #2D G_M_K_G
    #     #num_points_betweens = [9, 3, 6,9,9, 3, 6 ] #3D
    #     kpath_return = []
    #     kpath_draw_path_return = []

    #     with open("./kpaths/"+ filename, "w") as fout:
        
    #         Letter_prev = self.HighSymPointsNames[0]
    #         dist = 0.0
    #         k_prev = self.HighSymPointsCoords[0]
    #         print(Letter_prev)

    #         for HSP_ind in range(1, NHSP):
                
    #             Letter_new = self.HighSymPointsNames[HSP_ind]
    #             k_new = self.HighSymPointsCoords[HSP_ind]
                
    #             delta_k = k_new - k_prev
                
    #             if type(num_points_betweens) == int:
    #                 num_points_between = num_points_betweens
    #             else:
    #                 num_points_between = num_points_betweens[HSP_ind-1]
                
    #             for point in range(num_points_between + (HSP_ind==NHSP-1)):
    #                 k_to_write = k_prev +   delta_k/(num_points_between)*(point) 
    #                 k_to_write =     np.array(list(map(int,   k_to_write*N_points_direction)))  
    #                 # print(k_to_write)
    #                 if point == 0:
    #                     Letter_to_write =  Letter_prev
    #                 elif (HSP_ind == NHSP-1 and point == num_points_between):
    #                     Letter_to_write =  Letter_new
    #                 else:
    #                     Letter_to_write = '.'
    #                 fout.write( 
    #                     f'{Letter_to_write} {k_to_write[0]:.0f}  {k_to_write[1]:.0f} {k_to_write[2]:.0f}  \t {dist/qe2wan:.8f} \n'
    #                 )
    #                 kpath_return.append(k_to_write)
    #                 kpath_draw_path_return.append(dist/qe2wan)

    #                 dist += LA.norm(self.bcell.T@delta_k/(num_points_between))
                
    #             print(Letter_new)
    #             k_prev = k_new[:]
    #             Letter_prev = Letter_new 
    #     return kpath_return, kpath_draw_path_return

    # def get_integer_kpath(self, N_points_direction=10, num_points_between=5):
    #     # N_points = 10
    #     kmax = self.hDFT_up[0, -1 ,0]
    #     qe2wan =  self.HighSymPointsDists[-1]/kmax
        
    #     NHSP = len(self.HighSymPointsCoords)
    #     with open("./kpaths/kpath_integer.dat", "w") as fout:
        
    #         Letter_prev = self.HighSymPointsNames[0]
    #         dist = 0.0
    #         k_prev = self.HighSymPointsCoords[0]
    #         print(Letter_prev)

    #         for HSP_ind in range(1, NHSP):
                
    #             Letter_new = self.HighSymPointsNames[HSP_ind]
    #             k_new = self.HighSymPointsCoords[HSP_ind]
                
    #             delta_k = k_new - k_prev
                
                
    #             for point in range(num_points_between + (HSP_ind==NHSP-1)):
    #                 k_to_write = k_prev +   delta_k/(num_points_between)*point 
    #                 k_to_write =     np.array(list(map(int,   k_to_write*N_points_direction)))  
    #                 # print(k_to_write)
    #                 if point == 0:
    #                     Letter_to_write =  Letter_prev
    #                 elif (HSP_ind == NHSP-1 and point == num_points_between):
    #                     Letter_to_write =  Letter_new
    #                 else:
    #                     Letter_to_write = '.'
    #                 fout.write( 
    #                     f'{Letter_to_write} {k_to_write[0]:.0f}  {k_to_write[1]:.0f} {k_to_write[2]:.0f}  \t {dist/qe2wan:.8f} \n'
    #                 )


    #                 dist += LA.norm(self.bcell.T@delta_k/(num_points_between))
                
    #             print(Letter_new)
    #             k_prev = k_new[:]
    #             Letter_prev = Letter_new 
                  
                        
    # Wannier90 interface 
    def load_wannier(self, kpath_filename='kpath_qe2.dat'):
        self.wannier = wannier_loader.Wannier_loader_FM(self.directory, 'aa')
        self.wannier.load_kpath('./kpaths/'+ kpath_filename)
        self.BS_wannier_dn = self.wannier.get_wannier_BS(spin=1)
        self.BS_wannier_up = self.wannier.get_wannier_BS(spin=0)


    def plot_wannier_BS(self, efrom=None, eto=None):
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
        # dd.set_xlabel('rho')  # Add a y-label to the axes.
        # dd.set_title("pk/p from density")
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
