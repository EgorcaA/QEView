from .qe_base import qe_analyse_base # Relative import

import numpy as np
import pandas as pd
import numpy.linalg as LA
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from  tqdm import tqdm
import os
import re
from .wannier_loader import Wannier_loader_PM


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



class qe_analyse_PM(qe_analyse_base):

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
        self.dos = []
        self.efermi = 0
        try:
            with open(self.directory + "qe/dos.dat") as f:
                line = f.readline()
                self.efermi = float(re.search(r"EFermi =\s*(-?\d+\.\d*)\s*eV", line).group(1))
                for line in f:
                    if not line.strip():
                        continue
                    energy, edos, *_ = line.split()
                    self.eDOS.append(float(energy))
                    self.dos.append(float(edos))
        except IOError:
            print("Error: DOS file does not appear to exist.")
        print(f'efermi {self.efermi:.2f}')
        self.eDOS = np.array(self.eDOS)
        self.dos = np.array(self.dos)



    def get_hr(self):
        self.hDFT = self.get_spin_BS(self.directory +'qe/bands.dat.gnu')
        self.nbandsDFT = self.hDFT.shape[0]



    def plot_FullDOS(self, efrom=-5, eto=5, saveQ=False, picname='DOS.png'):
        fig, dd = plt.subplots() 
        
        dd.plot(self.eDOS - self.efermi, self.dos, color='red', linewidth=0.5)

        plt.fill_between(
                x= self.eDOS-self.efermi, 
                y1=self.dos,
                y2=0,
                color= "grey",
                alpha= 0.1)

        # locator = AutoMinorLocator()
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        dd.yaxis.set_major_locator(MultipleLocator(2))
        dd.xaxis.set_minor_locator(MultipleLocator(1))
        dd.xaxis.set_major_locator(MultipleLocator(2))

        dd.set_ylabel('Density of states')  # Add an x-label to the axes.
        dd.set_xlabel(r'$E-E_f$ [eV]')  # Add a y-label to the axes.
        dd.set_title("Nonspinpolarized DOS")
        # dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        
        dd.vlines(0, ymin=0, ymax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        # dd.hlines(0, xmin=-30, xmax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        
        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        dd.set_ylim((0, 10))
        dd.set_xlim((efrom, eto))
        
        if saveQ:
            plt.savefig('./'+ picname, dpi=200, bbox_inches='tight')
        plt.show()


    def get_pDOS(self):
        
        def read_pdos(file, i):
            df = pd.read_csv(self.directory +'qe/'+ str(file), sep='\s+', skiprows=[0], header=None)
            e, pdos = df.iloc[:, 0], df.iloc[:, [i,i+1]].sum(axis=1)
            return e, pdos

        def list_pdos_files(path):
            for f in os.listdir(path):
                
                if f.startswith( self.name + '.pdos_atm'):
                    match = re.search(
                        r"pdos_atm#(\d+)\((\w+)\)\_wfc#(\d+)\((\w+)\)", f)
                    if not match:
                        raise FileNotFoundError
                    yield f, match.groups()

        self.pdos = {"s": dict(), "p": dict(), "d": dict()}
        for file, info in list_pdos_files(self.directory + 'qe/'):
            atom_number,  _, _, orbital_type = info
            
            self.ePDOS, pdos = read_pdos(file, 1)#spinup
            self.pdos[orbital_type].update({atom_number: pdos})




    def plot_pDOS(self, element="1", efrom=None, eto=None, yto=None):
        if efrom is None:
            efrom = -15
        if eto is None:
            eto =15
        if yto is None:
            yto = 10

        fig, dd = plt.subplots()  # Create a figure containing a single axes.

        atom_pdos = {"s": None, "p": None, "d": None}
        atom_tdos = np.zeros((len(self.pdos['s']['1'])))
        
        for orbital_type in atom_pdos.keys():
            if str(element) in self.pdos[orbital_type].keys():
                atom_pdos[orbital_type] = self.pdos[orbital_type][str(element)]
                atom_tdos += self.pdos[orbital_type][str(element)]

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
        dd.set_ylim((0, yto))
        dd.set_xlim((efrom, eto))
        # plt.savefig('./2pub/pics/pDOS.png', dpi=200, bbox_inches='tight')

        plt.show()


    def print_bands_range(self, band_from=None, band_to=None):
        if band_from is None:
            band_from = 0
        if band_to is None:
            band_to = self.nbandsDFT

        print(f'efermi {self.efermi:.2f}')
        print("-------------BANDS---------------")
        for band_num in range(band_from,band_to):
            print(f'band {band_num+1} eV from  {min(self.hDFT[band_num, : ,1]) :.2f} to  {max(self.hDFT[band_num, : ,1]) :.2f} \
                eV-eF from  {min(self.hDFT[band_num, : ,1]) -self.efermi :.2f} to  {max(self.hDFT[band_num, : ,1]) - self.efermi:.2f}' )



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
                dd.plot(self.hDFT[band, : ,0], 
                        self.hDFT[band, : , 1] - self.efermi, label='up', color='black', linewidth=0.7,
                            alpha=1.0)

            else:
                dd.plot(self.hDFT[band, : ,0], 
                        self.hDFT[band, : , 1] - self.efermi,  color='black', linewidth=0.7,
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

         
    # Wannier90 interface 
    def load_wannier(self, kpath_filename='kpath_qe2.dat', wannier_hr='wannier90_hr.dat'):
        self.wannier = Wannier_loader_PM(self.directory, wannier_hr)
        self.wannier.load_kpath('./kpaths/'+ kpath_filename)
        self.BS_wannier = self.wannier.get_wannier_BS(spin=0)


    def plot_wannier_BS(self, efrom=None, eto=None):
        if efrom is None:
            efrom = -15
        if eto is None:
            eto =15

        nwa = self.BS_wannier.shape[1]

        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists

        fig, dd = plt.subplots()  # Create a figure containing a single axes.
        for band in range(self.nbandsDFT):
            if band == 0:
                dd.plot(self.hDFT[band, : ,0], 
                        self.hDFT[band, : , 1] - self.efermi, label='DFT', color='black', linewidth=0.7,
                            alpha=1.0)

            else:
                dd.plot(self.hDFT[band, : ,0], 
                        self.hDFT[band, : , 1] - self.efermi,  color='black', linewidth=0.7,
                        alpha=1.0)


        for band in range(nwa):
            if band == 0:
                
                dd.plot(self.wannier.kpath_dists_qe,
                        self.BS_wannier[ : , band] - self.efermi , label='wannier', color='r', alpha=0.5, linewidth=3)
        
            else:
                
                dd.plot(self.wannier.kpath_dists_qe,
                        self.BS_wannier[ : , band] - self.efermi , color='r', alpha=0.3, linewidth=3)


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
