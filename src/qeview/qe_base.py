import numpy as np
import numpy.linalg as LA
import warnings
warnings.filterwarnings('ignore')
import os
import qeschema
from abc import ABC, abstractmethod


Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr

Ry2eV = 13.6057

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)



class qe_analyse_base(ABC):
    """
    qe_analyse_base is an abstract base class for analyzing Quantum Espresso (QE) calculations. 
    It provides methods to read and process various QE output files, including density of states (DOS), 
    band structure, crystal structure, and high symmetry points.
    """

    def __init__(self, dir, name):
        """
        Initialize the QEBase object with the given directory and name.
        """
        self.directory = os.path.abspath(dir) # '.'
            
        self.name = name # 'CrTe2'
        self.HighSymPointsNames = None
        self.get_crystall_struct()



    @abstractmethod
    def get_full_DOS(self, filename='dos.dat'):
        '''
        Reads the density of states (DOS) data from a file and stores it in the instance variables.

        This method initializes the following instance variables:
            - eDOS: A list of energy values.
            - efermi: The Fermi energy level.

        For spinpolarized DOS:
            - dosup: A list of DOS values for spin-up electrons.
            - dosdn: A list of DOS values for spin-down electrons.
        
        For non-spinpolarized DOS:
            - dos: A list of DOS values.
        '''
        pass


    @abstractmethod
    def get_band_structure(self, bands_up_name=None, bands_dn_name=None):
        """
        Retrieves and processes spin band structure data from specified files.

        Attributes for FM:
            - hDFT_up (numpy.ndarray): Spin-up band structure data.
            - hDFT_dn (numpy.ndarray): Spin-down band structure data.
            - nbandsDFT (int): Number of bands in the DFT calculation.
        Attributes for PM:
           - hDFT (numpy.ndarray): band structure data.
           - nbandsDFT (int): Number of bands in the DFT calculation.
        """
        pass


    def get_crystall_struct(self, filename='data-file-schema.xml', qe_dir='qe'):
        """
        Reads and processes the crystallographic structure from a specified XML file.
        Prints:
            - Unit Cell Volume in cubic Angstroms.
            - Reciprocal-space vectors in Angstroms^-1 and in units of 2π/alat.
            - Real-space vectors in Angstroms and in units of alat.
            - Atomic positions in Cartesian coordinates (in units of alat and Angstroms) and fractional coordinates.
        
        
        Parameters
        ----------
        filename : str, optional
            The name of the XML file containing the crystallographic data. 
            Defaults to 'data-file-schema.xml'.
        qe_dir : str, optional
            The directory where the file is located. Default is 'qe'.

        Raises
        ------
        FileNotFoundError
            If the specified file is not found.
        IOError
            If there is an error reading the file with qeschema.

        Attributes
        ----------
        qerun_dict : dict
            Dictionary containing parsed data from the XML file.
            :ref:`tree_structure_example` : This page contains the structure of qerun_dict.
        alat : float
            Lattice constant in Angstroms.
        acell : numpy.ndarray
            Real-space lattice vectors in Angstroms.
        bcell : numpy.ndarray
            Reciprocal-space lattice vectors in Angstroms^-1.
        pos : tuple
            Atomic positions in the unit cell.

        
        """
        
        pw_document = qeschema.PwDocument()
        file_path = os.path.join(self.directory, qe_dir, filename)
        try:
            with open(file_path) as fin:
                pass
            pw_document.read(file_path)
            self.qerun_dict = pw_document.to_dict()['qes:espresso']
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file '{file_path}' was not found: {e}") from e
        except Exception as e:
            raise IOError(f"Failed read the file '{file_path}' with qeschema: {e}") from e
           
        
        acell = np.array(pw_document.get_cell_parameters())*Bohr2Ang
        self.alat = self.qerun_dict['input']['atomic_structure']['@alat']*Bohr2Ang
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

        print('positions (AA)')
        with printoptions(precision=10, suppress=True):
            print(  np.array(self.pos[1])*Bohr2Ang )


    def get_sym_points(self, filename='band.in', qe_dir='qe'):
        """
        Parses the high symmetry points from a Quantum Espresso input file
        and calculates their distances and coordinates.
        Populates the HighSymPointsNames, HighSymPointsDists, and HighSymPointsCoords attributes

        Parameters
        ----------
        filename : str, optional
            The name of the file to read the high symmetry points from. Default is 'band.in'.
        qe_dir : str, optional
            The directory where the file is located. Default is 'qe'.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        IOError
            If there is an error reading the file.

        """
        
        self.HighSymPointsNames = []
        self.HighSymPointsDists = []
        self.HighSymPointsCoords = []

        # directory = os.path.abspath(directory)  # Get absolute path
        # os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(self.directory, qe_dir, filename)

        try:
            with open(file_path) as fin:
                # Skip lines until 'K_POINTS' is found
                while True:
                    file_row = fin.readline()
                    if file_row.split()[0] == 'K_POINTS':
                        break
                
                n_strings = int(fin.readline())
                k_string = fin.readline().split()
                assert len(k_string) == 6, "kpoints should be named in band.in as 0 0 0 10 ! G"
                Letter_prev = k_string[5]
                 
                dist = 0.0
                k_prev = np.array(list(map(float, k_string[:3])))
                
                self.HighSymPointsNames.append(Letter_prev)
                self.HighSymPointsDists.append(dist)
                self.HighSymPointsCoords.append(k_prev)

                for _ in range(n_strings - 1):
                    line = fin.readline()
                    k_string = line.split()
                    assert len(k_string) == 6, "kpoints should be named in band.in as 0 0 0 10 ! G"
                    Letter_new = k_string[5]
                    k_new = np.array(list(map(float, k_string[:3])))
                    delta_k = k_new - k_prev
                    dist += LA.norm(self.bcell.T @ delta_k) / (2. * np.pi / self.alat)
                    k_prev = k_new
                    
                    self.HighSymPointsNames.append(Letter_new)
                    self.HighSymPointsDists.append(dist)
                    self.HighSymPointsCoords.append(k_prev)
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file '{file_path}' was not found: {e}")
        except IOError as e:
            raise IOError(f"Failed to read the file '{file_path}': {e}") from e


        """
        Generate a k-path with points between high-symmetry points, proportional to their distances.
        Writes to a file and returns the k-path as a list of formatted strings.
        
        :param filename: Name of the output file
        :param points_per_unit: Number of points per unit distance
        """

    def get_qe_kpathBS(self, points_per_unit=10, saveQ=True, filename="kpath_qe2.txt", directory="kpaths"):
        """
        Generate the k-path for band structure calculations and optionally save it to a file.
        
        Parameters
        ----------
        points_per_unit : int, optional
            Number of points per unit distance between high symmetry points (default is 10).
        saveQ : bool, optional
            If True, save the k-path to a file (default is True).
        filename : str, optional
            Name of the file to save the k-path (default is "kpath_qe2.txt").
        directory : str, optional
            Directory where the file will be saved (default is "kpaths").
        
        Returns
        --------
        kpath_coords : np.ndarray
            Array of k-point coordinates along the path.
        kpath_dists : np.ndarray
            Array of distances along the k-path.
        
        Raises
        -------
        Exception
            If high symmetry points were not parsed from the band.in file.
        AssertionError
            If there are not enough high symmetry points.
        IOError
            If there is an error writing to the file.
        """

        if self.HighSymPointsNames is None:
            raise Exception('High Symmetry Points were not parsed from band.in file. Run get_sym_points()')
        assert len(self.HighSymPointsNames) > 1, "there are not enough HighSymPoints"

        kpath = []
        kpath_coords = []
        kpath_dists = []
        
        for i in range(len(self.HighSymPointsNames) - 1):
            name1 = self.HighSymPointsNames[i]
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
            directory = os.path.abspath(directory)  # Get absolute path
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "w", encoding="utf-8") as fout:
                    fout.write("\n".join(kpath))
            except OSError as e:
                raise IOError(f"Failed to write to file '{file_path}': {e}") from e
            
                
        
        return np.array(kpath_coords), np.array(kpath_dists)



    def get_integer_kpath(self, 
                          N_points_direction=10, 
                          num_points_betweens=5, 
                          saveQ=False,
                          filename='kpath_integer.dat', 
                          directory="kpaths"):
        """
        Generates a k-path with integer coordinates for high symmetry points.
        The method prints the k-path lines and the high symmetry points names during execution.

        Parameters
        -----------
        N_points_direction : int, optional
            The number of points in each direction (default is 10).
        num_points_betweens : int or list of int, optional
            The number of points between each pair of high symmetry points. If an integer is provided, 
            the same number of points is used between all pairs. If a list is provided, it should have 
            length NHSP-1, where NHSP is the number of high symmetry points (default is 5).
        filename : str, optional
            The name of the file to save the k-path (default is 'kpath_integer.dat').
        
        Returns
        --------
        kpath_return : list of numpy arrays
            List of k-points with integer coordinates.
        kpath_draw_path_return : list of floats
            List of distances corresponding to each k-point in units 2 pi / alat.


        """
        if self.HighSymPointsNames is None:
            raise Exception('High Symmetry Points were not parsed from band.in file. Run get_sym_points()')
        assert len(self.HighSymPointsNames) > 1, "there are not enough HighSymPoints"
        
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
                
                kpath_lines.append(f'{Letter_to_write} {k_to_write[0]:.0f} {k_to_write[1]:.0f} {k_to_write[2]:.0f} \t {dist :.8f} \n')
                kpath_return.append(k_to_write)
                kpath_draw_path_return.append(dist )
                # print(kpath_lines[-1])
                dist += LA.norm(self.bcell.T @ delta_k / num_points_between)/ (2.*np.pi / self.alat)
            
            k_prev = k_new[:]
            Letter_prev = Letter_new
        print(''.join(kpath_lines))


        if saveQ:
            directory = os.path.abspath(directory)  # Get absolute path
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "w", encoding="utf-8") as fout:
                    fout.writelines(kpath_lines)
            except OSError as e:
                raise IOError(f"Failed to write to file '{file_path}': {e}") from e
            
        return np.array(kpath_return), np.array(kpath_draw_path_return)
    

    @staticmethod
    def get_spin_BS(file_path):
        """
        Reads spin band structure data from a default qe bands file and returns it as a numpy array.
        
        Parameters
        ----------
        file_path : str
            The path to the file containing the band structure data.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array where each element is a numpy array representing a band.

        Raises
        ------
        FileNotFoundError
            If the band structure file is not found.
        IOError
            If there is an error reading the band structure file.
        """
        
        hr_fact_data = []
        try:
            with open(file_path, "r") as f:
                band = 0
                hr_fact_data.append([])

                for line in f:
                    if line == ' \n':
                        hr_fact_data[-1] = np.array(hr_fact_data[-1])
                        hr_fact_data.append([])
                        band += 1
                    else:
                        hr_string = line.split()
                        hr_fact_data[-1].append(np.array([
                            float(hr_string[0]), float(hr_string[1]), 
                        ]))
                        
            hr_fact_data = np.array(hr_fact_data[:-1])  
            return hr_fact_data

        except FileNotFoundError as e:
            raise FileNotFoundError(f"The band structure file '{file_path}' was not found: {e}")
        except IOError as e:
            raise IOError(f"Failed to read band structure from file '{file_path}': {e}") from e
            


