import numpy as np
import numpy.linalg as LA
import warnings
warnings.filterwarnings('ignore')

import qeschema
from abc import ABC, abstractmethod


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



class qe_analyse_base(ABC):

    def __init__(self, dir, name):
        """
        Initialize the QEBase object with the given directory and name.

        Args:
            dir (str): The directory path where the data 
            (! and folders ./qe and optionaly ./wannier) is located.
            name (str): The name of the material.

        Attributes:
            directory (str): Stores the directory path.
            name (str): Stores the name of the material or dataset.

        Methods:
            get_full_DOS: Retrieves the full density of states (DOS) data and eF.
            get_crystell_str: Retrieves the crystal structure data.
            get_hr: Retrieves the Hamiltonian data (band structure).
            get_sym_points: Retrieves the symmetry points data.
        """
        self.directory = dir # './'
        self.name = name # 'CrTe2'

        self.get_full_DOS()
        self.get_crystell_str()
        self.get_hr()
        self.get_sym_points()



    @abstractmethod
    def get_full_DOS(self):
        '''
        Reads the density of states (DOS) data from a file and stores it in the instance variables.
        This method initializes the following instance variables:
        - eDOS: A list of energy values.
        - efermi: The Fermi energy level.
        FOR SPIN POLARIZED DOS:
        - dosup: A list of DOS values for spin-up electrons.
        - dosdn: A list of DOS values for spin-down electrons.
        FOR NON SPIN POLARIZED DOS:
        - dos: A list of DOS values.
        '''
        pass


    @abstractmethod
    def get_hr(self):
        """
        Retrieves and processes spin band structure data from specified files.

        Attributes for FM:
            hDFT_up (numpy.ndarray): Spin-up band structure data.
            hDFT_dn (numpy.ndarray): Spin-down band structure data.
            nbandsDFT (int): Number of bands in the DFT calculation.
        Attributes for PM:
            hDFT (numpy.ndarray): band structure data.
            nbandsDFT (int): Number of bands in the DFT calculation.
        """
        pass


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

        print('positions (AA)')
        with printoptions(precision=10, suppress=True):
            print(  np.array(self.pos[1])*Bohr2Ang )


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
                
                kpath_lines.append(f'{Letter_to_write} {k_to_write[0]:.0f} {k_to_write[1]:.0f} {k_to_write[2]:.0f} \t {dist :.8f} \n')
                kpath_return.append(k_to_write)
                kpath_draw_path_return.append(dist )
                # print(kpath_lines[-1])
                dist += LA.norm(self.bcell.T @ delta_k / num_points_between)/ (2.*np.pi / self.alat)
            
            k_prev = k_new[:]
            Letter_prev = Letter_new
        print(''.join(kpath_lines))
        if saveQ:
            with open("./kpaths/" + filename, "w") as fout:
                fout.writelines(kpath_lines)
                
        return np.array(kpath_return), np.array(kpath_draw_path_return)
    

    @staticmethod
    def get_spin_BS(path):
        """
        Reads spin band structure data from a qe calculations file and returns it as a nested NumPy array.
        Args:
            path (str): The file path to read the spin band structure data from.
        Returns:
            np.ndarray: A nested NumPy array containing the spin band structure data. 
                        Each sub-array represents a band 
                        and contains the k-point coordinates and the energy.
        """
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
                  
                