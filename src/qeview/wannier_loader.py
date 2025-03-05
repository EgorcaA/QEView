import numpy as np
import warnings
warnings.filterwarnings('ignore')
from abc import ABC, abstractmethod

import pickle 
from  tqdm import tqdm


Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr



class Wannier_loader(ABC):
    '''
    Wannier_loader is an abstract base class for loading and manipulating Wannier Hamiltonians.
        directory (str): The directory where Wannier files are located.
        name (str): The name of the Wannier file.
        nwa (int): Number of Wannier functions.
        nD (int): Dimensionality of the Hamiltonian (2D or 3D).
        k_path_qe (np.ndarray): Array of k-point coordinates.
        kpath_dists_qe (np.ndarray): Array of k-path distances in 2pi/alat.
        complex_hr (np.ndarray): Array of complex Hamiltonian matrices.
        hks_spins (np.ndarray): Array of Hamiltonians for each k-point.
        kpoints_adj_serial (np.ndarray): dense k-points grid.
    Methods:
        __init__(dir, name):
            Initialize the Wannier_loader with a directory and name.
        load_kpath(path):
        load_hr_pickle():
            Load Hamiltonian data from a pickle file.
        save_hk_pickle(name):
            Save Hamiltonian data to a pickle file.
        load_hk_pickle(name):
            Load Hamiltonian data from a pickle file.
        load_util(filename):
        get_hk_eig_path(path, find_eigsQ=False, spin=0):
        get_dense_hk_symmetric(nkpt=10, krange=1, find_eigsQ=False):
            Generate a dense grid of k-points from -1 to 1 and compute the Hamiltonian for each k-point.
        get_dense_hk(nkpt=10, find_eigsQ=False):
        get_hk_spins(path, find_eigsQ=False):
            Abstract method to get Hamiltonians and eigenvalues along the path for one/both spins.
        get_wannier_BS(spin=0):

    '''
    def __init__(self, dir, name):
        self.directory = dir # './'
        self.name = name # 'CrTe2'
        self.nwa = 0
        self.load_wannier90(wannier_hr_filename=name)
    

    def load_kpath(self, path):
        """
        Load k-path data from a file.

        Parameters:
        path (str): The file path to the k-path data file.

        The file should contain lines of space-separated values where:
        - The second, third, and fourth values represent the k-point coordinates.
        - The fifth value represents the k-path distance in 2pi/alat.

        example:
        G 0.00000000 0.00000000 0.00000000 0.00000000
        . 0.00000000 0.10000000 0.00000000 0.11820789
        . 0.00000000 0.20000000 0.00000000 0.23641577
        . 0.00000000 0.30000000 0.00000000 0.35462366
        . 0.00000000 0.40000000 0.00000000 0.47283155
        M 0.00000000 0.50000000 0.00000000 0.59103943
        . -0.11111111 0.55555556 0.00000000 0.70478502
        ...

        This method reads the file, extracts the k-point coordinates and k-path distances,
        and stores them in the instance variables `self.k_path_qe` and `self.kpath_dists_qe`.

        Instance Variables:
        self.k_path_qe (np.ndarray): Array of k-point coordinates with shape (n, self.nD).
        self.kpath_dists_qe (np.ndarray): Array of k-path distances.
        """
        k_path = []
        kpath_dists = []

        with open(path) as f:
            for line in f:
                kpts_string = line.split()
                k_path.append(np.array([
                    float(kpts_string[1]), float(kpts_string[2]), float(kpts_string[3])
                ]))
                kpath_dists.append(float(kpts_string[4]))

        self.k_path_qe = np.array(k_path)[:,:self.nD]   
        self.kpath_dists_qe = np.array(kpath_dists)   


    def load_hr_pickle(self):
        with open(self.directory + '/wannier/hr_mn' + self.name +'.pickle', 'rb') as f:
            self.complex_hr = pickle.load(f)

    def save_hk_pickle(self, name):
        with open(self.directory + '/wannier/hk_dense_' + name + '.pickle', 'wb') as f:
            pickle.dump(self.hks_spins, f)

    def load_hk_pickle(self, name):
        with open(self.directory + '/wannier/hk_dense_'+ name + '.pickle', 'rb') as f:
            self.hks_spins = pickle.load(f)



    def load_util(self, filename):
        """
        Load Wannier Hamiltonian data from a specified file.
        Parameters:
        filename (str): The name of the file (without extension) to load the data from.
        Returns:
        tuple: A tuple containing:
            - R_coords (list of lists): A list of R coordinates, where each coordinate is a list of three floats.
            - hr (numpy.ndarray): A 3D numpy array of complex numbers representing the Hamiltonian matrix.
        Attributes:
        self.nwa (int): Number of Wannier functions.
        self.nD (int): Dimensionality of the Hamiltonian (2D or 3D).
        Notes:
        The file should be located in the 'wannier' subdirectory of the instance's directory attribute.
        The file should have a '.dat' extension.
        """
        hr = 0
        R_coords = []
        R_weights = []
        is3D = 0
        with open(self.directory +'wannier/' + filename) as f:
                f.readline()
                
                nwa = int(f.readline().strip('\n'))
                print("nwa ", nwa)
                self.nwa = nwa
                Rpts = int(f.readline().strip('\n'))
                print("Rpts", Rpts)
                i=1
                hr = np.zeros((nwa, nwa, Rpts), dtype=complex)

                R_ind = -1
                line_ind = 0
                for line in f:
                    
                    # Check if the current line is part of the R weights section
                    if i < Rpts / 15 + 1:
                        R_weights.extend(int(x) for x in line.split() if x.isnumeric())
                        R_weights +=  [ int(x) for x in line.split() if x.isnumeric() ]
                        # print(line)
                        i+=1
                    else:

                        hr_string = line.split()
                        
                        # Check if the current line corresponds to the start of a new R coordinate block
                        if line_ind % nwa**2 == 0:
                            if float(hr_string[2]) != 0: is3D = 1 
                            R_coords.append([float(hr_string[0]), float(hr_string[1]), float(hr_string[2])]) 
                            R_ind += 1
                        hr[int(hr_string[3])-1, int(hr_string[4])-1, R_ind] = float(hr_string[5])+ 1j*float(hr_string[6])
                        
                        line_ind +=1 
        self.nD = is3D + 2

        print(f'we have {self.nD}D hamiltonian')
        return np.array(R_coords), hr

    
    def get_hk_eig_path(self, path, find_eigsQ=False, spin=0): 
        """
        Calculate the eigenvalues and Hamiltonians along a given k-path.

        Parameters:
        path (list of array-like): A list of k-points in the path.
        spin (int, optional): Spin index to select the appropriate Hamiltonian. Default is 0.

        Returns:
        tuple: 
            band_str (ndarray): Array of sorted real parts of eigenvalues for each k-point.
            hks_bs (ndarray): Array of Hamiltonians for each k-point.
        """
        band_str = []
        hks_bs = []
        for k in tqdm(path):
            hk = np.sum( [np.exp(-2*np.pi*1.j* np.dot(k, R) )*self.complex_hr[:, :, R_ind, spin] 
                          for R_ind, R in enumerate(self.R_coords)], axis=0 )
            hks_bs.append(hk)
            if find_eigsQ:
                band_str.append(np.sort(np.real(np.linalg.eig(hk)[0])))

        band_str = np.array(band_str) # (nkpt, nwa)
        return band_str, np.array(hks_bs) # (nkpt, nwa, nwa)



    def get_dense_hk_symmetric(self, nkpt=10, krange=1, find_eigsQ=False):
        """
        Generates a dense grid of k-points from -1 to 1 and computes the Hamiltonian for each k-point.
        Parameters:
        nkpt (int): Number of k-points along each dimension. Default is 10.
        krange (float): Range of k-points in reciprocal space. Default is 1.
        Returns:
        None: The function updates the instance variables `hks_spins` and `kpoints_adj_serial` with the computed Hamiltonians and k-points respectively.
        Notes:
        - For a 3-dimensional system (self.nD == 3), a 3D grid of k-points is generated.
        - For a 2-dimensional system (self.nD == 2), a 2D grid of k-points is generated.
        """
        
        if self.nD == 3: 
            kpoints_adj_serial = np.mgrid[-krange:krange:1.0/nkpt, -krange:krange:1.0/nkpt, -krange:krange:1.0/nkpt].reshape(3,-1).T
            bs, self.hks_spins = self.get_hk_spins(kpoints_adj_serial, find_eigsQ=find_eigsQ)
            
        else: # D= 2
            kpoints_adj_serial = np.mgrid[-krange:krange:1.0/nkpt, -krange:krange:1.0/nkpt].reshape(2,-1).T
            bs, self.hks_spins = self.get_hk_spins(kpoints_adj_serial, find_eigsQ=find_eigsQ)
            
        
        self.kpoints_adj_serial = kpoints_adj_serial
        return bs, self.hks_spins

    def get_dense_hk(self, nkpt=10, find_eigsQ=False):
        """
        Generate a dense grid of k-points and compute the Hamiltonian for each k-point.
        Parameters:
        nkpt (int): Number of k-points along each dimension. Default is 10.
        Returns:
        None: The function updates the instance variables `hks_spins` and `kpoints_adj_serial` with the computed Hamiltonians and k-points respectively.
        """
        if self.nD == 3: 
            kpoints_adj_serial = np.mgrid[0:1:1.0/nkpt, 0:1:1.0/nkpt, 0:1:1.0/nkpt].reshape(3,-1).T
            bs, self.hks_spins = self.get_hk_spins(kpoints_adj_serial, find_eigsQ=False)

        else: # D=2
            kpoints_adj_serial = np.mgrid[0:1:1.0/nkpt, 0:1:1.0/nkpt].reshape(2,-1).T
            bs, self.hks_spins = self.get_hk_spins(kpoints_adj_serial, find_eigsQ=False)

        self.kpoints_adj_serial = kpoints_adj_serial
        return bs, self.hks_spins


    @abstractmethod
    def get_hk_spins(self, path, find_eigsQ=False):
        '''
        get hamiltonians and eigenvalues along the path for one/both spins
        '''
        pass


    def get_wannier_BS(self, spin=0):
        """
        Retrieve the Wannier band structure for a given spin.

        Parameters:
        spin (int): The spin index for which to retrieve the band structure. Default is 0.

        Returns:
        numpy.ndarray: The eigenvalues of the Hamiltonian along the specified k-path.
        """
        band_str, _ = self.get_hk_eig_path(self.k_path_qe, find_eigsQ=True, spin=spin)
        return band_str
    

class Wannier_loader_FM(Wannier_loader):
    """
    A class to handle loading and processing of Wannier90 data for ferromagnetic hamiltonians.
    
    """

    def __init__(self,  dir, name):
        super().__init__( dir, name) 

    def load_wannier90(self, wannier_hr_filename):
        
        R_coords, hr_up = self.load_util('hrup.dat')
        _, hr_dn = self.load_util('hrdn.dat')
        # print(hr_up.shape, hr_dn.shape)
        self.complex_hr = np.transpose(np.array([hr_up, hr_dn]), (1,2,3,0)) # (spin, nwa, nwa, Rpts ) -> (nwa, nwa, Rpts, spin)
        self.R_coords = R_coords[:, :self.nD]
        

    def get_hk_spins(self, path, find_eigsQ=False):
        path = path[:, :self.nD]
        band_str_up, hks_up = self.get_hk_eig_path( path, find_eigsQ=find_eigsQ, spin=0)
        band_str_dn, hks_dn = self.get_hk_eig_path( path, find_eigsQ=find_eigsQ, spin=1)
        # print(band_str_dn.shape, hks_dn.shape)
        hks_spins = np.transpose( np.array([hks_up, hks_dn]) , (2,3, 1,0)) # (nwa, nwa, nkpt, spin)
        if find_eigsQ:
            bs_spins = np.transpose( np.array([band_str_up, band_str_dn]) , (2,1,0)) # (nwa, nwa, nkpt, spin)
            return bs_spins, hks_spins 
        return [], hks_spins 

    
class Wannier_loader_PM(Wannier_loader):
    """
    A class to handle loading and processing of Wannier90 data for paramagnetic hamiltonians.
    
    """

    def __init__(self,  dir, name):
        super().__init__( dir, name) 

    def load_wannier90(self, wannier_hr_filename):
        
        R_coords, hr = self.load_util(wannier_hr_filename)
        
        self.complex_hr = hr[:,:,:, np.newaxis] # (nwa, nwa, Rpts, spin=0)
        self.R_coords = R_coords[:, :self.nD]
        


    def get_hk_spins(self, path, find_eigsQ=False):
        path = path[:, :self.nD]
        band_str, hk = self.get_hk_eig_path( path, find_eigsQ=find_eigsQ, spin=0)
        hks_spins = np.transpose( hk , (2,1,0))[:,:,:, np.newaxis] # (nwa, nwa, nkpt, spin)
        if find_eigsQ:
            bs_spins = np.transpose( np.array([band_str]) , (2,1,0)) # (nwa, nkpt, spin)
            return bs_spins, hks_spins 
        return [], hks_spins 
