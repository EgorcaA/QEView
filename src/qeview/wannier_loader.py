import numpy as np
import warnings
warnings.filterwarnings('ignore')
from abc import ABC, abstractmethod
import os
import pickle 
from  tqdm import tqdm


Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr



class Wannier_loader(ABC):
    '''
    Wannier_loader is an abstract base class for loading and manipulating Wannier Hamiltonians.
    '''
    def __init__(self):
        self.nwa = 0
        
    def load_kpath(self, file_path):
        """
        Load k-points and their distances from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the k-points.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        IOError
            If there is an error reading the file.
        AssertionError
            If the file format is incorrect.

            
        Notes
        -----
        The file should contain lines formatted as:
        M 0.0 0.5 0.0 0.591
        
        Examples
        --------
        The following is an example of the expected input file format::

            G  0.00000000  0.00000000  0.00000000  0.00000000
            .  0.00000000  0.10000000  0.00000000  0.11820789
            .  0.00000000  0.20000000  0.00000000  0.23641577
            .  0.00000000  0.30000000  0.00000000  0.35462366
            .  0.00000000  0.40000000  0.00000000  0.47283155
            M  0.00000000  0.50000000  0.00000000  0.59103943
            . -0.11111111  0.55555556  0.00000000  0.70478502
            ...

        where:
            - The first element is a label (ignored in this function).
            - The next three elements are the k-point coordinates (fractional).
            - The last element is the distance associated with the k-point (in 2 pi/ alat).


        """
        
        try:

            k_path = []
            kpath_dists = []

            with open(file_path, "r") as f:
                for line in f:
                    kpts_string = line.split()
                    assert len(kpts_string) == 5, "kpoints file should be formated as M 0.0 0.5 0.0 0.591"
                    k_path.append(np.array([
                        float(kpts_string[1]), float(kpts_string[2]), float(kpts_string[3])
                    ]))
                    kpath_dists.append(float(kpts_string[4]))

            self.k_path_qe = np.array(k_path)[:,:self.nD]   
            self.kpath_dists_qe = np.array(kpath_dists)   
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The kpoints file '{file_path}' was not found: {e}")
        except IOError as e:
            raise IOError(f"Failed to read kpoints from file '{file_path}': {e}") from e


    def load_hr_pickle(self, dump_name, wannier_dir='wannier'):
        filedir = os.path.join(os.path.abspath(wannier_dir), 'hr_mn_' + dump_name +'.pickle')
        with open(filedir, 'rb') as f:
            self.complex_hr = pickle.load(f)

    def save_hk_pickle(self, dump_name, wannier_dir='wannier'):
        filedir = os.path.join(os.path.abspath(wannier_dir), 'hk_dense_' + dump_name +'.pickle')
        with open(filedir, 'wb') as f:
            pickle.dump(self.hks_spins, f)

    def load_hk_pickle(self, dump_name, wannier_dir='wannier'):
        filedir = os.path.join(os.path.abspath(wannier_dir), 'hk_dense_' + dump_name +'.pickle')
        with open(filedir, 'rb') as f:
            self.hks_spins = pickle.load(f)



    def load_util(self, hr_filepath):
        """
        Load Wannier Hamiltonian data from a specified file.

        Parameters
        ----------
        hr_filepath : str
            The file path to the Wannier Hamiltonian file.

        Returns
        -------
        R_coords : np.ndarray(Rpts, 3)
            An array of R coordinates. Second dimention should be 3 even in 2D case (zero value)
        hr : np.ndarray (nwa, nwa, Rpts)
            An array representing the Hamiltonian matrix.
        
        Raises
        ------
        FileNotFoundError
            If the specified file is not found.
        IOError
            If there is an error reading the file.
        """
        
        hr = 0
        R_coords = []
        R_weights = []
        is3D = 0
        try:
            with open(hr_filepath) as f:
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
                        if i < Rpts / 15 + 1: # wannier90 output form
                            R_weights.extend(int(x) for x in line.split() if x.isnumeric())
                            R_weights +=  [ int(x) for x in line.split() if x.isnumeric() ]
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
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The wannier hamiltonian file '{hr_filepath}' was not found: {e}")
        except IOError as e:
            raise IOError(f"Failed to read wannier hamiltonian from file '{hr_filepath}': {e}") from e
        
    
    def get_hk_eig_path(self, path, find_eigsQ=False, spin=0): 
        """
        Calculate the eigenvalues and Hamiltonians along a given k-path.

        Parameters
        ---------
        path : list of array-like 
            A list of k-points in the path.
        find_eigsQ : bool, optional
            If True, compute eigenvalues of the Hamiltonian matrices (interpolated band structure). Default is False.
        spin : int, optional 
            Spin index to select the appropriate Hamiltonian. Default is 0.

        Returns
        ---------
        band_str : ndarray (nkpt, nwa)
            Array of sorted real parts of eigenvalues for each k-point.
        hks_bs : ndarray (nkpt, nwa, nwa)
            Array of Hamiltonians for each k-point.
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
        Generate a dense grid of k-points from -krange to krange (fractional coords) 
        and compute the Hamiltonian matrix for each k-point.

        Parameters
        -----------
        nkpt : int, optional
            Number of k-points along each dimension. Default is 10.
        krange : float, optional
            Range of k-points in reciprocal space. Default is 1.
        find_eigsQ : bool, optional
            If True, compute eigenvalues of the Hamiltonian matrices (interpolated band structure). Default is False.
        
        Returns
        --------
        bs : ndarray (nwa, nkpt, spin)
            Array of band structures or eigenvalues, depending on the value of `find_eigsQ`.
        hks_spins : ndarray (nwa, nwa, nkpt, spin)
            Array of Hamiltonian matrices for each k-point.
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
        
        Parameters
        ----------
        nkpt : int, optional 
            Number of k-points along each dimension. Default is 10.
        find_eigsQ : bool, optional
            If True, compute eigenvalues of the Hamiltonian matrices (interpolated band structure). Default is False.
        
        Returns
        -------
        bs : ndarray (nwa, nkpt, spin)
            Array of band structures or eigenvalues, depending on the value of `find_eigsQ`.
        hks_spins : ndarray (nwa, nwa, nkpt, spin)
            Array of Hamiltonian matrices for each k-point.

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

        Parameters
        ----------
        spin : int
            The spin index for which to retrieve the band structure. Default is 0.

        Returns
        -------
        band_str : numpy.ndarray (nkpt, nwa)
            The eigenvalues of the Hamiltonian along the specified k-path.
        """
        band_str, _ = self.get_hk_eig_path(self.k_path_qe, find_eigsQ=True, spin=spin)
        return band_str
    

class Wannier_loader_FM(Wannier_loader):
    """
    A class to handle loading and processing of Wannier90 data for ferromagnetic hamiltonians.
    
    """

    def __init__(self,  hr_up_name=None, hr_dn_name=None, wannier_dir='wannier', **kwargs):
        super().__init__(  ) 
        os.makedirs(os.path.abspath(wannier_dir), exist_ok=True)

        self.load_wannier90(hr_up_name, hr_dn_name, wannier_dir)
    

    def load_wannier90(self, hr_up_name=None, hr_dn_name=None, wannier_dir='wannier', **kwargs):
        """
        Load Wannier90 Hamiltonians from specified or default filenames.

        This method attempts to load Wannier90 Hamiltonians from the provided filenames
        or from default filenames ('hrup.dat' and 'hrdn.dat') within the specified directory.
        The Hamiltonians are expected to have the same R mesh.

        Parameters
        ----------
        hr_up_name : str, optional
            Filename for the spin-up Hamiltonian. Defaults to None.
        hr_dn_name : str, optional
            Filename for the spin-down Hamiltonian. Defaults to None.
        wannier_dir : str, optional
            Directory containing the Wannier90 files. Defaults to 'wannier'.
        **kwargs : Additional keyword arguments.

        Raises
        ------
        FileNotFoundError
            If no valid Wannier90 Hamiltonians are found in the specified directory.

        Fields
        -------
        complex_hr : numpy.ndarray(nwa, nwa, Rpts, spin)
            complex wannier hamiltonian in R basis
        R_coords: numpy.ndarray(Rpts, 3)

        """
        
        
        potential_files = [
            (hr_up_name, hr_dn_name),  # User-specified filenames
            ('hrup.dat', 'hrdn.dat')
        ]

        for up_name, dn_name in potential_files:
            if up_name and dn_name:  # Ensure both filenames are available
        
                file_hr_up = os.path.join(os.path.abspath(wannier_dir), up_name)
                file_hr_dn = os.path.join(os.path.abspath(wannier_dir), dn_name)

                try:
                    R_coords1, hr_up = self.load_util(file_hr_up)
                    R_coords2, hr_dn = self.load_util(file_hr_dn)
                    assert np.array_equal(R_coords1 , R_coords2), "Different R meshes in wannier hamiltonians"

                    self.complex_hr = np.transpose(np.array([hr_up, hr_dn]), (1,2,3,0)) # (spin, nwa, nwa, Rpts ) -> (nwa, nwa, Rpts, spin)
                    self.R_coords = R_coords1[:, :self.nD]

                    return  # Exit once successful
                except FileNotFoundError:
                    continue  # Try the next fallback

        # If all fallbacks fail, raise an error
        raise FileNotFoundError(f"No valid wannier hamiltonians found in {os.path.abspath(wannier_dir)}.")

        

    def get_hk_spins(self, path, find_eigsQ=False):
        """
        Computes the Hamiltonian matrix elements and optionally the eigenvalues for spin-up and spin-down states along a given path.

        Parameters
        ----------
        path : np.ndarray (n_points, n_dimensions)
            The k-point path along which to compute the Hamiltonian and eigenvalues. 
        find_eigsQ : bool, optional
            If True, the eigenvalues will also be computed. Defaults to False.

        Returns
        -------
        bs_spins : np.ndarray(nwa, nkpt, spin)
            Array of band structures or eigenvalues, depending on the value of `find_eigsQ`. 
        hks_spins : np.ndarray(nwa, nwa, nkpt, spin)
            Array of Hamiltonian matrices for each k-point.
        """
        path = path[:, :self.nD]
        band_str_up, hks_up = self.get_hk_eig_path( path, find_eigsQ=find_eigsQ, spin=0)
        band_str_dn, hks_dn = self.get_hk_eig_path( path, find_eigsQ=find_eigsQ, spin=1)
        
        hks_spins = np.transpose( np.array([hks_up, hks_dn]) , (2,3, 1,0)) # (nwa, nwa, nkpt, spin)
        if find_eigsQ:
            bs_spins = np.transpose( np.array([band_str_up, band_str_dn]) , (2,1,0)) # (nwa, nkpt, spin)
            return bs_spins, hks_spins 
        return [], hks_spins 

    
class Wannier_loader_PM(Wannier_loader):
    """
    A class to handle loading and processing of Wannier90 data for paramagnetic hamiltonians.
    
    """

    def __init__(self,  wannier_hr_filename, wannier_dir='wannier'):
        super().__init__() 
        os.makedirs(os.path.abspath(wannier_dir), exist_ok=True)

        self.load_wannier90(wannier_hr_filename, wannier_dir)

    def load_wannier90(self, wannier_hr_filename, wannier_dir='wannier', **kwargs):
        """
        Load Wannier90 Hamiltonian from specified filename.

        Parameters
        ----------
        wannier_hr_filename : str
            Filename for the wannier Hamiltonian. 
        wannier_dir : str, optional
            Directory containing the Wannier90 files. Defaults to 'wannier'.
        **kwargs : Additional keyword arguments.

        """

        file_hr = os.path.join(os.path.abspath(wannier_dir), wannier_hr_filename)
        R_coords, hr = self.load_util(file_hr)
        
        self.complex_hr = hr[:,:,:, np.newaxis] # (nwa, nwa, Rpts, spin=0)
        self.R_coords = R_coords[:, :self.nD]
        


    def get_hk_spins(self, path, find_eigsQ=False):
        """
        Computes the Hamiltonian matrix elements and optionally the eigenvalues along a given path.

        Parameters
        ----------
        path : np.ndarray (n_points, n_dimensions)
            The k-point path along which to compute the Hamiltonian and eigenvalues.
        find_eigsQ : bool, optional
            If True, the eigenvalues will also be computed. Defaults to False.

        Returns
        -------
        bs_spins : np.ndarray(nwa, nkpt, 1)
            Array of band structures or eigenvalues, depending on the value of `find_eigsQ`. 
        hks_spins : np.ndarray(nwa, nwa, nkpt, 1)
            Array of Hamiltonian matrices for each k-point.
        """
        path = path[:, :self.nD]
        band_str, hk = self.get_hk_eig_path( path, find_eigsQ=find_eigsQ, spin=0)
        hks_spins = np.transpose( hk , (2,1,0))[:,:,:, np.newaxis] # (nwa, nwa, nkpt, spin)
        if find_eigsQ:
            bs_spins = np.transpose( np.array([band_str]) , (2,1,0)) # (nwa, nkpt, spin)
            return bs_spins, hks_spins 
        return [], hks_spins 
