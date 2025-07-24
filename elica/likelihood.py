import os
import pickle

import numpy as np
from cobaya.likelihoods.base_classes import DataSetLikelihood


class Elica(DataSetLikelihood):
    """
    Abstract class defining the E-mode Likelihood with Cross-correlation
    Analysis (ELICA) likelihood.

    This is meant to be the general-purpose likelihood containing the main
    computations. Then, specific likelihoods can be derived from this one
    by specifying the datafile.

    Attributes
    ----------
        lmin (int):
            define the starting multipole of the fields.
        lmax (int):
            define the maximum multipole of the fields.
        nsims (int):
            number of simulations.
        nsp (int):
            number of fields in the analysis.
        offset (array_like):
            offset needed for the computation of the log likelihood
            (modification to H&L).
        Clfiducial (array_like):
            fiducial spectra for the E mode analysis.
        Cldata (array_like):
            Data from experiments or from simulations.
        inv_cov (array_like):
            inverse of covariance matrix.
    """

    install_options = {}

    def init_params(self, ini):
        self.lmin = ini.int("lmin")
        self.lmax = ini.int("lmax")
        self.nsims = ini.int("number_simulations")
        self.nsp = ini.int("number_fields")
        self.offset = np.loadtxt(ini.relativeFileName("offset_file"))
        self.Clfiducial = np.loadtxt(ini.relativeFileName("fiducial_file"))
        self.Cldata = np.loadtxt(ini.relativeFileName("Cl_file")) 
        self.inv_cov = np.loadtxt(ini.relativeFileName("inv_covariance_matrix_file"))
        self.noise_bias = np.loadtxt(ini.relativeFileName("noise_bias_file"))
        self.check_equal_to_dict()

    def check_equal_to_dict(self):  # TODO: eventually remove this method
        #file_dir ottiene il percorso della directory corrente
        file_dir = os.path.abspath(os.path.dirname(__file__))
        #definisce il dizionario da caicare
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            #assegna il contenuto del file pickle a data
            data = pickle.load(pickle_file)
        assert np.allclose(data.get("lmin"), self.lmin)
        assert np.allclose(data.get("lmax"), self.lmax)
        assert np.allclose(data.get("number_simulations"), self.nsims)
        assert np.allclose(data.get("number_fields"), self.nsp)
        assert np.allclose(data.get("offset"), self.offset)
        assert np.allclose(data.get("fiducial"), self.Clfiducial)
        assert np.allclose(data.get("Cl") , self.Cldata)
        assert np.allclose(data.get("inv_covariance_matrix"), self.inv_cov)
        assert np.allclose(data.get("noise_bias"), self.noise_bias)

    def dict_to_plain_data(self):  # TODO: eventually remove this method
        name_data = self._name.replace("elica.EE_", "")
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        folder = os.path.join(file_dir, f"data/{name_data}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(file_dir, f"data/{name_data}/params.dataset")
        with open(file, "w") as f:
            f.write(f"lmin={data.get('lmin')}\n")
            f.write(f"lmax={data.get('lmax')}\n")
            f.write(f"number_simulations={data.get('number_simulations')}\n")
            f.write(f"number_fields={data.get('number_fields')}\n\n")

            f.write("offset_file=offset.dat\n\n")
            f.write("fiducial_file=fiducial.dat\n\n")
            f.write("Cl_file=Cl.dat\n\n")
            f.write("inv_covariance_matrix_file=inv_covariance_matrix.dat\n\n")

        file = os.path.join(file_dir, f"data/{name_data}/offset.dat")
        np.savetxt(file, data.get("offset"))
        file = os.path.join(file_dir, f"data/{name_data}/fiducial.dat")
        np.savetxt(file, data.get("fiducial"))
        file = os.path.join(file_dir, f"data/{name_data}/Cl.dat")
        np.savetxt(file, data.get("Cl"))
        file = os.path.join(file_dir, f"data/{name_data}/inv_covariance_matrix.dat")
        np.savetxt(file, data.get("inv_covariance_matrix"))
        file = os.path.join(file_dir, f"data/{name_data}/noise_bias.dat")
        np.savetxt(file, data.get("noise_bias"))

    def ghl(self,x):
        return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))

    def glolli(self, x):
        return np.sign(x) * self.ghl(np.abs(x))
    
    def log_likelihood(self, cls_EE):
        X = (self.Clfiducial +self.noise_bias + self.offset) * self.glolli((self.Cldata + self.offset) / (cls_EE +self.noise_bias+ self.offset))
        chi2 = np.dot(X, np.dot(self.inv_cov, X))
        chi2 = self.nsims * np.log((1 + chi2 / (self.nsims - 1)))
        return -chi2*0.5


    def get_requirements(self):
        return {"Cl": {"ee": 1000}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True )["ee"][self.lmin : self.lmax + 1]
        return self.log_likelihood(cls)




# Derivative classes (they need the .yaml file)


class EE_100x143(Elica): ...


class EE_100xWL(Elica): ...


class EE_143xWL(Elica): ...


class EE_WLxWL(Elica): ...


class EE_100x143_100xWL(Elica): ...


class EE_100x143_143xWL(Elica): ...


class EE_100x143_WLxWL(Elica): ...


class EE_100x143_100xWL_143xWL(Elica): ...


class EE_full(Elica): ...

class mHL(DataSetLikelihood):
    
    install_options = {}

    def init_params(self,ini):
        self.lmin = ini.int("lmin")
        self.lmax = ini.int("lmax")
        self.nsims = ini.int("number_simulations")
        self.nsp = ini.int("number_fields")
        self.offset = np.loadtxt(ini.relativeFileName("offset_file"))
        self.Clfiducial = np.loadtxt(ini.relativeFileName("fiducial_file"))
        self.Cl = np.loadtxt(ini.relativeFileName("Cl_file")) 
        self.inv_cov = np.loadtxt(ini.relativeFileName("inv_covariance_matrix_file"))
        self.noise_bias = np.loadtxt(ini.relativeFileName("noise_bias_file"))
        self.clth=np.loadtxt(ini.relativeFileName("clth_file")).reshape(1681,6,29)
        self.check_equal_to_dict()

    def check_equal_to_dict(self):
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        assert np.allclose(data.get("lmin"), self.lmin)
        assert np.allclose(data.get("lmax"), self.lmax)
        assert np.allclose(data.get("number_simulations"), self.nsims)
        assert np.allclose(data.get("number_fields"), self.nsp)
        assert np.allclose(data.get("offset"), self.offset)
        assert np.allclose(data.get("fiducial"), self.Clfiducial)
        assert np.allclose(data.get("Cl") , self.Cl)
        assert np.allclose(data.get("inv_covariance_matrix"), self.inv_cov)
        assert np.allclose(data.get("noise_bias"), self.noise_bias)
        # assert np.allclose(data.get("clth"), self.clth)

    def dict_to_plain_data(self):  # TODO: eventually remove this method
        # name_data = self._name.replace("elica.EE_", "")
        # name_data = self._name.replace("mHL")
        name_data = "mHL"
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        folder = os.path.join(file_dir, f"data/{name_data}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(file_dir, f"data/{name_data}/params.dataset")
        with open(file, "w") as f:
            f.write(f"lmin={data.get('lmin')}\n")
            f.write(f"lmax={data.get('lmax')}\n")
            f.write(f"number_simulations={data.get('number_simulations')}\n")
            f.write(f"number_fields={data.get('number_fields')}\n\n")
            f.write("offset_file=offset.dat\n\n")
            f.write("fiducial_file=fiducial.dat\n\n")
            f.write("Cl_file=Cl.dat\n\n")
            f.write("inv_covariance_matrix_file=inv_covariance_matrix.dat\n\n")
            f.write("noise_bias_file=noise_bias.dat\n\n")
            f.write("clth_file=clth.dat\n\n")


        file = os.path.join(file_dir, f"data/{name_data}/offset.dat")
        np.savetxt(file, data.get("offset"))
        file = os.path.join(file_dir, f"data/{name_data}/fiducial.dat")
        np.savetxt(file, data.get("fiducial"))
        file = os.path.join(file_dir, f"data/{name_data}/Cl.dat")
        np.savetxt(file, data.get("Cl"))
        file = os.path.join(file_dir, f"data/{name_data}/inv_covariance_matrix.dat")
        np.savetxt(file, data.get("inv_covariance_matrix"))
        file = os.path.join(file_dir, f"data/{name_data}/noise_bias.dat")
        np.savetxt(file, data.get("noise_bias"))
        file = os.path.join(file_dir, f"data/{name_data}/clth.dat")
        np.savetxt(file, data.get("clth"))

    def ghl(self,x):
        return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))

    def glolli(self, x):
        return np.sign(x) * self.ghl(np.abs(x))
    
    def chs2idx(self, ch1, ch2, N_chs):
        return ch1 * (2 * N_chs - ch1 + 1) // 2 + (ch2 - ch1)

    def idx2chs(self, idx, N_chs):
        total = N_chs * (N_chs + 1) // 2
        if idx < 0 or idx >= total:
            raise ValueError(
            f"Index {idx} out of bounds for a matrix with {N_chs} channels."
        )

        i = 0
        while (i * (2 * N_chs - i + 1)) // 2 <= idx:
            i += 1
        i -= 1
        elements_before_i = i * (2 * N_chs - i + 1) // 2
        j = i + (idx - elements_before_i)
        return i, j

    def vec2mat(self, vect, N_fields, *, all=True, cross=True):
        cross_idxs = np.array(
            [self.chs2idx(ch1, ch2, N_fields)
         for ch1 in range(N_fields)
         for ch2 in range(ch1 + 1, N_fields)]
    )
        auto_idxs = np.array([self.chs2idx(ch, ch, N_fields) for ch in range(N_fields)])

        if len(vect.shape) == 2:
            mat = np.zeros((vect.shape[0], N_fields, N_fields))
        else:
            mat = np.zeros((1, N_fields, N_fields))
            vect = vect[None, :]

        if all:
            for i in auto_idxs:
                ch1, ch2 = self.idx2chs(i, N_fields)
                mat[:, ch1, ch2] = vect[:, i]

        if cross:
            for i in cross_idxs:
                ch1, ch2 = self.idx2chs(i, N_fields)
                mat[:, ch1, ch2] = mat[:, ch2, ch1] = vect[:, i]

        return np.squeeze(mat)

    def ghl(self,x):
        return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))

    def mat2vec(self, mat, *, all=True, cross=True):
    #N_chs=3
        N_chs = mat.shape[1]
        vec = []
    #range 6
        for idx in range(N_chs * (N_chs + 1) // 2):
            ch1, ch2 = self.idx2chs(idx, N_chs)
            if cross:
                if all:
                    vec.append(mat[:, ch1, ch2])
                else:
                    if ch1 != ch2:
                        vec.append(mat[:, ch1, ch2])
            else:
                if ch1 == ch2:
                    vec.append(mat[:, ch1, ch1])
        return np.array(vec).T
    
    def log_likelihood(self,cls_EE):
        #sulla dimensione 1 ci sono i multipoli
        N_ell = self.Cl.shape[1]
        # print(N_ell)
        self.N_fields=3
        
        x = np.zeros((self.Cl.shape[0],self.Cl.shape[1]))
        
        self.clfidu=(self.Clfiducial+self.noise_bias)
        self.Cross_only_x=np.zeros((3*N_ell))
        self.cltheory = np.zeros((6, N_ell))
        
        for i in range(6):
            self.cltheory[i]= cls_EE+ self.noise_bias[i]

       

        # for isim in range (1681):
        for ell_idx in range(N_ell):
            Off = self.vec2mat(self.offset[:, ell_idx], self.N_fields)
            D = self.vec2mat(self.Cl[:, ell_idx], self.N_fields) + Off[None, :]
            #qui va richiamato cobaya e aggiunto il noise
            # M = self.vec2mat(self.clth[isim, :, ell_idx], self.N_fields) + Off[None, :]
            M = self.vec2mat(self.cltheory[:, ell_idx], self.N_fields) + Off[None, :]

            #qui va aggiunto il rumore
            F = self.vec2mat(self.clfidu[:, ell_idx], self.N_fields) + Off[None, :] 

            w, V = np.linalg.eigh(M[0])
            L = np.einsum("ij,j,kj->ik", V, 1 / np.sqrt(w), V)
            P = np.einsum("ji,njk,kl->nil", L, D, L)

            w, V = np.linalg.eigh(P)
            gg = np.sign(w) * self.ghl(np.abs(w))
            G = np.einsum("nij,nj,nkj->nik", V, gg, V)

            w, V = np.linalg.eigh(F[0])
            L = np.einsum("ij,j,kj->ik", V, np.sqrt(w), V)
            X = np.einsum("ji,njk,kl->nil", L, G, L)
        
            x[:, ell_idx] = self.mat2vec(X)

      
            cross_idxs=[1,2,4]
            

        self.Cross_only_x[:]=((x[cross_idxs, :]).reshape (-1))

        chi2= np.einsum("i,ij,j->", self.Cross_only_x, self.inv_cov, self.Cross_only_x)
        chi2 = -2 * np.log((1 + chi2 / (self.nsims - 1)) ** (-self.nsims / 2))

        return -chi2/2
    
    def get_requirements(self):
        return {"Cl": {"ee": 1000}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True )["ee"][self.lmin : self.lmax + 1]
        return self.log_likelihood(cls)
    




class hybridHL(DataSetLikelihood):
    
    install_options = {}

    def init_params(self,ini):
        self.lmin = ini.int("lmin")
        self.lmax = ini.int("lmax")
        self.nsims = ini.int("number_simulations")
        self.nsp = ini.int("number_fields")
        self.offset = np.loadtxt(ini.relativeFileName("offset_file"))
        self.Clfiducial = np.loadtxt(ini.relativeFileName("fiducial_file"))
        self.Cl = np.loadtxt(ini.relativeFileName("Cl_file")) 
        self.inv_cov = np.loadtxt(ini.relativeFileName("inv_covariance_matrix_file"))
        self.noise_bias = np.loadtxt(ini.relativeFileName("noise_bias_file"))
        # self.clth=np.loadtxt(ini.relativeFileName("clth_file")).reshape(1681,6,29)
        self.check_equal_to_dict()

    def check_equal_to_dict(self):
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        assert np.allclose(data.get("lmin"), self.lmin)
        assert np.allclose(data.get("lmax"), self.lmax)
        assert np.allclose(data.get("number_simulations"), self.nsims)
        assert np.allclose(data.get("number_fields"), self.nsp)
        assert np.allclose(data.get("offset"), self.offset)
        assert np.allclose(data.get("fiducial"), self.Clfiducial)
        assert np.allclose(data.get("Cl") , self.Cl)
        assert np.allclose(data.get("inv_covariance_matrix"), self.inv_cov)
        assert np.allclose(data.get("noise_bias"), self.noise_bias)
        # assert np.allclose(data.get("clth"), self.clth)

    def dict_to_plain_data(self):  # TODO: eventually remove this method
        # name_data = self._name.replace("elica.EE_", "")
        # name_data = self._name.replace("mHL")
        name_data = "hybridHL"
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        folder = os.path.join(file_dir, f"data/{name_data}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(file_dir, f"data/{name_data}/params.dataset")
        with open(file, "w") as f:
            f.write(f"lmin={data.get('lmin')}\n")
            f.write(f"lmax={data.get('lmax')}\n")
            f.write(f"number_simulations={data.get('number_simulations')}\n")
            f.write(f"number_fields={data.get('number_fields')}\n\n")
            f.write("offset_file=offset.dat\n\n")
            f.write("fiducial_file=fiducial.dat\n\n")
            f.write("Cl_file=Cl.dat\n\n")
            f.write("inv_covariance_matrix_file=inv_covariance_matrix.dat\n\n")
            f.write("noise_bias_file=noise_bias.dat\n\n")
            # f.write("clth_file=clth.dat\n\n")


        file = os.path.join(file_dir, f"data/{name_data}/offset.dat")
        np.savetxt(file, data.get("offset"))
        file = os.path.join(file_dir, f"data/{name_data}/fiducial.dat")
        np.savetxt(file, data.get("fiducial"))
        file = os.path.join(file_dir, f"data/{name_data}/Cl.dat")
        np.savetxt(file, data.get("Cl"))
        file = os.path.join(file_dir, f"data/{name_data}/inv_covariance_matrix.dat")
        np.savetxt(file, data.get("inv_covariance_matrix"))
        file = os.path.join(file_dir, f"data/{name_data}/noise_bias.dat")
        np.savetxt(file, data.get("noise_bias"))
        # file = os.path.join(file_dir, f"data/{name_data}/clth.dat")
        # np.savetxt(file, data.get("clth"))

    def ghl(self,x):
        return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))

    def glolli(self, x):
        return np.sign(x) * self.ghl(np.abs(x))
    
    def chs2idx(self, ch1, ch2, N_chs):
        return ch1 * (2 * N_chs - ch1 + 1) // 2 + (ch2 - ch1)

    def idx2chs(self, idx, N_chs):
        total = N_chs * (N_chs + 1) // 2
        if idx < 0 or idx >= total:
            raise ValueError(
            f"Index {idx} out of bounds for a matrix with {N_chs} channels."
        )

        i = 0
        while (i * (2 * N_chs - i + 1)) // 2 <= idx:
            i += 1
        i -= 1
        elements_before_i = i * (2 * N_chs - i + 1) // 2
        j = i + (idx - elements_before_i)
        return i, j

    def vec2mat(self, vect, N_fields, *, all=True, cross=True):
        cross_idxs = np.array(
            [self.chs2idx(ch1, ch2, N_fields)
         for ch1 in range(N_fields)
         for ch2 in range(ch1 + 1, N_fields)]
    )
        auto_idxs = np.array([self.chs2idx(ch, ch, N_fields) for ch in range(N_fields)])

        if len(vect.shape) == 2:
            mat = np.zeros((vect.shape[0], N_fields, N_fields))
        else:
            mat = np.zeros((1, N_fields, N_fields))
            vect = vect[None, :]

        if all:
            for i in auto_idxs:
                ch1, ch2 = self.idx2chs(i, N_fields)
                mat[:, ch1, ch2] = vect[:, i]

        if cross:
            for i in cross_idxs:
                ch1, ch2 = self.idx2chs(i, N_fields)
                mat[:, ch1, ch2] = mat[:, ch2, ch1] = vect[:, i]

        return np.squeeze(mat)

    def ghl(self,x):
        return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))

    def mat2vec(self, mat, *, all=True, cross=True):
    #N_chs=3
        N_chs = mat.shape[1]
        vec = []
    #range 6
        for idx in range(N_chs * (N_chs + 1) // 2):
            ch1, ch2 = self.idx2chs(idx, N_chs)
            if cross:
                if all:
                    vec.append(mat[:, ch1, ch2])
                else:
                    if ch1 != ch2:
                        vec.append(mat[:, ch1, ch2])
            else:
                if ch1 == ch2:
                    vec.append(mat[:, ch1, ch1])
        return np.array(vec).T
    
    def log_likelihood(self,cls_EE):
        #sulla dimensione 1 ci sono i multipoli
        N_ell = self.Cl.shape[1]
        # print(N_ell)
        self.N_fields=3
        
        x = np.zeros((self.Cl.shape[0],self.Cl.shape[1]))
        
        self.clfidu=(self.Clfiducial+self.noise_bias)
        self.hybrid_x=np.zeros((4*N_ell))
        self.cltheory = np.zeros((6, N_ell))
        
        for i in range(6):
            self.cltheory[i]= cls_EE+ self.noise_bias[i]

       

        # for isim in range (1681):
        for ell_idx in range(N_ell):
            Off = self.vec2mat(self.offset[:, ell_idx], self.N_fields)
            D = self.vec2mat(self.Cl[:, ell_idx], self.N_fields) + Off[None, :]
            #qui va richiamato cobaya e aggiunto il noise
            # M = self.vec2mat(self.clth[isim, :, ell_idx], self.N_fields) + Off[None, :]
            M = self.vec2mat(self.cltheory[:, ell_idx], self.N_fields) + Off[None, :]

            #qui va aggiunto il rumore
            F = self.vec2mat(self.clfidu[:, ell_idx], self.N_fields) + Off[None, :] 

            w, V = np.linalg.eigh(M[0])
            L = np.einsum("ij,j,kj->ik", V, 1 / np.sqrt(w), V)
            P = np.einsum("ji,njk,kl->nil", L, D, L)

            w, V = np.linalg.eigh(P)
            gg = np.sign(w) * self.ghl(np.abs(w))
            G = np.einsum("nij,nj,nkj->nik", V, gg, V)

            w, V = np.linalg.eigh(F[0])
            L = np.einsum("ij,j,kj->ik", V, np.sqrt(w), V)
            X = np.einsum("ji,njk,kl->nil", L, G, L)
        
            x[:, ell_idx] = self.mat2vec(X)

      
            cross_idxs=[1,2,4,5]
            

        self.hybrid_x[:]=((x[cross_idxs, :]).reshape (-1))

        chi2= np.einsum("i,ij,j->", self.hybrid_x, self.inv_cov, self.hybrid_x)
        chi2 = -2 * np.log((1 + chi2 / (self.nsims - 1)) ** (-self.nsims / 2))

        return -chi2/2
    
    def get_requirements(self):
        return {"Cl": {"ee": 1000}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True )["ee"][self.lmin : self.lmax + 1]
        return self.log_likelihood(cls)
    



class fullHL(DataSetLikelihood):
    
    install_options = {}

    def init_params(self,ini):
        self.lmin = ini.int("lmin")
        self.lmax = ini.int("lmax")
        self.nsims = ini.int("number_simulations")
        self.nsp = ini.int("number_fields")
        self.offset = np.loadtxt(ini.relativeFileName("offset_file"))
        self.Clfiducial = np.loadtxt(ini.relativeFileName("fiducial_file"))
        self.Cl = np.loadtxt(ini.relativeFileName("Cl_file")) 
        self.inv_cov = np.loadtxt(ini.relativeFileName("inv_covariance_matrix_file"))
        self.noise_bias = np.loadtxt(ini.relativeFileName("noise_bias_file"))
        self.check_equal_to_dict()

    def check_equal_to_dict(self):
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        assert np.allclose(data.get("lmin"), self.lmin)
        assert np.allclose(data.get("lmax"), self.lmax)
        assert np.allclose(data.get("number_simulations"), self.nsims)
        assert np.allclose(data.get("number_fields"), self.nsp)
        assert np.allclose(data.get("offset"), self.offset)
        assert np.allclose(data.get("fiducial"), self.Clfiducial)
        assert np.allclose(data.get("Cl") , self.Cl)
        assert np.allclose(data.get("inv_covariance_matrix"), self.inv_cov)
        assert np.allclose(data.get("noise_bias"), self.noise_bias)

    def dict_to_plain_data(self):  # TODO: 
        name_data = "fullHL"
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        folder = os.path.join(file_dir, f"data/{name_data}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(file_dir, f"data/{name_data}/params.dataset")
        with open(file, "w") as f:
            f.write(f"lmin={data.get('lmin')}\n")
            f.write(f"lmax={data.get('lmax')}\n")
            f.write(f"number_simulations={data.get('number_simulations')}\n")
            f.write(f"number_fields={data.get('number_fields')}\n\n")
            f.write("offset_file=offset.dat\n\n")
            f.write("fiducial_file=fiducial.dat\n\n")
            f.write("Cl_file=Cl.dat\n\n")
            f.write("inv_covariance_matrix_file=inv_covariance_matrix.dat\n\n")
            f.write("noise_bias_file=noise_bias.dat\n\n")


        file = os.path.join(file_dir, f"data/{name_data}/offset.dat")
        np.savetxt(file, data.get("offset"))
        file = os.path.join(file_dir, f"data/{name_data}/fiducial.dat")
        np.savetxt(file, data.get("fiducial"))
        file = os.path.join(file_dir, f"data/{name_data}/Cl.dat")
        np.savetxt(file, data.get("Cl"))
        file = os.path.join(file_dir, f"data/{name_data}/inv_covariance_matrix.dat")
        np.savetxt(file, data.get("inv_covariance_matrix"))
        file = os.path.join(file_dir, f"data/{name_data}/noise_bias.dat")
        np.savetxt(file, data.get("noise_bias"))
        # file = os.path.join(file_dir, f"data/{name_data}/clth.dat")
        # np.savetxt(file, data.get("clth"))

    def ghl(self,x):
        return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))

    def glolli(self, x):
        return np.sign(x) * self.ghl(np.abs(x))
    
    def chs2idx(self, ch1, ch2, N_chs):
        return ch1 * (2 * N_chs - ch1 + 1) // 2 + (ch2 - ch1)

    def idx2chs(self, idx, N_chs):
        total = N_chs * (N_chs + 1) // 2
        if idx < 0 or idx >= total:
            raise ValueError(
            f"Index {idx} out of bounds for a matrix with {N_chs} channels."
        )

        i = 0
        while (i * (2 * N_chs - i + 1)) // 2 <= idx:
            i += 1
        i -= 1
        elements_before_i = i * (2 * N_chs - i + 1) // 2
        j = i + (idx - elements_before_i)
        return i, j

    def vec2mat(self, vect, N_fields, *, all=True, cross=True):
        cross_idxs = np.array(
            [self.chs2idx(ch1, ch2, N_fields)
         for ch1 in range(N_fields)
         for ch2 in range(ch1 + 1, N_fields)]
    )
        auto_idxs = np.array([self.chs2idx(ch, ch, N_fields) for ch in range(N_fields)])

        if len(vect.shape) == 2:
            mat = np.zeros((vect.shape[0], N_fields, N_fields))
        else:
            mat = np.zeros((1, N_fields, N_fields))
            vect = vect[None, :]

        if all:
            for i in auto_idxs:
                ch1, ch2 = self.idx2chs(i, N_fields)
                mat[:, ch1, ch2] = vect[:, i]

        if cross:
            for i in cross_idxs:
                ch1, ch2 = self.idx2chs(i, N_fields)
                mat[:, ch1, ch2] = mat[:, ch2, ch1] = vect[:, i]

        return np.squeeze(mat)

    def ghl(self,x):
        return np.sign(x - 1) * np.sqrt(2.0 * (x - np.log(x) - 1))

    def mat2vec(self, mat, *, all=True, cross=True):
    #N_chs=3
        N_chs = mat.shape[1]
        vec = []
    #range 6
        for idx in range(N_chs * (N_chs + 1) // 2):
            ch1, ch2 = self.idx2chs(idx, N_chs)
            if cross:
                if all:
                    vec.append(mat[:, ch1, ch2])
                else:
                    if ch1 != ch2:
                        vec.append(mat[:, ch1, ch2])
            else:
                if ch1 == ch2:
                    vec.append(mat[:, ch1, ch1])
        return np.array(vec).T
    
    def log_likelihood(self,cls_EE):
        #sulla dimensione 1 ci sono i multipoli
        N_ell = self.Cl.shape[1]
        # print(N_ell)
        self.N_fields=3
        
        x = np.zeros((self.Cl.shape[0],self.Cl.shape[1]))
        
        self.clfidu=(self.Clfiducial+self.noise_bias)
        self.full_x=np.zeros((6*N_ell))
        self.cltheory = np.zeros((6, N_ell))
        
        for i in range(6):
            self.cltheory[i]= cls_EE+ self.noise_bias[i]

       

        # for isim in range (1681):
        for ell_idx in range(N_ell):
            Off = self.vec2mat(self.offset[:, ell_idx], self.N_fields)
            D = self.vec2mat(self.Cl[:, ell_idx], self.N_fields) + Off[None, :]
            #qui va richiamato cobaya e aggiunto il noise
            # M = self.vec2mat(self.clth[isim, :, ell_idx], self.N_fields) + Off[None, :]
            M = self.vec2mat(self.cltheory[:, ell_idx], self.N_fields) + Off[None, :]

            #qui va aggiunto il rumore
            F = self.vec2mat(self.clfidu[:, ell_idx], self.N_fields) + Off[None, :] 

            w, V = np.linalg.eigh(M[0])
            L = np.einsum("ij,j,kj->ik", V, 1 / np.sqrt(w), V)
            P = np.einsum("ji,njk,kl->nil", L, D, L)

            w, V = np.linalg.eigh(P)
            gg = np.sign(w) * self.ghl(np.abs(w))
            G = np.einsum("nij,nj,nkj->nik", V, gg, V)

            w, V = np.linalg.eigh(F[0])
            L = np.einsum("ij,j,kj->ik", V, np.sqrt(w), V)
            X = np.einsum("ji,njk,kl->nil", L, G, L)
        
            x[:, ell_idx] = self.mat2vec(X)

      
            # cross_idxs=[1,2,4,5]
            

        self.full_x[:]=(x.reshape (-1))

        chi2= np.einsum("i,ij,j->", self.full_x, self.inv_cov, self.full_x)
        chi2 = -2 * np.log((1 + chi2 / (self.nsims - 1)) ** (-self.nsims / 2))

        return -chi2/2
    
    def get_requirements(self):
        return {"Cl": {"ee": 1000}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True )["ee"][self.lmin : self.lmax + 1]
        return self.log_likelihood(cls)
    





    
    
class old_Elica(DataSetLikelihood):
    """
    Abstract class defining the E-mode Likelihood with Cross-correlation
    Analysis (ELICA) likelihood.

    This is meant to be the general-purpose likelihood containing the main
    computations. Then, specific likelihoods can be derived from this one
    by specifying the datafile.

    Attributes
    ----------
        lmin (int):
            define the starting multipole of the fields.
        lmax (int):
            define the maximum multipole of the fields.
        nsims (int):
            number of simulations.
        nsp (int):
            number of fields in the analysis.
        offset (array_like):
            offset needed for the computation of the log likelihood
            (modification to H&L).
        Clfiducial (array_like):
            fiducial spectra for the E mode analysis.
        Cldata (array_like):
            Data from experiments or from simulations.
        inv_cov (array_like):
            inverse of covariance matrix.
    """

    install_options = {}

    def init_params(self, ini):
        self.lmin = ini.int("lmin")
        self.lmax = ini.int("lmax")
        self.nsims = ini.int("number_simulations")
        self.nsp = ini.int("number_fields")
        self.offset = np.loadtxt(ini.relativeFileName("offset_file"))
        self.Clfiducial = np.loadtxt(ini.relativeFileName("fiducial_file"))
        self.Clfiducial = np.tile(self.Clfiducial, self.nsp) + self.offset
        self.Cldata = np.loadtxt(ini.relativeFileName("Cl_file")) + self.offset

        # self.inv_cov = np.linalg.inv(
        #     np.loadtxt(ini.relativeFileName("covariance_matrix_file"))
        # )
        self.inv_cov = (
            np.loadtxt(ini.relativeFileName("covariance_matrix_file"))
        )

        self.check_equal_to_dict()

    def check_equal_to_dict(self):  # TODO: eventually remove this method
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        assert np.allclose(data.get("lmin"), self.lmin)
        assert np.allclose(data.get("lmax"), self.lmax)
        assert np.allclose(data.get("number_simulations"), self.nsims)
        assert np.allclose(data.get("number_fields"), self.nsp)

        assert np.allclose(data.get("offset"), self.offset)

        assert np.allclose(
            np.tile(data.get("fiducial"), self.nsp) + self.offset, self.Clfiducial
        )

        assert np.allclose(data.get("Cl") + self.offset, self.Cldata)
        # print("DIFFERENZA MASSIMA:", np.max(np.abs(np.linalg.inv(data.get("Covariance_matrix")) - self.inv_cov)))
        # print("DIFFERENZA MEDIA:", np.mean(np.abs(np.linalg.inv(data.get("Covariance_matrix")) - self.inv_cov)))
        # print("MATRICE 1:", np.linalg.inv(data.get("Covariance_matrix")))
        # print("MATRICE 2:", self.inv_cov)

        # assert np.allclose(np.linalg.inv(data.get("Covariance_matrix")), self.inv_cov)
        assert np.allclose((data.get("Covariance_matrix")), self.inv_cov)

    def dict_to_plain_data(self):  # TODO: eventually remove this method
        name_data = self._name.replace("old_elica.EE_", "")
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.dictionary_file = os.path.join(file_dir, self.dictionary_file)
        with open(self.dictionary_file, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        folder = os.path.join(file_dir, f"data/{name_data}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(file_dir, f"data/{name_data}/params.dataset")
        with open(file, "w") as f:
            f.write(f"lmin={data.get('lmin')}\n")
            f.write(f"lmax={data.get('lmax')}\n")
            f.write(f"number_simulations={data.get('number_simulations')}\n")
            f.write(f"number_fields={data.get('number_fields')}\n\n")

            f.write("offset_file=offset.dat\n\n")
            f.write("fiducial_file=fiducial.dat\n\n")
            f.write("Cl_file=Cl.dat\n\n")
            f.write("covariance_matrix_file=covariance_matrix.dat\n\n")

        file = os.path.join(file_dir, f"data/{name_data}/offset.dat")
        np.savetxt(file, data.get("offset"))

        file = os.path.join(file_dir, f"data/{name_data}/fiducial.dat")
        np.savetxt(file, data.get("fiducial"))

        file = os.path.join(file_dir, f"data/{name_data}/Cl.dat")
        np.savetxt(file, data.get("Cl"))

        file = os.path.join(file_dir, f"data/{name_data}/covariance_matrix.dat")
        np.savetxt(file, data.get("Covariance_matrix"))

    def g(self, x):
        return (
            np.sign(x)
            * np.sign(np.abs(x) - 1)
            * np.sqrt(2.0 * (np.abs(x) - np.log(np.abs(x)) - 1))
        )
    
    #l'offset viene aggiunto alla Cldata e Clfiducial alla lettura
    def log_likelihood(self, cls_EE):
        Clth = np.tile(cls_EE, self.nsp) + self.offset
        diag = self.Cldata / Clth
        Xl = self.Clfiducial * self.g(diag)
        likeSH = self.nsims * np.log(
            1 + np.dot(Xl, np.dot(self.inv_cov, Xl)) / (self.nsims - 1)
        )
        return -likeSH / 2

    def get_requirements(self):
        return {"Cl": {"ee": 1000}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)["ee"][self.lmin : self.lmax + 1]
        return self.log_likelihood(cls)


# Derivative classes (they need the .yaml file)


class EE_old100x143(old_Elica): ...

class EE_oldWLxWL(old_Elica): ...
    
    






