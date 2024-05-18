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
        self.Clfiducial = np.tile(self.Clfiducial, self.nsp) + self.offset

        self.Cldata = np.loadtxt(ini.relativeFileName("Cl_file")) + self.offset

        self.inv_cov = np.linalg.inv(
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

        assert np.allclose(np.linalg.inv(data.get("Covariance_matrix")), self.inv_cov)

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

    def log_likelihood(self, cls_EE):
        Clth = np.tile(cls_EE, self.nsp) + self.offset
        diag = self.Cldata / Clth
        Xl = self.Clfiducial * self.g(diag)
        likeSH = self.nsims * (
            1 + np.dot(Xl, np.dot(self.inv_cov, Xl)) / (self.nsims - 1)
        )
        return -likeSH / 2

    def get_requirements(self):
        return {"Cl": {"ee": 1000}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)["ee"][self.lmin : self.lmax + 1]
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
