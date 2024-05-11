import pickle

import numpy as np
from cobaya.likelihood import Likelihood


class Elica(Likelihood):
    """
    Abstract class defining the E-mode Likelihood with Cross-correlation
    Analysis (ELICA) likelihood.

    This is meant to be the general-purpose likelihood containing the main
    computations. Then, specific likelihoods can be derived from this one
    by specifying the datafile.

    Parameters
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

    def initialize(self):
        # The datafile is read from the .yaml file
        with open(self.datafile, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        self.lmin = data.get("lmin")
        self.lmax = data.get("lmax")

        self.nsims = data.get("number_simulations")
        self.nsp = data.get("number_fields")

        self.offset = data.get("offset")

        self.Clfiducial = np.tile(data.get("fiducial"), self.nsp) + self.offset
        self.Cldata = data.get("Cl") + self.offset

        self.inv_cov = np.linalg.inv(data.get("Covariance_matrix"))

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
        return {"Cl": {"ee": self.lmax}}

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
