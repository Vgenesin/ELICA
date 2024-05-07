import os
import numpy as np
import pickle

from cobaya.likelihood import Likelihood


class Elica(Likelihood):

    """
    Defining the parameters needed in the likelihood:

        lmin:
            define the starting multipole of the fields.
        lmax:
            define the maximum multipole of the fields.
        nsims:
            number of simulations.
        nsp:
            number of fields in the analysis
        offset:
            offset needed for the computation of the log likelihood (modification to H&L )
        Clfiducial:
            fiducial spectra for the E mode analysis
        Cldata:
            Data from experiments or from simulations
        inv_cov:
            inverse of covariance matrix.
    """

    def __init__(self, datafile):
        with open(datafile, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        self.lmin = data.get("lmin")
        self.lmax = data.get("lmax")

        self.nsims = data.get("number_simulations")
        self.nsp = data.get("number_fields")

        self.offset = data.get("offset")

        self.Clfiducial = np.tile(data.get("fiducial"), self.nsp) + self.offset
        self.Cldata = data.get("Cl") + self.offset

        self.inv_cov = np.linalg.inv(data.get("Covariance_matrix"))

    def g(x):
        return (
            np.sign(x)
            * np.sign(np.abs(x) - 1)
            * np.sqrt(2.0 * (np.abs(x) - np.log(np.abs(x)) - 1))
        )

    def log_likelihood(self, cls_EE):
        Clth = np.tile(cls_EE, self.nsp) + self.offset
        diag = self.Cldata / Clth
        Xl = self.Clfiducial * g(diag)
        likeSH = (
            -self.nsims
            / 2
            * (1 + np.dot(Xl, np.dot(self.inv_cov, Xl)) / (self.nsims - 1))
        )

        return -0.5 * likeSH

    def get_requirements(self):
        return {'Cl': {'ee': self.lmin, self.lmax}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)["ee"]
        return self.log_likelihood(cls)
