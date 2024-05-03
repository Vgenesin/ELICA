import os
import numpy as np
import pickle

from cobaya.likelihood import Likelihood

class Elica(Likelihood):

    def __init__(
        self,
        datafile: str = None,
    ):

    self.lmin
    self.lmax
    self.nsims
    self.nsp

    self.offset
    self.fiducial

    self.Cldata
    self.covariance

    self.covariance = np.linalg.inv(self.covariance)

    def get_requirements(self):
        return {'Cl': {'ee': self.lmin, self.lmax}}

    def log_likelihood(self, cls_EE):
        r"""
		Test
        """
        Clth = np.tile(cls_EE,self.nsp)



    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)['ee']
        return self.log_likelihood(cls)