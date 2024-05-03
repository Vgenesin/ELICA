import os
import numpy as np
import pickle

from cobaya.likelihood import Likelihood

class Elica(Likelihood):

    def __init__(
        self,
        name: str = None,
        datafile: str = None,
        lmin: int = 2,
        lmax: int = 29,
    ):	

    def get_requirements(self):
        return {'Cl': {'ee': self.lmax}}

    def log_likelihood(self, cls_EE):
        r"""
		Test
        """
        
    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)['ee']
        return self.log_likelihood(cls)