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
        fiducial:
            fiducial spectra for the E mode analysis
        Cldata:
            Data from experiments or from simulations
        Covariance:
            Covariance matrix. 

    """
    def __init__(
        self,
         : str = None,
    ):

    self.lmin
    self.lmax
    self.nsims
    self.nsp

    self.offset
    self.fiducial

    self.Cldata
    self.covariance

    """ Fiducial spectra """
    self.fiducial=np.tile(self.fiducial, self.nsp)
    
    self.inv_covariance = np.linalg.inv(self.covariance)
    
    """ file lettura pickle
    def get_fiducial_spectra(self):

    with open(self.cl_file, "rb") as pickle_file:
                return pickle.load(pickle_file)"""

    def g(x):
        return np.sign(x) * np.sign(np.abs(x) - 1) * np.sqrt(2.0 * (np.abs(x) - np.log(np.abs(x)) - 1))

    def log_likelihood(self, cls_EE):
        """Clth are related to EE spectra that we get out of theory. The lenght of the array is defined by the number of the fields."""
        Clth = np.tile(cls_EE,self.nsp)

        diag = (self.Cldata+self.offset)/th
        Xl = (self.fiducial+self.offset)*g(diag)
        likeSH = -self.nsims/2*(1+np.dot(Xl,np.dot(self.inv_covariance,Xl))/(self.nsims-1))

        return likeSH*(-0.5)


    def get_requirements(self):
        return {'Cl': {'ee': self.lmin, self.lmax}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)['ee']
        return self.log_likelihood(cls)




