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
        npar:
            number of theoretical spectra, i.e the number of spectra with different tau value
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
    self.npar

    self.offset
    self.fiducial

    self.Cldata
    self.covariance

    self.inv_covariance = np.linalg.inv(self.covariance)

    def get_requirements(self):
        return {'Cl': {'ee': self.lmin, self.lmax}}

    def g(x):
        return np.sign(x) * np.sign(np.abs(x) - 1) * np.sqrt(2.0 * (np.abs(x) - np.log(np.abs(x)) - 1))

    def log_likelihood(self, cls_EE):
        """Clth are related to EE spectra that we get out of theory. The lenght of the array is defined by the number of the fields."""
        Clth = np.tile(cls_EE,self.nsp)

        """ Fiducial spectra """
        fid=np.tile(self.fiducial, self.nsp)

        for itau in range(self.npar):
            """Fixing the simulation with a different th smulation"""
            th=Clth[itau]+self.offset

            for isim in range(self.nsims):
                diag = (self.Cldata[isim,:]-self.bias+self.offset)/th
                Xl = (self.fiducial+self.offset)*g(diag)
                likeSH[isim,itau] = -nval/2*(1+np.dot(Xl,np.dot(self.inv_covariance,Xl))/(nval-1))

        return likeSH*(-0.5)


        
    
        


    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)['ee']
        return self.log_likelihood(cls)




