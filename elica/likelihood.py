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
    cl_file = "/Users/valentinagenesini/Documents/GitHub/ELICA/data/100x143_100xWL_143xWL_dict.pickle"
    with open(_cl_file, "rb") as pickle_file:
        self.fiduCLS= pickle.load(pickle_file)

    def __init__(self, fiduCLS):
        self.lmin=fiduCLS.get('lmin')
        self.lmax=fiduCLS.get('lmax')
        self.nsims=fiduCLS.get('number_simulations')
        self.nsp=fiduCLS.get('number_fields')

        self.offset=fiduCLS.get('offset')
        self.fiducial=fiduCLS.get('fiducial')

        self.th=fiduCLS.get('theory')
        self.Cldata=fiduCLS.get('Cl')
        self.covariance=fiduCLS.get('Covariance_matrix')

        self.inv_covariance = np.linalg.inv(self.covariance)  
        """ Fiducial spectra """
        self.fiducial=np.tile(self.fiducial, self.nsp) 
    


"""my_instance = Elica(fiduCLS)"""


    def g(x):
        return np.sign(x) * np.sign(np.abs(x) - 1) * np.sqrt(2.0 * (np.abs(x) - np.log(np.abs(x)) - 1))

    def log_likelihood(self, cls_EE):
        Clth = np.tile(cls_EE,self.nsp)
        diag = (self.Cldata+self.offset)/(self.th+self.offset)
        Xl = (self.fiducial+self.offset)*g(diag)
        likeSH = -self.nsims/2*(1+np.dot(Xl,np.dot(self.inv_covariance,Xl))/(self.nsims-1))

        return likeSH*(-0.5)


    def get_requirements(self):
        return {'Cl': {'ee': self.lmin, self.lmax}}

    def logp(self, **params_values):
        cls = self.provider.get_Cl(ell_factor=True)['ee']
        return self.log_likelihood(cls)




