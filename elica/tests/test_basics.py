import os
import pytest
from elica.likelihood import mHL

class DummyIni:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.params = {
            "lmin": 2,
            "lmax": 30,
            "number_simulations": 500,
            "number_fields": 3,
            "offset_file": "offset.dat",
            "fiducial_file": "fiducial.dat",
            "Cl_file": "Cl.dat",
            "inv_covariance_matrix_file": "inv_covariance_matrix.dat",
            "noise_bias_file": "noise_bias.dat",
            # "clth_file": "clth.dat",
            "dictionary_file": "mhl_dict.pickle"
        }

    def int(self, key):
        return self.params[key]
    
    def relativeFileName(self, key):
        return os.path.join(self.base_dir, self.params[key])

@pytest.fixture
def data_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/mHL"))

def test_check_equal_to_dict(data_dir):
    ini = DummyIni(data_dir)
    info = {"dataset_file": os.path.join(data_dir, "params.dataset"),
            "dictionary_file":"/Users/valentinagenesini/Documents/thesis/ELICA/elica/data/mhl_dict.pickle"}

    likelihood = mHL(info)

    likelihood.dictionary_file = "/Users/valentinagenesini/Documents/thesis/ELICA/elica/data/mhl_dict.pickle"  # necessario per il path nel metodo
    likelihood.init_params(ini)
    # Il test passa se non viene sollevato AssertionError
    likelihood.check_equal_to_dict()