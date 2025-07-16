import os
import pytest
import pickle
import numpy as np
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
            "dictionary_file": "mhl_dict.pickle"
        }
    def int(self, key):
        return self.params[key]
    def relativeFileName(self, key):
        return os.path.join(self.base_dir, self.params[key])

@pytest.fixture
def mock_dictionary_file(tmp_path):
    # Create a mock dictionary file with valid data
    mock_file = tmp_path / "mhl_dict.pickle"
    mock_data = {
        "lmin": 2,
        "lmax": 30,
        "number_simulations": 500,
        "number_fields": 3,
        "offset": np.random.rand(10),
        "fiducial": np.random.rand(10),
        "Cl": np.random.rand(10, 10),
        "inv_covariance_matrix": np.random.rand(10, 10),
        "noise_bias": np.random.rand(10)
    }
    with open(mock_file, "wb") as f:
        pickle.dump(mock_data, f)
    return mock_file

def test_dict_to_plain_data(tmp_path, mock_dictionary_file):
    # Mock ini object
    ini = DummyIni(tmp_path)
    # info = {"dictionary_file": str(mock_dictionary_file)}
    info = {"dataset_file": str(mock_dictionary_file),
            "dictionary_file":str(mock_dictionary_file)}

    # Initialize likelihood
    likelihood = mHL(info)
    likelihood.dictionary_file = str(mock_dictionary_file)

    # Test dict_to_plain_data
    likelihood.dict_to_plain_data()

    # Verify output files
    output_dir = tmp_path / "data/mHL"
    for fname in [
        "offset.dat", "fiducial.dat", "Cl.dat",
        "inv_covariance_matrix.dat", "noise_bias.dat"
    ]:
        assert (output_dir / fname).exists()

    # Check that Cl.dat is 2D
    cl_data = np.loadtxt(output_dir / "Cl.dat")
    assert cl_data.ndim == 2