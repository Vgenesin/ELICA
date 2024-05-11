from cobaya.model import get_model
import numpy as np

from elica import EE_100x143


def test_class_instance():

    likelihood = EE_100x143()

    assert likelihood.lmin == 2, f"lmin = {likelihood.lmin}"
    assert likelihood.lmax == 29, f"lmax = {likelihood.lmax}"
    assert likelihood.nsims == 500, f"nsims = {likelihood.nsims}"
    assert likelihood.nsp == 1, f"nsp = {likelihood.nsp}"
    assert likelihood._name == "elica.EE_100x143", f"_name = {likelihood._name}"


def test_log_likelihood(simple_info):

    model = get_model(simple_info)
    log_like = model.logposterior({"tau": 0.060}).loglikes[0]
    assert np.isclose(
        log_like, -260.99, atol=1e-4
    ), f"Obtained log-likelihood {log_like} != -260.99"
