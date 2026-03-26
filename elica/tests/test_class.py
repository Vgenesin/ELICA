import numpy as np
from cobaya.model import get_model

from elica import EE_WLxWL


def test_class_instance():
    likelihood = EE_WLxWL()

    assert likelihood.lmin == 2, f"lmin = {likelihood.lmin}"
    assert likelihood.lmax == 30, f"lmax = {likelihood.lmax}"
    assert likelihood.nsims == 500, f"nsims = {likelihood.nsims}"
    assert likelihood.nsp == 1, f"nsp = {likelihood.nsp}"
    assert likelihood._name == "elica.EE_WLxWL", f"_name = {likelihood._name}"


def test_log_likelihood(simple_info):
    model = get_model(simple_info)
    log_like = model.logposterior({"tau": 0.060}).loglikes[0]
    expected =-7.957805490348974
    assert np.isclose(
        log_like, expected, atol=1e-6
    ), f"Obtained log-likelihood {log_like} != {expected}"
