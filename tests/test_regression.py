import numpy as np
import pytest
from cobaya.model import get_model
from conftest import make_cobaya_info

REFERENCE_VALUES = {
    "EE_100x100": -1.694695323446801e01,
    "EE_100x143": -1.182303223078453e01,
    "EE_100xWL": -1.248755432596433e01,
    "EE_143x143": -1.358281344335101e01,
    "EE_143xWL": -1.280549327691968e01,
    "EE_WLxWL": -6.984066728707005e00,
    "elica": -6.000384701833366e01,
    "cross": -3.113067960873551e01,
    "full": -1.382444802878525e02,
}


@pytest.mark.parametrize(
    "likelihood_name, expected_loglike",
    list(REFERENCE_VALUES.items()),
    ids=list(REFERENCE_VALUES.keys()),
)
def test_loglike_at_tau_0060(likelihood_name, expected_loglike):
    info = make_cobaya_info(likelihood_name)
    model = get_model(info)
    loglike = model.loglikes({})[0][0]
    assert np.isclose(loglike, expected_loglike, atol=1e-6), (
        f"{likelihood_name}: got {loglike}, expected {expected_loglike}"
    )
