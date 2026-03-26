import numpy as np
import pytest
from cobaya.model import get_model
from conftest import make_cobaya_info

REFERENCE_VALUES = {
    "EE_100x100": -1.694695318749717e01,
    "EE_100x143": -1.787578541935289e01,
    "EE_100xWL": -3.521580876440604e01,
    "EE_143x143": -1.358281326011380e01,
    "EE_143xWL": -4.645672462630856e01,
    "EE_WLxWL": -6.984066776490811e00,
    "elica": -6.000384701848693e01,
    "cross": -4.939219597073087e01,
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
