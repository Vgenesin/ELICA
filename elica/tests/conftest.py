import pytest


@pytest.fixture
def simple_info():
    info = {}
    info["likelihood"] = {"elica.EE_WLxWL": None}
    info["params"] = {
        "As": {"value": "lambda tau:  1.884e-09*np.exp(2*tau)"},
        "H0": 67.32,
        "mnu": 0.06,
        "ns": 0.9651,
        "ombh2": 0.02237,
        "omch2": 0.1201,
        "tau": {
            "latex": "\\tau_\\mathrm{reio}",
            "prior": {"max": 0.8, "min": 0.01},
            "proposal": 0.001,
            "ref": 0.060,
        },
    }
    info["output"] = ".test/test_simple_sampling"
    info["force"] = True
    info["resume"] = False
    info["debug"] = False
    info["sampler"] = {
        "mcmc": {
            "max_tries": 1000,
            "Rminus1_stop": 0.001,
            "max_samples": 10,
        }
    }
    info["theory"] = {
        "camb": {
            "extra_args": {
                "lens_potential_accuracy": 1,
                "nnu": 3.044,
                "num_massive_neutrinos": 1,
                "theta_H0_range": [20, 100],
            }
        }
    }
    return info


# @pytest.fixture
# def complete_info():
#     info = {}
#     info["likelihood"] = {
#         "elica.EE_WLxWL": None,
#         # "planck_2018_lowl.EE": None,
#         "planck_2018_lowl.TT": None,
#         "planck_NPIPE_highl_CamSpec.TTTEEE": None,
#         "planckpr4lensing.PlanckPR4Lensing": None,
#     }
#     info["params"] = {
#         "As": {"latex": "A_\\mathrm{s}", "value": "lambda logA: 1e-10*np.exp(logA)"},
#         "H0": {"latex": "H_0", "max": 100, "min": 20},
#         "cosmomc_theta": {
#             "derived": False,
#             "value": "lambda theta_MC_100: 1.e-2*theta_MC_100",
#         },
#         "logA": {
#             "drop": True,
#             "latex": "\\log(10^{10} A_\\mathrm{s})",
#             "prior": {"max": 3.91, "min": 1.61},
#             "proposal": 0.001,
#             "ref": {"dist": "norm", "loc": 3.04478383213, "scale": 0.0001},
#         },
#         "mnu": 0.06,
#         "ns": {
#             "latex": "n_\\mathrm{s}",
#             "prior": {"max": 1.2, "min": 0.8},
#             "proposal": 0.002,
#             "ref": {"dist": "norm", "loc": 0.9660499, "scale": 0.0004},
#         },
#         "ombh2": {
#             "latex": "\\Omega_\\mathrm{b} h^2",
#             "prior": {"max": 0.1, "min": 0.005},
#             "proposal": 0.0001,
#             "ref": {"dist": "norm", "loc": 0.0223828, "scale": 0.00001},
#         },
#         "omch2": {
#             "latex": "\\Omega_\\mathrm{c} h^2",
#             "prior": {"max": 0.99, "min": 0.001},
#             "proposal": 0.0005,
#             "ref": {"dist": "norm", "loc": 0.1201075, "scale": 0.0001},
#         },
#         "tau": {
#             "latex": "\\tau_\\mathrm{reio}",
#             "prior": {"max": 0.8, "min": 0.01},
#             "proposal": 0.003,
#             "ref": 0.060,
#         },
#         "theta_MC_100": {
#             "drop": True,
#             "latex": "100\\theta_\\mathrm{MC}",
#             "prior": {"max": 10, "min": 0.5},
#             "proposal": 0.0002,
#             "ref": {"dist": "norm", "loc": 1.04109, "scale": 0.00004},
#             "renames": "theta",
#         },
#     }
#     info["output"] = ".test/test_complete_sampling"
#     info["force"] = True
#     info["resume"] = False
#     info["debug"] = False
#     info["sampler"] = {
#         "mcmc": {
#             "max_tries": 1000,
#             "Rminus1_stop": 0.001,
#             "max_samples": 10,
#         }
#     }
#     info["theory"] = {
#         "camb": {
#             "extra_args": {
#                 "lens_potential_accuracy": 1,
#                 "nnu": 3.044,
#                 "num_massive_neutrinos": 1,
#                 "theta_H0_range": [20, 100],
#             }
#         }
#     }
#     return info
