"""Sample tau using the ELiCA likelihood with cobaya's MCMC sampler."""

from cobaya.run import run

info = {
    "likelihood": {"elica": None},
    "params": {
        "As": {
            "latex": "A_\\mathrm{s}",
            "value": "lambda logA: 1e-10*np.exp(logA)",
        },
        "H0": 67.32,
        "mnu": 0.06,
        "ns": 0.9651,
        "ombh2": 0.02237,
        "omch2": 0.1201,
        "tau": {
            "latex": "\\tau_\\mathrm{reio}",
            "prior": {"max": 0.8, "min": 0.01},
            "proposal": 0.003,
            "ref": {"dist": "norm", "loc": 0.060, "scale": 0.001},
        },
        "logA": {
            "drop": True,
            "latex": "\\log(10^{10} A_\\mathrm{s})",
            "prior": {"max": 3.91, "min": 2.61},
            "proposal": 0.001,
            "ref": {"dist": "norm", "loc": 3.054, "scale": 0.001},
        },
    },
    "theory": {
        "camb": {
            "extra_args": {
                "lens_potential_accuracy": 1,
                "nnu": 3.044,
                "num_massive_neutrinos": 1,
                "theta_H0_range": [20, 100],
            }
        }
    },
    "sampler": {
        "mcmc": {
            "Rminus1_stop": 0.001,
            "Rminus1_cl_stop": 0.01,
        }
    },
    "output": "chains/elica_tau",
    "force": True,
}

updated_info, sampler = run(info)
