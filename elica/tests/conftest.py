def make_cobaya_info(likelihood_name):
    """Build a cobaya info dict for the given likelihood, evaluated at tau=0.060."""
    if likelihood_name == "elica":
        lkl_block = {"elica": None}
    else:
        lkl_block = {f"elica.{likelihood_name}": None}

    return {
        "likelihood": lkl_block,
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
            "tau": 0.060,
            "logA": 3.054,
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
    }
