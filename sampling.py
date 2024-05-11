from cobaya.log import LoggedError
from cobaya.run import run
from mpi4py import MPI

info = {}
info["likelihood"] = {"elica.EE_100x143": None}
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
        "ref": {"dist": "norm", "loc": 0.060, "scale": 0.0005},
    },
}
info["output"] = ".test/test_simple_sampling"
info["force"] = True
info["resume"] = False
info["debug"] = False
info["sampler"] = {
    "mcmc": {
        "Rminus1_stop": 0.001,
        "Rminus1_cl_stop": 0.01,
    }
}
info["theory"] = {
    "camb": {
        "extra_args": {
            "bbn_predictor": "PArthENoPE_880.2_standard.dat",
            # "halofit_version": "mead",
            # "lens_potential_accuracy": 1,
            # "NonLinear": "NonLinear_both",
            "max_l": 2500,
            "WantTransfer": True,
            "Transfer.high_precision": True,
            "num_nu_massless": 2.046,
            # "share_delta_neff": True,
            "YHe": 0.2454006,
            "num_massive_neutrinos": 1,
            "theta_H0_range": [20, 100],
        }
    }
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

success = False
try:
    upd_info, mcmc = run(info)
    success = True
except LoggedError as err:
    print(err)

success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")