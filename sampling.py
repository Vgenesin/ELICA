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
        # "ref": 0.060,
    },
}
info["output"] = ".test/EE_100x143"
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
            "lens_potential_accuracy": 1,
            "nnu": 3.044,
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
