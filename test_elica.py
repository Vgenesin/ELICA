import time

from cobaya.log import LoggedError
from cobaya.run import run
from mpi4py import MPI

from elica import EE_100x143


def test_class_instance():

    likelihood = EE_100x143()

    print(f"lmin = {likelihood.lmin}")
    print(f"lmax = {likelihood.lmax}")
    print(f"nsims = {likelihood.nsims}")
    print(f"nsp = {likelihood.nsp}")
    print(f"name of the likelihood = {likelihood._name}")

    return


def test_simple_sampling():

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
            "max_tries": 1000,
            "Rminus1_stop": 0.001,
            "max_samples": 1000,
        }
    }
    info["theory"] = {
        "camb": {
            "extra_args": {
                "bbn_predictor": "PArthENoPE_880.2_standard.dat",
                "halofit_version": "mead",
                "lens_potential_accuracy": 1,
                "NonLinear": "NonLinear_both",
                "max_l": 2700,
                "WantTransfer": True,
                "Transfer.high_precision": True,
                "parameterization": 2,
                "num_nu_massless": 2.046,
                "share_delta_neff": True,
                "YHe": 0.2454006,
                "pivot_tensor": 0.05,
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

    return


if __name__ == "__main__":
    print("Running tests...")
    test_class_instance()
    print("Class instance test passed.\n")

    print("Running simple sampling test...")
    start = time.time()
    test_simple_sampling()
    end = time.time()

    print(f"******** ALL DONE IN {round(end-start, 2)} SECONDS! ********")
