from cobaya.log import LoggedError
from cobaya.run import run
from mpi4py import MPI


def test_simple_sampling(simple_info):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    success = False
    try:
        upd_info, mcmc = run(simple_info)
        success = True
    except LoggedError as err:
        print(err)

    success = all(comm.allgather(success))

    if not success and rank == 0:
        print("Sampling failed!")

    return
