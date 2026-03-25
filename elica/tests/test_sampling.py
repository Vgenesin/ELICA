from cobaya.log import LoggedError
from cobaya.run import run


def test_simple_sampling(simple_info):
    success = False
    try:
        upd_info, mcmc = run(simple_info)
        success = True
    except LoggedError as err:
        print(err)

    if not success:
        print("Sampling failed!")

    return
