# ELICA
This repository is dedicated to the likelihood code for E mode polarisation
study with the aim of constraining the epoch of reioization

## Development

The code is tested with `pytest`. To run the tests, simply runS
```bash
pytest .
```
from the root directory of the repository.

Also, the code is formatted with `ruff`. To check the formatting of the code, simply run
```bash
ruff check .
```

To format the code, you can run
```bash
ruff format --diff .
```

The formatting is also encoded in a pre-commit hook. To install the pre-commit hook, run
```bash
pip install pre-commit; pre-commit install
```

This will run the `ruff` commands before each commit.
This will ensure that the code is always formatted correctly and uniformly.
