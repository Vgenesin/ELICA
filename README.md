# ELiCA

E-mode Likelihood for Cross-Analysis: an external [cobaya](https://cobaya.readthedocs.io/) likelihood package for CMB E-mode polarization, targeting constraints on the epoch of reionization.

## Installation

```bash
pip install elica
```

Or from source:

```bash
git clone https://github.com/Vgenesin/ELICA.git
cd ELICA
uv sync
```

You can also use `pip install .` if you don't have [uv](https://docs.astral.sh/uv/).

## Usage

ELiCA provides several likelihoods that can be used directly in cobaya:

| Likelihood | cobaya name | Description |
| --- | --- | --- |
| `elica` | `elica` | Flagship hybrid (cross-spectra + WLxWL) |
| `cross` | `elica.cross` | Cross-spectra only |
| `full` | `elica.full` | All auto + cross spectra |
| `EE_100x100` | `elica.EE_100x100` | Single-field 100GHz auto |
| `EE_100x143` | `elica.EE_100x143` | Single-field 100x143 cross |
| `EE_100xWL` | `elica.EE_100xWL` | Single-field 100xWL cross |
| `EE_143x143` | `elica.EE_143x143` | Single-field 143GHz auto |
| `EE_143xWL` | `elica.EE_143xWL` | Single-field 143xWL cross |
| `EE_WLxWL` | `elica.EE_WLxWL` | Single-field WL auto |

Example cobaya input:

```yaml
likelihood:
  elica:

theory:
  camb:
    extra_args:
      lens_potential_accuracy: 1
      nnu: 3.044
      num_massive_neutrinos: 1
```

See `examples/` for a full sampling script.

## Development

```bash
git clone https://github.com/Vgenesin/ELICA.git
cd ELICA
uv sync --all-extras --dev
pre-commit install
```

Run tests:

```bash
uv run pytest tests/ -v
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format --check .
```
