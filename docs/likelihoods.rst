Likelihoods
===========

ELiCA provides 9 likelihoods, all inheriting from a common base class that extends
cobaya's ``CMBlikes`` with the gLoLLi transform and Sellentin-Heavens correction.

Multi-field likelihoods
-----------------------

These operate on the full 3-map system (100GHz, 143GHz, WL) and differ in which
spectra enter the chi-squared:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - cobaya name
     - Class
     - Description
   * - ``elica``
     - :class:`elica.elica`
     - Flagship hybrid: cross-spectra + WLxWL (4 spectra in chi-squared)
   * - ``elica.cross``
     - :class:`elica.cross`
     - Cross-spectra only (3 spectra in chi-squared)
   * - ``elica.full``
     - :class:`elica.full`
     - All 6 auto + cross spectra in chi-squared

All three apply the HL transform to the full 3x3 spectral matrix. The ``covmat_cl``
parameter in the dataset file controls which spectra are retained in the data vector
after the transform — spectra not listed are effectively marginalized over.

Single-field likelihoods
------------------------

Each operates on a single map pair with a 1x1 spectral matrix:

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - cobaya name
     - Description
   * - ``elica.EE_100x100``
     - 100GHz auto-spectrum
   * - ``elica.EE_100x143``
     - 100GHz x 143GHz cross-spectrum
   * - ``elica.EE_100xWL``
     - 100GHz x WL cross-spectrum
   * - ``elica.EE_143x143``
     - 143GHz auto-spectrum
   * - ``elica.EE_143xWL``
     - 143GHz x WL cross-spectrum
   * - ``elica.EE_WLxWL``
     - WL auto-spectrum

Method
------

The likelihood computation follows:

1. **Hamimeche & Lewis (HL) transform** — Gaussianizes the power spectrum data via
   eigendecomposition. ELiCA uses a modified gHL computation following
   `Mangilli et al. (2015) <https://doi.org/10.1093/mnras/stv1733>`_
   that handles negative eigenvalues via ``sign(x) * gHL(|x|)`` instead of clipping
   them to zero.

2. **Offset** — An additive correction baked into the data, noise, and fiducial
   spectra before the HL transform to ensure positive-definiteness.

3. **Sellentin-Heavens correction** — Accounts for the finite number of simulations
   used to estimate the covariance matrix:

   .. math::

      \chi^2_{\mathrm{SH}} = N_{\mathrm{sims}} \ln\left(1 + \frac{\chi^2}{N_{\mathrm{sims}} - 1}\right)

4. **Spectrum marginalization** — The HL transform operates on the full spectral
   matrix, but only a subset of the transformed spectra enters the chi-squared.
   This effectively marginalizes over the excluded spectra, following the approach
   in `Galloni et al. (2025) <https://doi.org/10.1088/1475-7516/2025/12/052>`_.
