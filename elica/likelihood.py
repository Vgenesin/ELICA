import numpy as np
from cobaya.likelihoods.base_classes import CMBlikes
from scipy.linalg import sqrtm


class _ElicaCMBlikes(CMBlikes):
    """Private base class for ELiCA likelihoods.

    Extends CMBlikes with:
    - Offset: additive correction to data, noise, and fiducial
    - new ghl transform: handles negative eigenvalues in the HL transform
    - Sellentin-Heavens correction: accounts for finite simulation covariance
    """

    install_options = {}

    def init_params(self, ini):
        """
        Read number of simulations, then add offset to bandpowers, noise, and fiducial.
        """
        self.nsims = ini.int("number_simulations")
        super().init_params(ini)

        # Read offset and bake it into data arrays
        self.offset = self.read_cl_array(ini, "offset")
        self.bandpowers += self.offset
        if self.cl_noise is not None:
            self.cl_noise += self.offset
        if self.cl_fiducial is not None:
            self.cl_fiducial += self.offset

        # Rebuild per-bin matrices with offset included
        for b in range(self.nbins_used):
            self.elements_to_matrix(
                self.bandpowers[:, b], self.bandpower_matrix[b, :, :]
            )
            if self.cl_noise is not None:
                self.elements_to_matrix(self.cl_noise[:, b], self.noise_matrix[b, :, :])
            if self.cl_fiducial is not None:
                self.elements_to_matrix(
                    self.cl_fiducial[:, b], self.fiducial_sqrt_matrix[b, :, :]
                )
                self.fiducial_sqrt_matrix[b, :, :] = sqrtm(
                    self.fiducial_sqrt_matrix[b, :, :]
                )

    def ReadCovmat(self, ini):
        """Override for unbinned covariance (upstream only supports binned)."""
        covmat_cl = ini.string("covmat_cl", allowEmpty=False)
        self.full_cov = np.loadtxt(ini.relativeFileName("covmat_fiducial"))
        covmat_scale = ini.float("covmat_scale", 1.0)

        cl_in_index = self.UseString_to_cols(covmat_cl)
        self.ncl_used = np.sum(cl_in_index >= 0)
        self.cl_used_index = np.zeros(self.ncl_used, dtype=int)
        cov_cl_used = np.zeros(self.ncl_used, dtype=int)
        ix = 0
        for i, index in enumerate(cl_in_index):
            if index >= 0:
                self.cl_used_index[ix] = index
                cov_cl_used[ix] = i
                ix += 1

        if self.binned:
            return super().ReadCovmat(ini)

        # Unbinned: covariance is already (ncl_used * nbins_used) x (...)
        # Just apply scaling and return
        num_in = len(cl_in_index)
        n_total = self.nbins_used * self.ncl_used
        pcov = np.empty((n_total, n_total))

        for binx in range(self.nbins_used):
            for biny in range(self.nbins_used):
                pcov[
                    binx * self.ncl_used : (binx + 1) * self.ncl_used,
                    biny * self.ncl_used : (biny + 1) * self.ncl_used,
                ] = (
                    covmat_scale
                    * self.full_cov[
                        np.ix_(
                            binx * num_in + cov_cl_used,
                            biny * num_in + cov_cl_used,
                        )
                    ]
                )
        return pcov

    @staticmethod
    def transform(C, Chat, Cfhalf):
        """HL transformation with gLoLLi extension for negative eigenvalues."""
        if C.shape[0] == 1:
            rat = Chat[0, 0] / C[0, 0]
            abs_rat = np.abs(rat)
            ghl_val = np.sqrt(2 * np.maximum(0, abs_rat - np.log(abs_rat) - 1))
            C[0, 0] = np.sign(rat) * np.sign(abs_rat - 1) * ghl_val * Cfhalf[0, 0] ** 2
            return
        diag, U = np.linalg.eigh(C)
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        diag, rot = np.linalg.eigh(rot)
        abs_diag = np.abs(diag)
        diag = (
            np.sign(diag)
            * np.sign(abs_diag - 1)
            * np.sqrt(2 * np.maximum(0, abs_diag - np.log(abs_diag) - 1))
        )
        Cfhalf.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        rot.dot(U.T, C)

    def log_likelihood(self, dls, **data_params):
        """Override to add Sellentin-Heavens correction."""
        self.get_theory_map_cls(dls, data_params)
        C = np.empty((self.nmaps, self.nmaps))
        big_x = np.empty(self.nbins_used * self.ncl_used)
        vecp = np.empty(self.ncl)

        if self.binned:
            binned_theory = self.get_binned_map_cls(self.map_cls)
        else:
            Cs = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
            for i in range(self.nmaps):
                for j in range(i + 1):
                    CL = self.map_cls[i, j]
                    if CL is not None:
                        Cs[:, i, j] = CL.CL[
                            self.bin_min - self.pcl_lmin : self.bin_max
                            - self.pcl_lmin
                            + 1
                        ]
                        Cs[:, j, i] = CL.CL[
                            self.bin_min - self.pcl_lmin : self.bin_max
                            - self.pcl_lmin
                            + 1
                        ]

        for b in range(self.nbins_used):
            if self.binned:
                self.elements_to_matrix(binned_theory[b, :], C)
            else:
                C[:, :] = Cs[b, :, :]
            if self.cl_noise is not None:
                C += self.noise_matrix[b]
            try:
                self.transform(
                    C, self.bandpower_matrix[b], self.fiducial_sqrt_matrix[b]
                )
            except np.linalg.LinAlgError:
                self.log.debug("Likelihood computation failed.")
                return -np.inf
            self.matrix_to_elements(C, vecp)
            big_x[b * self.ncl_used : (b + 1) * self.ncl_used] = vecp[
                self.cl_used_index
            ]

        chi2 = self._fast_chi_squared(self.covinv, big_x)

        # Sellentin-Heavens correction
        chi2 = self.nsims * np.log(1 + chi2 / (self.nsims - 1))

        return -0.5 * chi2


# Concrete likelihood classes (configured via .yaml + .dataset)


class elica(_ElicaCMBlikes): ...


class cross(_ElicaCMBlikes): ...


class full(_ElicaCMBlikes): ...


class EE_100x100(_ElicaCMBlikes): ...


class EE_100x143(_ElicaCMBlikes): ...


class EE_100xWL(_ElicaCMBlikes): ...


class EE_143x143(_ElicaCMBlikes): ...


class EE_143xWL(_ElicaCMBlikes): ...


class EE_WLxWL(_ElicaCMBlikes): ...
