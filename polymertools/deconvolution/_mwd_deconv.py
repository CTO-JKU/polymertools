import numpy as np
from scipy.integrate import trapezoid
from numpy import linspace, argmin


class _Deconvolution:
    """
    Base class for all deconvolution methods.
    """

    _parameter_constraints = {
        "mode": ["mn_free", "mn_fixed"],
        "active_sites": lambda x: isinstance(x, int) and x > 2,
    }

    def _validate_params(self, params):
        for param, value in params.items():
            if param not in self._parameter_constraints:
                raise ValueError(f"Unknown parameter: {param}")

            constraint = self._parameter_constraints[param]
            if callable(constraint):  # Check if active_sites is an integer greater than 2
                if not constraint(value):
                    raise ValueError(
                        f"Parameter '{param}'={value} failed validation."
                    )
            elif isinstance(constraint, list):  # Check if mode is in the list of valid modes
                if value not in constraint:
                    raise ValueError(
                        f"Parameter '{param}'={value} is not in {constraint}."
                    )

    @staticmethod
    def _normalize(log_m, mmd):
        return -mmd / trapezoid(y=mmd, x=log_m) # Negative sign to flip the curve (log_m is decreasing)


class MWDDeconv(_Deconvolution):
    """
    Perform deconvolution of GPC data into molecular weight distribution into Schulz-Flory distribution.
    """

    def __init__(self, mode="mn_free", active_sites=2):
        self.mode = mode
        self.active_sites = active_sites

        # Validate parameters
        self._validate_params({
            "mode": self.mode,
            "active_sites": self.active_sites,
        })

    def fit(self, log_m, mmd):
        """Fit model to data.

        Parameters
        ----------
        log_m : array-like of shape (n_datapoints,)
                Logarithmic molar mass, where `n_datapoints` is the number of recorded datapoints

        mmd : array-like of shape (n_datapoints,)
                Molar mass distribution data, where `n_datapoints` is the number of recorded datapoints

        Returns
        -------
        self : object
            Fitted model.
        """

        # Normalize data
        mmd_normalized = self._normalize(log_m, mmd)

        # Create the initial guess for non-linear least squares optimization
        for i in range(2, self.active_sites+1):
            vect_rng = linspace(log_m.min(), log_m.max(), i+2)

            wf_idx = np.array([])
            rng = np.array([])
            for j in range(1,i+1):
                min_idx = argmin(abs(log_m - vect_rng[j]))
                wf_idx = np.append(wf_idx, mmd[min_idx])
                rng = np.append(rng, 10**(-vect_rng[j]))

            mmd_rel = wf_idx / wf_idx.sum()
            param = np.vstack((mmd_rel, rng))

    def __repr__(self):
        return (
            f"mode={self.mode}, "
            f"active_sites={self.active_sites})"
        )
