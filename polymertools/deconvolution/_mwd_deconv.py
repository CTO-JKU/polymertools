import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from numpy import linspace, argmin
import matplotlib.pyplot as plt


class _Deconvolution:
    """
    Base class for all deconvolution methods.
    """

    _parameter_constraints = {
        "mode": ["mn_free", "mn_fixed"],
        "active_sites": lambda x: isinstance(x, int) and x > 2,
        "log_m_range": lambda x: x[0] < x[1],
    }

    def _validate_params(self, params):
        for param, value in params.items():
            if param not in self._parameter_constraints:
                raise ValueError(f"Unknown parameter: {param}")

            constraint = self._parameter_constraints[param]
            if callable(constraint):
                if not constraint(value):
                    raise ValueError(f"Parameter '{param}'={value} failed validation.")
            elif isinstance(constraint, list):
                if value not in constraint:
                    raise ValueError(f"Parameter '{param}'={value} is not in {constraint}.")
            elif isinstance(constraint, tuple):
                if value[0] >= value[1]:
                    raise ValueError(f"Parameter '{param}'={value} is not in the valid range.")

    def _normalize(self, log_m, mmd):
        """Normalize the molar mass distribution."""
        return -mmd / trapezoid(y=mmd, x=log_m)  # Negative sign to flip the curve (log_m is decreasing)

    def _compute_initial_guesses(self, active_sites, log_m, mmd):
        """Compute initial guesses for the deconvolution."""
        guesses = []
        for i in range(2, active_sites + 1):
            vector_range = linspace(log_m.min(), log_m.max(), i + 2)
            indices = [argmin(abs(log_m - point)) for point in vector_range[1:-1]]
            weights = mmd[indices]
            molar_ranges = 10 ** (-vector_range[1:-1])  # Convert logarithmic scale back to linear
            relative_weights = weights / weights.sum()
            guesses.append(np.column_stack((relative_weights, molar_ranges)).reshape(-1))
        return guesses

    def _compute_bounds(self, lower_log_m, upper_log_m, length):
        """Compute bounds for the curve fitting."""
        if length <= 0:
            raise ValueError("Length must be a positive integer.")
        lower_bounds = [0 if i % 2 == 0 else 10 ** -upper_log_m for i in range(length)]
        upper_bounds = [1 if i % 2 == 0 else 10 ** -lower_log_m for i in range(length)]
        return lower_bounds, upper_bounds

    def _flory_schulz_cumulative(self, molar_mass, *params):
        """Compute the cumulative Flory-Schulz distribution."""
        num_param_pairs = len(params) // 2
        result = 0
        for i in range(num_param_pairs):
            weight_fraction, molar_ranges = params[2 * i], params[2 * i + 1]
            result += 1 / np.log10(np.exp(1)) * weight_fraction * (molar_mass * molar_ranges) ** 2 * np.exp(
                -molar_mass * molar_ranges)
        return result

    def _flory_schulz_single_sites(self, molar_mass, *params):
        """Compute the Flory-Schulz distribution for single sites."""
        num_param_pairs = len(params) // 2
        return [1 / np.log10(np.exp(1)) * params[2 * i] * (molar_mass * params[2 * i + 1]) ** 2 * np.exp(
            -molar_mass * params[2 * i + 1]) for i in range(num_param_pairs)]

    def _calculate_averages(self, distributions, average_type='weight'):
        """Calculate weight or number averages for the distributions."""
        averages = []
        for distribution in distributions:
            if average_type == 'weight':
                averages.append([np.sum(dist * 10 ** self.log_m) / np.sum(dist) for dist in distribution])
            else:
                averages.append([np.sum(dist) / np.sum(dist / (10 ** self.log_m)) for dist in distribution])
        return averages

    def export_deconvolution(self, path="deconvolution_export.xlsx"):
        """Export the deconvolution results to an Excel file."""
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for i, array in enumerate(self.deconvoluted_distributions):
                log_m_series = pd.Series(self.log_m, name="Log M")
                original_mmd_series = pd.Series(self.mmd, name="Original MMD")
                df = pd.DataFrame(array).transpose()
                df.columns = [f'MMD AS{i + 1}' for i in range(len(array))]
                df['Cumulative MMD'] = df.sum(axis=1)
                sheet_name = f'{i + 2} AS'
                pd.concat([log_m_series, original_mmd_series, df], axis=1).to_excel(writer, sheet_name=sheet_name, index=False)

            padded_weight_averages = [sublist + [np.nan] * (self.active_sites - len(sublist)) for sublist in self.weight_averages]
            padded_number_averages = [sublist + [np.nan] * (self.active_sites - len(sublist)) for sublist in self.number_averages]

            cols = [f'AS{i+1}' for i in range(self.active_sites)]
            pd.DataFrame(padded_weight_averages, columns=cols).to_excel(writer, sheet_name='Mw', index=False)
            pd.DataFrame(padded_number_averages, columns=cols).to_excel(writer, sheet_name='Mn', index=False)

        print(f"Export successfully written to {path}")


class MWDDeconv(_Deconvolution):
    """
    Perform deconvolution of GPC data into molecular weight distribution into Schulz-Flory distribution.
    """

    def __init__(self, mode="mn_free", active_sites=2, log_m_range=(2.8, 7)):
        super().__init__()
        self.mode = mode
        self.active_sites = active_sites
        self.log_m_range = log_m_range

        self.fitted = False
        self.log_m = None
        self.mmd = None
        self.deconvoluted_distributions = None
        self.number_averages = None
        self.weight_averages = None

        self._validate_params({
            "mode": self.mode,
            "active_sites": self.active_sites,
            "log_m_range": self.log_m_range,
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
        self.log_m = log_m
        self.mmd = mmd

        mmd_normalized = self._normalize(log_m, mmd)
        guesses = self._compute_initial_guesses(self.active_sites, log_m, mmd)

        self.deconvoluted_distributions = []
        for guess in guesses:
            popt = curve_fit(self._flory_schulz_cumulative, 10 ** log_m, mmd_normalized, p0=guess,
                             bounds=self._compute_bounds(*self.log_m_range, len(guess)))
            self.deconvoluted_distributions.append(self._flory_schulz_single_sites(10 ** log_m, *popt[0]))

        self.number_averages = self._calculate_averages(self.deconvoluted_distributions, average_type='number')
        self.weight_averages = self._calculate_averages(self.deconvoluted_distributions, average_type='weight')

        self.fitted = True
        return self

    def plot_deconvolution(self):
        """Plot the deconvoluted distributions."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Please call the `fit` method first.")

        num_distributions = len(self.deconvoluted_distributions)
        cols = 2
        rows = (num_distributions + cols - 1) // cols

        fig, ax = plt.subplots(rows, cols, figsize=(15, 6 * rows))
        axs = ax.flatten()

        for i, distribution in enumerate(self.deconvoluted_distributions):
            axs[i].plot(self.log_m, self.mmd, label="Experimental MMD")
            axs[i].plot(self.log_m, np.sum(distribution, axis=0), '--', label="Cumulative MMD")
            for j, dist in enumerate(distribution):
                axs[i].plot(self.log_m, dist, label=f"Site {j + 1}")

            axs[i].set_title(f"{i + 2} Active Sites")
            axs[i].set_xlabel(r"$\log\left(M_w\right)$ / g mol$^{-1}$")
            axs[i].set_ylabel(r"$\frac{\mathrm{d}wt}{\mathrm{d}\log(M_w)}$ / a.u.")
            axs[i].legend()

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"mode={self.mode}, active_sites={self.active_sites}"