import numpy as np
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
            # Check if the first element in log_m_range is smaller
            elif isinstance(constraint, tuple):
                if value[0] >= value[1]:
                    raise ValueError(
                        f"Parameter '{param}'={value} is not in the valid range first element < second element."
                    )

    @staticmethod
    def _normalize(log_m, mmd):
        return -mmd / trapezoid(y=mmd, x=log_m) # Negative sign to flip the curve (log_m is decreasing)

    @staticmethod
    def _compute_initial_guesses(active_sites, log_m, mmd):
        guesses = []
        for i in range(2, active_sites + 1):
            # Generate equally spaced vector in the log scale
            vector_range = linspace(log_m.min(), log_m.max(), i + 2)

            # Find the closest indices and corresponding values
            indices = [argmin(abs(log_m - point)) for point in vector_range[1:-1]]
            weights = mmd[indices]
            molar_ranges = 10 ** (-vector_range[1:-1])  # Convert logarithmic scale back to linear

            # Normalize weights
            relative_weights = weights / weights.sum()

            # Stack parameters for this site configuration (weights + ranges)
            guesses.append(np.column_stack((relative_weights, molar_ranges)).reshape(-1))

        return guesses

    @staticmethod
    def _compute_bounds(lower_log_m, upper_log_m, length):
        """
        Generates and prints two arrays:
        - One alternating between 'a' and 'b'
        - Another alternating between 'c' and 'd'

        The length of the sequences is determined by the input integer `length`.

        Parameters:
            length (int): The length of each sequence.
        """
        if length <= 0:
            print("Length must be a positive integer.")
            return

        # Generate arrays
        lower_bounds = [0 if i % 2 == 0 else 10 ** -upper_log_m  for i in range(length)]
        upper_bounds = [1 if i % 2 == 0 else 10 ** -lower_log_m for i in range(length)]

        return lower_bounds, upper_bounds

    @staticmethod
    def _flory_schulz_cumulative(molar_mass, *params):
        #
        num_param_pairs = len(params) // 2

        result = 0
        for i in range(num_param_pairs):
            weight_fraction, molar_ranges = params[2 * i], params[2 * i + 1]
            result += 1 / np.log10(np.exp(1)) * weight_fraction * (molar_mass * molar_ranges) ** 2 * np.exp(
                -molar_mass * molar_ranges)

        return result

    @staticmethod
    def _flory_schulz_single_sites(molar_mass, *params):
        #
        num_param_pairs = len(params) // 2

        result = []
        for i in range(num_param_pairs):
            weight_fraction, molar_ranges = params[2 * i], params[2 * i + 1]
            result.append(1 / np.log10(np.exp(1)) * weight_fraction * (molar_mass * molar_ranges) ** 2 * np.exp(
                -molar_mass * molar_ranges))

        return result




class MWDDeconv(_Deconvolution):
    """
    Perform deconvolution of GPC data into molecular weight distribution into Schulz-Flory distribution.
    """

    def __init__(self, mode="mn_free", active_sites=2, log_m_range=(2.8, 7)):
        self.mode = mode
        self.active_sites = active_sites
        self.log_m_range = log_m_range

        self.fitted = False
        self.log_m = None
        self.mmd = None
        self.deconvoluted_distributions = None


        # Validate parameters
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



        # Normalize data
        mmd_normalized = self._normalize(log_m, mmd)

        guesses = self._compute_initial_guesses(self.active_sites, log_m, mmd)

        self.deconvoluted_distributions = []
        for guess in guesses:

            popt = curve_fit(self._flory_schulz_cumulative,
                                  10 ** log_m,
                                  mmd_normalized,
                                  p0=guess,
                                  bounds=self._compute_bounds(*self.log_m_range, len(guess))
                          )
            self.deconvoluted_distributions.append(self._flory_schulz_single_sites(10 ** log_m, *popt[0]))

            self.fitted = True
            self.log_m = log_m
            self.mmd = mmd

    def plot_deconvolution(self):
        if self.fitted:
            num_distributions = len(self.deconvoluted_distributions)

            # Calculate the number of rows and columns for the grid
            cols = 2  # Set the number of columns
            rows = (num_distributions + cols - 1) // cols  # Calculate rows based on the number of distributions

            # Create subplots dynamically
            fig, ax = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust figure size by rows
            axs = ax.flatten()  # Flatten axes for easy indexing

            # Loop through each distribution
            for i, distribution in enumerate(self.deconvoluted_distributions):
                axs[i].plot(self.log_m, self.mmd, label="Experimental MMD")  # Plot the original data
                axs[i].plot(self.log_m, np.sum(distribution, axis=0), '--', label="Cumulative MMD")
                # Plot each component of the distribution
                for j, dist in enumerate(distribution):
                    axs[i].plot(self.log_m, dist, label=f"Site {j + 1}")

                axs[i].set_title(f"{i + 2} Active Sites")

                # Add labels and legends
                axs[i].set_xlabel(r"$\log\left(M_w\right)$ / g mol$^{-1}$")
                axs[i].set_ylabel(r"$\frac{\mathrm{d}wt}{\mathrm{d}\log(M_w)}$ / a.u.")
                axs[i].legend()

            # Hide unused subplots if any
            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])  # Remove unused axes

            plt.tight_layout()  # Adjust layout for better spacing
            plt.show()

        else:
            raise ValueError("Model has not been fitted yet. Please call the `fit` method first.")


    def __repr__(self):
        return (
            f"mode={self.mode}, "
            f"active_sites={self.active_sites})"
        )
