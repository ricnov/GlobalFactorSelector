import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy import stats

class GlobalFactorSelector:
    """
    GlobalFactorSelector: An implementation of the iterative Double-Selection LASSO method
    based on Feng, Giglio, & Xiu (2020) "Taming the Factor Zoo".

    Now includes improvements for:
    1. Feature Scaling (essential for LASSO).
    2. Multiple Testing Correction (to mitigate Winner's Curse).
    """
    def __init__(self, r_bar, H, G, candidate_categories,
                 t_threshold=1.64, cv=5, random_state=0, tol=1e-6,
                 scale_inputs=False, adjustment_method='none', significance_level=0.10):
        """
        New Parameters:
        ---------------
        scale_inputs : bool, default False
            If True, standardizes H and G (zero mean, unit variance) before running.
            Recommended for LASSO stability.
        adjustment_method : str, default 'none'
            'none': Uses fixed t_threshold.
            'bonferroni': Dynamically adjusts threshold based on number of candidates tested
                          in each iteration (reduces false positives).
        significance_level : float, default 0.10
            Used only if adjustment_method is not 'none'. The base alpha for correction.
            (e.g., 0.10 corresponds to initial t approx 1.64).
        """
        self.r_bar = r_bar
        # Copy to avoid modifying original data arrays
        self.H_raw = H.copy()
        self.G_raw = G.copy()
        self.candidate_categories = candidate_categories
        self.t_threshold = t_threshold
        self.cv = cv
        self.random_state = random_state
        self.tol = tol

        # Improvement 1: Store Scaling and Correction preferences
        self.scale_inputs = scale_inputs
        self.adjustment_method = adjustment_method
        self.significance_level = significance_level
        self.scaler_H = StandardScaler()
        self.scaler_G = StandardScaler()

    def _prepare_data(self):
        """
        Internal method to handle data scaling if enabled.
        """
        if self.scale_inputs:
            # Fit/Transform H and G.
            # Note: We do NOT scale r_bar as it represents the target risk premia.
            H_scaled = self.scaler_H.fit_transform(self.H_raw)
            G_scaled = self.scaler_G.fit_transform(self.G_raw)
            return H_scaled, G_scaled
        else:
            return self.H_raw, self.G_raw

    def _get_dynamic_threshold(self, n_candidates):
        """
        Internal method to calculate t-threshold based on adjustment method.
        """
        if self.adjustment_method == 'bonferroni':
            if n_candidates == 0:
                return self.t_threshold
            # Bonferroni correction: alpha / m
            # We use two-tailed critical value
            adj_alpha = self.significance_level / n_candidates
            # ppf(1 - alpha/2) gives the critical t-value
            critical_val = stats.norm.ppf(1 - adj_alpha / 2)
            return critical_val
        else:
            # Default: Return the fixed threshold provided by user
            return self.t_threshold

    def first_stage_lasso(self, H_current):
        lasso_cv = LassoCV(cv=self.cv, random_state=self.random_state).fit(H_current, self.r_bar)
        coef = lasso_cv.coef_
        indices = [i for i, c in enumerate(coef) if np.abs(c) > self.tol]
        return indices, lasso_cv

    def second_stage_lasso(self, g_candidate, H_current):
        lasso_cv = LassoCV(cv=self.cv, random_state=self.random_state).fit(H_current, g_candidate)
        coef = lasso_cv.coef_
        indices = [i for i, c in enumerate(coef) if np.abs(c) > self.tol]
        return indices, lasso_cv

    def ols_regression_t_value(self, H_selected, g_candidate):
        X = np.column_stack((H_selected, g_candidate))
        X = sm.add_constant(X)
        model = sm.OLS(self.r_bar, X).fit()
        # The candidate factor is the last column (index -1)
        t_value = np.abs(model.tvalues[-1])
        return t_value, model

    def post_selection_ols(self, H_final):
        X = sm.add_constant(H_final)
        model = sm.OLS(self.r_bar, X).fit()
        return model

    def run(self):
        # 1. Prepare Data (Scale if requested)
        H_working, G_working = self._prepare_data()

        n, _ = H_working.shape
        k = G_working.shape[1]
        candidate_indices = list(range(k))
        selected_indices = []
        selected_categories = set()

        H_current = H_working.copy()
        iteration = 0

        while candidate_indices:
            best_t = 0
            best_idx = None

            # 2. Determine Threshold for this iteration
            # If Bonferroni is on, this threshold will be high initially and decrease as candidates are removed.
            current_threshold = self._get_dynamic_threshold(len(candidate_indices))

            # Loop through current candidates
            for idx in candidate_indices:
                g_candidate = G_working[:, idx]

                # Double Selection LASSO Logic
                I1, _ = self.first_stage_lasso(H_current)
                I2, _ = self.second_stage_lasso(g_candidate, H_current)
                I3 = list(set(I1).union(set(I2)))

                if len(I3) == 0:
                    H_selected = np.empty((n, 0))
                else:
                    H_selected = H_current[:, I3]

                t_val, _ = self.ols_regression_t_value(H_selected, g_candidate)

                if t_val > best_t:
                    best_t = t_val
                    best_idx = idx

            # 3. Evaluation against (potentially dynamic) threshold
            if best_t > current_threshold:
                cat = self.candidate_categories[best_idx]

                print(f"Iter {iteration}: Selected Index {best_idx} (Cat {cat}) "
                      f"| t-stat: {best_t:.2f} > Threshold: {current_threshold:.2f}")

                # Add selected factor to Control set H
                H_current = np.column_stack((H_current, G_working[:, best_idx]))
                selected_indices.append(best_idx)
                selected_categories.add(cat)

                # Remove all candidates from the same category
                old_len = len(candidate_indices)
                candidate_indices = [idx for idx in candidate_indices
                                     if self.candidate_categories[idx] != cat]
                removed_count = old_len - len(candidate_indices)
                print(f"   -> Removed {removed_count} candidates from Category {cat}")

                iteration += 1
            else:
                print(f"Stop: Best t-stat {best_t:.2f} did not exceed threshold {current_threshold:.2f}")
                break

        final_model = self.post_selection_ols(H_current)
        return H_current, selected_indices, list(selected_categories), final_model

# =============================================================================
# Updated Scenario Execution
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n, m, k = 100, 1, 152

    r_bar = np.random.randn(n)
    H = np.random.randn(n, m)
    # Intentionally add scale difference to demonstrate scaling need
    G = np.random.randn(n, k) * 10
    candidate_categories = np.random.choice(np.arange(1, 14), size=k)

    print("=== Running with Improvements ===")
    # Enable scaling and Bonferroni correction
    selector = GlobalFactorSelector(
        r_bar, H, G, candidate_categories,
        cv=5,
        random_state=42,
        scale_inputs=True,           # IMPROVEMENT 1: Enable Scaling
        adjustment_method='bonferroni', # IMPROVEMENT 2: Enable Correction
        significance_level=0.05      # Target 5% significance
    )

    H_final, selected_indices, selected_categories, final_model = selector.run()

    print("\nFinal Results:")
    print("Selected Indices:", selected_indices)
    print("Selected Categories:", selected_categories)
    print(final_model.summary())
