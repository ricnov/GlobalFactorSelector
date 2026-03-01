# Global Factor Selector: Methodology & Documentation

## Overview

This module implements the **Iterative Double-Selection LASSO** method for asset pricing, based on the framework established by **Feng, Giglio, & Xiu (2020)** in their paper *"Taming the Factor Zoo"*.

The primary goal of this tool is to identify specific risk factors that provide independent pricing information from a high-dimensional set of candidate factors, effectively separating true pricing signals from redundant factors or noise.

### Key Capabilities
1.  **Double-Selection LASSO:** Mitigates omitted variable bias when evaluating candidate factors.
2.  **Iterative Selection:** Sequentially selects the most significant factors until no remaining candidates exceed the significance threshold.
3.  **Feature Scaling:** Optional standardization to ensure LASSO regularization performs correctly across factors with different units/volatilities.
4.  **Multiple Testing Correction:** Optional dynamic Bonferroni adjustment to control the Family-Wise Error Rate (FWER) and mitigate the "Winner's Curse."

---

## Mathematical Framework

The algorithm operates on the linear factor model structure:

$$r_t = \alpha + \beta_H H_t + \beta_g g_t + \epsilon_t$$

Where:
* **$r_t$**: The target asset returns (or risk premia).
* **$H_t$**: The set of "Control" factors (already accepted or benchmark factors).
* **$g_t$**: The "Candidate" factor currently being tested.

### The Double-Selection Logic
Standard OLS is unreliable when the number of potential controls in $H$ is large relative to the sample size. We employ a three-step procedure for every candidate $g$:

1.  **First Stage (LASSO on Returns):**
    Select variables from $H$ that are useful for predicting returns ($r$).
    $$r = H \theta + u$$
2.  **Second Stage (LASSO on Candidate):**
    Select variables from $H$ that are correlated with the candidate factor ($g$). This is crucial to remove omitted variable bias.
    $$g = H \delta + v$$
3.  **Post-Selection OLS:**
    Run a standard OLS regression of $r$ on $g$ and the *union* of predictors selected in steps 1 and 2. The t-statistic of $g$ in this regression is the test statistic.

---

## Class: `GlobalFactorSelector`

### Initialization Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `r_bar` | Array | Required | Vector of average excess returns (risk premia). |
| `H` | Matrix | Required | Matrix of control factors (benchmark models). |
| `G` | Matrix | Required | Matrix of candidate factors (the "Zoo"). |
| `candidate_categories` | Array | Required | Labels/IDs for candidate categories. Used to exclude correlated cluster members after a selection. |
| `scale_inputs` | `bool` | `False` | **(Improvement)** If `True`, standardizes $H$ and $G$ to zero mean and unit variance. Highly recommended for LASSO. |
| `adjustment_method` | `str` | `'none'` | **(Improvement)** Controls the t-stat threshold logic.<br>• `'none'`: Fixed threshold.<br>• `'bonferroni'`: Dynamic adjustment. |
| `t_threshold` | `float` | `1.64` | The base critical value (approx. 90% confidence) if no adjustment is used. |
| `significance_level` | `float` | `0.10` | The target alpha (e.g., 0.05 or 0.10) used when calculating dynamic Bonferroni corrections. |

---

## Algorithm Logic Flow

The `run()` method executes the following iterative loop:

1.  **Preprocessing:**
    * If `scale_inputs` is active, $H$ and $G$ are standardized using `StandardScaler`.
    * $r\_bar$ is left untouched (target variable).

2.  **Iterative Loop:**
    * While candidates remain in $G$:
        1.  **Determine Threshold:**
            * If `adjustment_method='bonferroni'`, calculate critical $t$ based on $\alpha / m$ (where $m$ is remaining candidates).
            * Otherwise, use fixed `t_threshold`.
        2.  **Scan Candidates:**
            * For every remaining factor $g$ in $G$:
                * Perform **Double-Selection LASSO**.
                * Calculate the t-statistic for $g$ via OLS on the selected union of controls.
        3.  **Selection Decision:**
            * Identify the candidate with the **maximum t-statistic**.
            * Compare max t-statistic to the current threshold.
        4.  **Update or Stop:**
            * **If t > Threshold:**
                * Add $g$ to the control set $H$.
                * Record the selection.
                * **Exclusion Step:** Remove $g$ *and* all other candidates belonging to the same category/cluster from $G$ to prevent multicollinearity or redundant variations of the same signal.
                * Repeat loop.
            * **If t < Threshold:**
                * Stop iteration. No further factors add explanatory power.

---

## Improvements Explained

### 1. Feature Scaling (`scale_inputs`)
LASSO regression penalizes the sum of absolute coefficients ($\lambda \sum |\beta|$). If factors have vastly different scales (e.g., one factor is 0.01 and another is 100), LASSO will unfairly penalize the larger-scaled variable even if it is predictive.
* **Recommendation:** Always set to `True` unless data is pre-normalized.

### 2. Bonferroni Correction (`adjustment_method`)
In a "Factor Zoo" with hundreds of potential factors, testing them one by one increases the probability of finding a significant result purely by chance (False Positives).
* **The Fix:** The threshold becomes stricter as the number of candidates ($m$) increases.
* **Formula:** Critical value corresponds to $p < \frac{\alpha}{m}$.
* **Dynamics:** As factors are selected and categories removed, $m$ decreases, slightly relaxing the threshold for subsequent iterations.

---

## Outputs

The `run()` method returns:

1.  **`H_final`**: The final matrix of control factors (Original $H$ + Newly Selected Factors).
2.  **`selected_indices`**: The column indices of the factors selected from the original $G$ matrix.
3.  **`selected_categories`**: A list of the unique categories associated with the selected factors.
4.  **`final_model`**: A `statsmodels` OLS regression result object using the final set of factors, allowing for immediate inspection of $R^2$ and coefficients.
