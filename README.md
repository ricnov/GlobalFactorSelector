Overview

GlobalFactorSelector implements the iterative Double-Selection LASSO procedure proposed in:

Feng, Giglio, & Xiu (2020), Taming the Factor Zoo

The algorithm is designed to identify statistically significant factors from a large universe of candidate asset pricing signals while mitigating:

Overfitting

Multicollinearity

The "Winner’s Curse"

Multiple testing bias

This implementation adds two important improvements:

Feature Scaling (essential for LASSO stability)

Bonferroni Multiple Testing Correction

Theoretical Background
The Factor Selection Problem

We assume:

r_bar → vector of average returns (risk premia)

H → matrix of control factors (already accepted factors)

G → matrix of candidate factors

candidate_categories → grouping labels for candidates

Goal:

Select additional factors from G that provide statistically significant incremental explanatory power for r_bar, controlling for H.

Double-Selection LASSO Logic

For each candidate factor g_j:

Step 1 — First-Stage LASSO
Regress:

r_bar ~ H

Select control variables important for explaining returns.

Step 2 — Second-Stage LASSO
Regress:

g_j ~ H

Select control variables important for explaining the candidate.

Step 3 — Union of Controls

I3 = I1 ∪ I2

Step 4 — Post-LASSO OLS

r_bar ~ H_selected + g_j

Compute the t-statistic of g_j.

Selection Rule

At each iteration:

Evaluate all remaining candidates

Select the one with the highest absolute t-stat

Compare against threshold

Threshold can be:

Fixed threshold (default: 1.64)

Bonferroni-adjusted threshold

Stop when no candidate exceeds the threshold.

Class Documentation
Constructor

GlobalFactorSelector(
r_bar,
H,
G,
candidate_categories,
t_threshold=1.64,
cv=5,
random_state=0,
tol=1e-6,
scale_inputs=False,
adjustment_method='none',
significance_level=0.10
)

Parameters
Core Inputs

r_bar
Vector (n,) of risk premia.

H
Matrix (n × m) of control factors.

G
Matrix (n × k) of candidate factors.

candidate_categories
Category label per candidate.

Regularization Controls

cv
Cross-validation folds for LassoCV.

random_state
Reproducibility seed.

tol
Threshold for non-zero LASSO coefficients.

Improvement 1 — Feature Scaling

scale_inputs
If True, standardizes H and G.

Why scaling matters:

LASSO penalizes coefficients based on magnitude. If variables have different scales, selection becomes biased.

Scaling ensures:

Numerical stability

Fair penalty comparison

Improved convergence

Improvement 2 — Multiple Testing Adjustment

adjustment_method
'none' or 'bonferroni'

significance_level
Base alpha level.

Bonferroni adjustment:

alpha_adj = alpha / m

Threshold computed as:

z_(1 − alpha_adj / 2)

This reduces false positives when testing many candidates.

Methods

_prepare_data()

Handles optional feature scaling.

Returns:

H_working, G_working

_get_dynamic_threshold(n_candidates)

Returns:

Fixed threshold

OR Bonferroni-adjusted threshold

first_stage_lasso(H_current)

Performs:

LassoCV(H_current → r_bar)

Returns:

Selected indices

Fitted model

second_stage_lasso(g_candidate, H_current)

Performs:

LassoCV(H_current → g_candidate)

Returns:

Selected indices

Fitted model

ols_regression_t_value(H_selected, g_candidate)

Runs:

OLS(r_bar ~ H_selected + g_candidate)

Returns:

Absolute t-statistic

Fitted model

post_selection_ols(H_final)

Final regression after selection:

OLS(r_bar ~ H_final)

Returns:

Final statsmodels regression object

run()

Main algorithm loop:

Prepare data

Iterate over remaining candidates

Apply double-selection

Select best candidate

Remove same-category candidates

Repeat until stopping condition

Returns:

H_final
selected_indices
selected_categories
final_model

Iterative Category Removal

After selecting a factor:

All candidates from the same category are removed.

This enforces:

Economic interpretability

Reduces redundancy

Prevents overfitting

Example Usage

if name == "main":

Simulation setup:

n = 100 observations

m = 1 control factor

k = 152 candidate factors

13 categories

Scaling intentionally required because:

G = np.random.randn(n, k) * 10

Candidates have larger variance than controls.

Running the improved version:

selector = GlobalFactorSelector(
r_bar,
H,
G,
candidate_categories,
cv=5,
random_state=42,
scale_inputs=True,
adjustment_method='bonferroni',
significance_level=0.05
)

Output during execution:

Iter 0: Selected Index ...
-> Removed X candidates from Category ...
Stop: Best t-stat ... did not exceed threshold ...

Final output includes:

Selected indices

Selected categories

OLS summary table

Computational Complexity

Let:

k = number of candidates
m = number of controls
n = observations

Each iteration:

k LASSO fits

k OLS regressions

Worst-case complexity approximately:

O(k² × LASSO_cost)

In practice, shrinks quickly due to category removal.

Practical Recommendations

Always enable scaling:

scale_inputs=True

Use Bonferroni correction when:

k is large

Many categories

Concerned about false discoveries

Typical threshold choices:

Alpha 10% → t ≈ 1.64
Alpha 5% → t ≈ 1.96
Alpha 1% → t ≈ 2.58

Interpretation of Results

After selection:

Significant factors represent incremental explanatory power

Final OLS provides unbiased post-selection estimates

Categories reveal economically distinct signals

Extensions

Possible improvements:

False Discovery Rate (FDR) correction

Stability selection

Elastic Net instead of LASSO

Parallel candidate evaluation

Bootstrap inference

Dependencies

numpy
statsmodels
sklearn
scipy

Install via:

pip install numpy statsmodels scikit-learn scipy

Summary

This implementation provides:

A faithful implementation of Feng–Giglio–Xiu (2020)

Stable LASSO via scaling

Controlled inference via Bonferroni correction

Category-aware iterative selection

Post-selection unbiased estimation

It is suitable for:

Asset pricing research

High-dimensional factor selection

Cross-sectional return modeling

Empirical finance applications
