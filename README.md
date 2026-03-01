# GlobalFactorSelector Documentation

## Overview

`GlobalFactorSelector` implements the **iterative Double-Selection LASSO procedure**
proposed in:

> Feng, Giglio, & Xiu (2020), *Taming the Factor Zoo*

The algorithm is designed to identify statistically significant factors from a large
universe of candidate asset pricing signals while mitigating:

- Overfitting
- Multicollinearity
- The "Winner’s Curse"
- Multiple testing bias

This implementation adds two important improvements:

1. **Feature Scaling** (essential for LASSO stability)
2. **Bonferroni Multiple Testing Correction**

---

# Theoretical Background

## The Factor Selection Problem

We assume:

- `r_bar` → vector of average returns (risk premia)
- `H` → matrix of control factors (already accepted factors)
- `G` → matrix of candidate factors
- `candidate_categories` → grouping labels for candidates

Goal:

Select additional factors from `G` that provide statistically significant
incremental explanatory power for `r_bar`, controlling for `H`.

---

## Double-Selection LASSO Logic

For each candidate factor \( g_j \):

### Step 1 — First-Stage LASSO
Regress:
