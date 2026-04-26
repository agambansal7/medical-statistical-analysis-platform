"""Regression analysis module."""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit, poisson
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    model_type: str
    formula: str
    n_observations: int
    n_predictors: int

    # Model fit
    r_squared: Optional[float] = None
    adj_r_squared: Optional[float] = None
    pseudo_r_squared: Optional[float] = None
    auc: Optional[float] = None  # Area under ROC curve for logistic
    aic: Optional[float] = None
    bic: Optional[float] = None
    log_likelihood: Optional[float] = None

    # Overall test
    f_statistic: Optional[float] = None
    f_pvalue: Optional[float] = None
    chi2_statistic: Optional[float] = None
    chi2_pvalue: Optional[float] = None

    # Coefficients
    coefficients: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Diagnostics
    residual_std: Optional[float] = None
    durbin_watson: Optional[float] = None

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class RegressionAnalysis:
    """Regression analysis methods."""

    def __init__(self, significance_level: float = 0.05,
                 confidence_level: float = 0.95):
        self.alpha = significance_level
        self.confidence_level = confidence_level

    def linear_regression(self, data: pd.DataFrame,
                          outcome: str,
                          predictors: List[str],
                          include_intercept: bool = True) -> RegressionResult:
        """Simple or multiple linear regression.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names
            include_intercept: Include intercept in model

        Returns:
            RegressionResult object
        """
        # Prepare data
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()

        if len(clean_data) < len(predictors) + 2:
            return self._insufficient_data("Linear Regression")

        # Build formula
        formula = f"{outcome} ~ {' + '.join(predictors)}"
        if not include_intercept:
            formula += " - 1"

        try:
            model = ols(formula, data=clean_data).fit()

            # Extract coefficients
            coefficients = {}
            for var in model.params.index:
                coefficients[var] = {
                    'coefficient': float(model.params[var]),
                    'std_error': float(model.bse[var]),
                    't_value': float(model.tvalues[var]),
                    'p_value': float(model.pvalues[var]),
                    'ci_lower': float(model.conf_int().loc[var, 0]),
                    'ci_upper': float(model.conf_int().loc[var, 1]),
                    'significant': model.pvalues[var] < self.alpha
                }

            # Durbin-Watson test
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(model.resid)

            # Generate summary
            sig_predictors = [p for p in predictors
                             if coefficients.get(p, {}).get('significant', False)]

            summary = self._generate_linear_summary(model, outcome, sig_predictors)

            return RegressionResult(
                model_type="Linear Regression (OLS)",
                formula=formula,
                n_observations=int(model.nobs),
                n_predictors=len(predictors),
                r_squared=float(model.rsquared),
                adj_r_squared=float(model.rsquared_adj),
                aic=float(model.aic),
                bic=float(model.bic),
                f_statistic=float(model.fvalue),
                f_pvalue=float(model.f_pvalue),
                coefficients=coefficients,
                residual_std=float(np.sqrt(model.mse_resid)),
                durbin_watson=float(dw),
                summary_text=summary
            )

        except Exception as e:
            result = self._insufficient_data("Linear Regression")
            result.warnings = [str(e)]
            return result

    def logistic_regression(self, data: pd.DataFrame,
                           outcome: str,
                           predictors: List[str]) -> RegressionResult:
        """Binary logistic regression.

        Args:
            data: DataFrame with the data
            outcome: Binary dependent variable name (0/1)
            predictors: List of predictor variable names

        Returns:
            RegressionResult object
        """
        # Prepare data
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna().copy()

        if len(clean_data) < len(predictors) + 10:
            return self._insufficient_data("Logistic Regression")

        # Check binary outcome
        unique_vals = clean_data[outcome].unique()
        if len(unique_vals) != 2:
            result = self._insufficient_data("Logistic Regression")
            result.warnings = [f"Outcome must be binary. Found {len(unique_vals)} unique values."]
            return result

        # Handle categorical variables - set largest category as reference
        for pred in predictors:
            if clean_data[pred].dtype == 'object' or clean_data[pred].dtype.name == 'category':
                # Get value counts and find most common category
                value_counts = clean_data[pred].value_counts()
                if len(value_counts) > 1:
                    most_common = value_counts.index[0]
                    # Convert to categorical with most common as first (reference)
                    categories = [most_common] + [c for c in value_counts.index if c != most_common]
                    clean_data[pred] = pd.Categorical(clean_data[pred], categories=categories)

        formula = f"{outcome} ~ {' + '.join(predictors)}"

        try:
            # Use method='bfgs' for better convergence, maxiter for more iterations
            model = logit(formula, data=clean_data).fit(disp=0, method='bfgs', maxiter=100)

            # Extract coefficients with odds ratios
            coefficients = {}
            model_warnings = []

            for var in model.params.index:
                coef = model.params[var]
                std_err = model.bse[var]
                odds_ratio = np.exp(coef)
                ci = model.conf_int().loc[var]
                or_ci_lower = np.exp(ci[0])
                or_ci_upper = np.exp(ci[1])

                # Check for convergence issues (extreme coefficients)
                if np.abs(coef) > 10 or std_err > 100:
                    model_warnings.append(f"Possible separation issue for {var}")

                coefficients[var] = {
                    'coefficient': float(coef) if not np.isinf(coef) else np.sign(coef) * 999,
                    'odds_ratio': float(odds_ratio) if not np.isinf(odds_ratio) and odds_ratio < 1e10 else 999.0,
                    'std_error': float(std_err) if not np.isinf(std_err) else 999.0,
                    'z_value': float(model.tvalues[var]) if not np.isnan(model.tvalues[var]) else 0.0,
                    'p_value': float(model.pvalues[var]) if not np.isnan(model.pvalues[var]) else 1.0,
                    'ci_lower': float(ci[0]) if not np.isinf(ci[0]) else -999,
                    'ci_upper': float(ci[1]) if not np.isinf(ci[1]) else 999,
                    'or_ci_lower': float(or_ci_lower) if not np.isinf(or_ci_lower) and or_ci_lower < 1e10 else 0.001,
                    'or_ci_upper': float(or_ci_upper) if not np.isinf(or_ci_upper) and or_ci_upper < 1e10 else 999.0,
                    'significant': model.pvalues[var] < self.alpha if not np.isnan(model.pvalues[var]) else False
                }

            # Pseudo R-squared (McFadden)
            pseudo_r2 = float(model.prsquared)

            # Calculate AUC
            try:
                from sklearn.metrics import roc_auc_score
                y_true = clean_data[outcome].values
                y_pred = model.predict(clean_data)
                auc = roc_auc_score(y_true, y_pred)
            except Exception:
                auc = None

            summary = self._generate_logistic_summary(model, outcome, predictors, coefficients)

            result = RegressionResult(
                model_type="Binary Logistic Regression",
                formula=formula,
                n_observations=int(model.nobs),
                n_predictors=len(predictors),
                pseudo_r_squared=pseudo_r2,
                aic=float(model.aic),
                bic=float(model.bic),
                log_likelihood=float(model.llf),
                chi2_statistic=float(model.llr),
                chi2_pvalue=float(model.llr_pvalue),
                coefficients=coefficients,
                summary_text=summary,
                warnings=model_warnings
            )

            # Add AUC to result dict (will be available via to_dict())
            if auc is not None:
                result.auc = auc

            return result

        except Exception as e:
            result = self._insufficient_data("Logistic Regression")
            result.warnings = [str(e)]
            return result

    def poisson_regression(self, data: pd.DataFrame,
                          outcome: str,
                          predictors: List[str]) -> RegressionResult:
        """Poisson regression for count data.

        Args:
            data: DataFrame with the data
            outcome: Count dependent variable name
            predictors: List of predictor variable names

        Returns:
            RegressionResult object
        """
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()

        if len(clean_data) < len(predictors) + 10:
            return self._insufficient_data("Poisson Regression")

        formula = f"{outcome} ~ {' + '.join(predictors)}"

        try:
            model = poisson(formula, data=clean_data).fit(disp=0)

            # Extract coefficients with rate ratios
            coefficients = {}
            for var in model.params.index:
                coef = model.params[var]
                rate_ratio = np.exp(coef)
                ci = model.conf_int().loc[var]

                coefficients[var] = {
                    'coefficient': float(coef),
                    'rate_ratio': float(rate_ratio),
                    'std_error': float(model.bse[var]),
                    'z_value': float(model.tvalues[var]),
                    'p_value': float(model.pvalues[var]),
                    'ci_lower': float(ci[0]),
                    'ci_upper': float(ci[1]),
                    'rr_ci_lower': float(np.exp(ci[0])),
                    'rr_ci_upper': float(np.exp(ci[1])),
                    'significant': model.pvalues[var] < self.alpha
                }

            return RegressionResult(
                model_type="Poisson Regression",
                formula=formula,
                n_observations=int(model.nobs),
                n_predictors=len(predictors),
                pseudo_r_squared=float(model.prsquared),
                aic=float(model.aic),
                bic=float(model.bic),
                log_likelihood=float(model.llf),
                chi2_statistic=float(model.llr),
                chi2_pvalue=float(model.llr_pvalue),
                coefficients=coefficients
            )

        except Exception as e:
            result = self._insufficient_data("Poisson Regression")
            result.warnings = [str(e)]
            return result

    def ordinal_regression(self, data: pd.DataFrame,
                          outcome: str,
                          predictors: List[str]) -> RegressionResult:
        """Ordinal logistic regression (proportional odds model).

        Args:
            data: DataFrame with the data
            outcome: Ordinal dependent variable name
            predictors: List of predictor variable names

        Returns:
            RegressionResult object
        """
        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel

            all_vars = [outcome] + predictors
            clean_data = data[all_vars].dropna()

            if len(clean_data) < len(predictors) + 10:
                return self._insufficient_data("Ordinal Regression")

            # Ensure outcome is categorical and ordered
            y = clean_data[outcome]
            X = clean_data[predictors]
            X = sm.add_constant(X)

            model = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=0)

            coefficients = {}
            for i, var in enumerate(X.columns):
                if i < len(model.params) - len(y.unique()) + 1:
                    coefficients[var] = {
                        'coefficient': float(model.params[i]),
                        'odds_ratio': float(np.exp(model.params[i])),
                        'std_error': float(model.bse[i]),
                        'z_value': float(model.tvalues[i]),
                        'p_value': float(model.pvalues[i]),
                        'significant': model.pvalues[i] < self.alpha
                    }

            return RegressionResult(
                model_type="Ordinal Logistic Regression",
                formula=f"{outcome} ~ {' + '.join(predictors)}",
                n_observations=len(clean_data),
                n_predictors=len(predictors),
                pseudo_r_squared=float(model.prsquared) if hasattr(model, 'prsquared') else None,
                aic=float(model.aic),
                bic=float(model.bic),
                log_likelihood=float(model.llf),
                coefficients=coefficients
            )

        except Exception as e:
            result = self._insufficient_data("Ordinal Regression")
            result.warnings = [str(e)]
            return result

    def hierarchical_regression(self, data: pd.DataFrame,
                               outcome: str,
                               blocks: List[List[str]]) -> Dict[str, Any]:
        """Hierarchical (block) regression.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            blocks: List of predictor blocks (each block is a list of variables)

        Returns:
            Dictionary with results for each block
        """
        results = {}
        cumulative_predictors = []

        for i, block in enumerate(blocks):
            block_name = f"Block {i+1}"
            cumulative_predictors.extend(block)

            result = self.linear_regression(data, outcome, cumulative_predictors.copy())
            results[block_name] = {
                'predictors_added': block,
                'cumulative_predictors': cumulative_predictors.copy(),
                'result': result
            }

            # Calculate R² change
            if i > 0:
                prev_r2 = results[f"Block {i}"]['result'].r_squared or 0
                curr_r2 = result.r_squared or 0
                results[block_name]['r_squared_change'] = curr_r2 - prev_r2

        # Summary
        results['summary'] = {
            'total_blocks': len(blocks),
            'final_r_squared': results[f"Block {len(blocks)}"]['result'].r_squared,
            'final_adj_r_squared': results[f"Block {len(blocks)}"]['result'].adj_r_squared
        }

        return results

    def mixed_effects_model(self, data: pd.DataFrame,
                           outcome: str,
                           fixed_effects: List[str],
                           random_effects: str,
                           random_slopes: Optional[List[str]] = None) -> RegressionResult:
        """Linear mixed effects model.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            fixed_effects: List of fixed effect predictors
            random_effects: Grouping variable for random intercepts
            random_slopes: Optional variables for random slopes

        Returns:
            RegressionResult object
        """
        try:
            import statsmodels.formula.api as smf

            # Build formula
            fixed_formula = f"{outcome} ~ {' + '.join(fixed_effects)}"

            all_vars = [outcome, random_effects] + fixed_effects
            if random_slopes:
                all_vars.extend(random_slopes)
            clean_data = data[all_vars].dropna()

            if len(clean_data) < len(fixed_effects) + 20:
                return self._insufficient_data("Mixed Effects Model")

            # Fit model
            if random_slopes:
                re_formula = f"~{' + '.join(random_slopes)}"
                model = smf.mixedlm(fixed_formula, clean_data,
                                   groups=clean_data[random_effects],
                                   re_formula=re_formula).fit()
            else:
                model = smf.mixedlm(fixed_formula, clean_data,
                                   groups=clean_data[random_effects]).fit()

            # Extract coefficients
            coefficients = {}
            for var in model.fe_params.index:
                coefficients[var] = {
                    'coefficient': float(model.fe_params[var]),
                    'std_error': float(model.bse_fe[var]),
                    'z_value': float(model.tvalues[var]),
                    'p_value': float(model.pvalues[var]),
                    'significant': model.pvalues[var] < self.alpha
                }

            return RegressionResult(
                model_type="Linear Mixed Effects Model",
                formula=fixed_formula,
                n_observations=int(model.nobs),
                n_predictors=len(fixed_effects),
                aic=float(model.aic),
                bic=float(model.bic),
                log_likelihood=float(model.llf),
                coefficients=coefficients,
                summary_text=f"Random intercepts for: {random_effects}"
            )

        except Exception as e:
            result = self._insufficient_data("Mixed Effects Model")
            result.warnings = [str(e)]
            return result

    def regression_diagnostics(self, data: pd.DataFrame,
                               outcome: str,
                               predictors: List[str]) -> Dict[str, Any]:
        """Comprehensive regression diagnostics.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names

        Returns:
            Dictionary of diagnostic results
        """
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()

        formula = f"{outcome} ~ {' + '.join(predictors)}"
        model = ols(formula, data=clean_data).fit()

        diagnostics = {}

        # 1. Normality of residuals
        residuals = model.resid
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])
        diagnostics['normality'] = {
            'test': 'Shapiro-Wilk',
            'statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'satisfied': shapiro_p > self.alpha,
            'interpretation': "Residuals are normally distributed" if shapiro_p > self.alpha
                            else "Residuals deviate from normal distribution"
        }

        # 2. Homoscedasticity (Breusch-Pagan test)
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
        diagnostics['homoscedasticity'] = {
            'test': 'Breusch-Pagan',
            'statistic': float(bp_stat),
            'p_value': float(bp_p),
            'satisfied': bp_p > self.alpha,
            'interpretation': "Homoscedasticity assumption satisfied" if bp_p > self.alpha
                            else "Evidence of heteroscedasticity"
        }

        # 3. Independence (Durbin-Watson)
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(model.resid)
        diagnostics['independence'] = {
            'test': 'Durbin-Watson',
            'statistic': float(dw_stat),
            'satisfied': 1.5 <= dw_stat <= 2.5,
            'interpretation': "No significant autocorrelation" if 1.5 <= dw_stat <= 2.5
                            else "Possible autocorrelation in residuals"
        }

        # 4. Multicollinearity (VIF)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = clean_data[predictors]
        X = sm.add_constant(X)
        vif_data = {}
        for i, col in enumerate(X.columns):
            if col != 'const':
                vif_data[col] = float(variance_inflation_factor(X.values, i))

        diagnostics['multicollinearity'] = {
            'test': 'VIF',
            'vif_values': vif_data,
            'max_vif': max(vif_data.values()) if vif_data else 0,
            'satisfied': all(v < 5 for v in vif_data.values()),
            'interpretation': "No multicollinearity concern" if all(v < 5 for v in vif_data.values())
                            else f"High VIF detected for: {[k for k, v in vif_data.items() if v >= 5]}"
        }

        # 5. Influential observations (Cook's distance)
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        n = len(clean_data)
        threshold = 4 / n
        influential = np.where(cooks_d > threshold)[0]

        diagnostics['influential_observations'] = {
            'test': "Cook's distance",
            'threshold': float(threshold),
            'n_influential': len(influential),
            'influential_indices': influential.tolist()[:10],  # First 10
            'max_cooks_d': float(max(cooks_d)),
            'interpretation': f"{len(influential)} potentially influential observations detected"
        }

        return diagnostics

    def _generate_linear_summary(self, model, outcome: str,
                                sig_predictors: List[str]) -> str:
        """Generate summary text for linear regression."""
        r2 = model.rsquared * 100
        summary = f"The model explains {r2:.1f}% of variance in {outcome}"

        if model.f_pvalue < self.alpha:
            summary += f" (F = {model.fvalue:.2f}, p < 0.001). "
        else:
            summary += f" (F = {model.fvalue:.2f}, p = {model.f_pvalue:.3f}). "

        if sig_predictors:
            summary += f"Significant predictors: {', '.join(sig_predictors)}."
        else:
            summary += "No predictors reached statistical significance."

        return summary

    def _generate_logistic_summary(self, model, outcome: str,
                                  predictors: List[str],
                                  coefficients: Dict) -> str:
        """Generate summary text for logistic regression."""
        sig_preds = [p for p in predictors if coefficients.get(p, {}).get('significant', False)]

        summary = f"Logistic regression predicting {outcome}. "

        if model.llr_pvalue < self.alpha:
            summary += f"Model is significant (χ² = {model.llr:.2f}, p < 0.001). "
        else:
            summary += f"Model is not significant (χ² = {model.llr:.2f}, p = {model.llr_pvalue:.3f}). "

        if sig_preds:
            summary += "Significant predictors: "
            for p in sig_preds:
                or_val = coefficients[p]['odds_ratio']
                direction = "increases" if or_val > 1 else "decreases"
                summary += f"{p} (OR = {or_val:.2f}, {direction} odds); "

        return summary.rstrip("; ") + "."

    def _insufficient_data(self, model_type: str) -> RegressionResult:
        """Create result for insufficient data."""
        return RegressionResult(
            model_type=model_type,
            formula="N/A",
            n_observations=0,
            n_predictors=0,
            summary_text="Insufficient data to fit model"
        )
