"""Longitudinal data analysis module.

Provides methods for analyzing repeated measures data:
- Mixed effects models (linear and generalized)
- Generalized Estimating Equations (GEE)
- Growth curve models
- Time series analysis for panel data
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class MixedModelResult:
    """Result of mixed effects model analysis."""
    model_type: str
    formula: str
    n_observations: int
    n_groups: int
    n_time_points: Optional[int] = None

    # Fixed effects
    fixed_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Random effects
    random_effects_variance: Dict[str, float] = field(default_factory=dict)
    icc: Optional[float] = None  # Intraclass correlation

    # Model fit
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None

    # Convergence
    converged: bool = True

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Educational content
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class GEEResult:
    """Result of GEE analysis."""
    model_type: str
    formula: str
    n_observations: int
    n_clusters: int
    correlation_structure: str
    working_correlation: Optional[np.ndarray] = None

    # Coefficients
    coefficients: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Model fit
    qic: Optional[float] = None  # Quasi-likelihood Information Criterion

    # Scale parameter
    scale: Optional[float] = None

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Educational content
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.working_correlation is not None:
            result['working_correlation'] = self.working_correlation.tolist()
        return {k: v for k, v in result.items() if v is not None}


class LongitudinalAnalysis:
    """Longitudinal data analysis methods."""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.confidence_level = 1 - significance_level

    def linear_mixed_model(
        self,
        data: pd.DataFrame,
        outcome: str,
        fixed_effects: List[str],
        random_intercept: str,
        random_slope: Optional[str] = None,
        time_variable: Optional[str] = None
    ) -> MixedModelResult:
        """Fit a linear mixed effects model.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            fixed_effects: List of fixed effect predictor names
            random_intercept: Grouping variable for random intercept
            random_slope: Variable for random slope (optional)
            time_variable: Time variable for repeated measures

        Returns:
            MixedModelResult object
        """
        # Prepare data
        all_vars = [outcome] + fixed_effects + [random_intercept]
        if time_variable:
            all_vars.append(time_variable)
        if random_slope:
            all_vars.append(random_slope)

        all_vars = list(set(all_vars))  # Remove duplicates
        clean_data = data[all_vars].dropna()

        n_groups = clean_data[random_intercept].nunique()
        n_obs = len(clean_data)

        if n_obs < len(fixed_effects) + 5:
            return self._insufficient_data("Linear Mixed Model")

        # Build formula
        formula = f"{outcome} ~ {' + '.join(fixed_effects)}"

        try:
            # Define random effects structure
            if random_slope:
                re_formula = f"1 + {random_slope}"
            else:
                re_formula = "1"

            # Fit model
            model = MixedLM.from_formula(
                formula,
                groups=random_intercept,
                re_formula=re_formula,
                data=clean_data
            )
            result = model.fit(method='powell', maxiter=1000)

            # Extract fixed effects
            fixed_eff = {}
            for var in result.fe_params.index:
                fixed_eff[var] = {
                    'coefficient': float(result.fe_params[var]),
                    'std_error': float(result.bse_fe[var]) if var in result.bse_fe.index else None,
                    'z_value': float(result.tvalues[var]) if hasattr(result, 'tvalues') else None,
                    'p_value': float(result.pvalues[var]) if hasattr(result, 'pvalues') else None,
                    'ci_lower': float(result.conf_int().loc[var, 0]) if var in result.conf_int().index else None,
                    'ci_upper': float(result.conf_int().loc[var, 1]) if var in result.conf_int().index else None,
                    'significant': result.pvalues[var] < self.alpha if hasattr(result, 'pvalues') else None
                }

            # Extract random effects variance
            re_var = {}
            cov = result.cov_re
            if hasattr(cov, 'values'):
                re_var['intercept_variance'] = float(cov.values[0, 0]) if cov.size > 0 else None
                if random_slope and cov.size > 1:
                    re_var['slope_variance'] = float(cov.values[1, 1])
                    re_var['covariance'] = float(cov.values[0, 1])

            # Calculate ICC
            residual_var = float(result.scale)
            intercept_var = re_var.get('intercept_variance', 0)
            icc = intercept_var / (intercept_var + residual_var) if (intercept_var + residual_var) > 0 else None

            # Generate summary
            summary = self._generate_mixed_model_summary(
                result, outcome, fixed_effects, random_intercept, icc
            )

            # Educational notes
            edu_notes = {
                'what_is_mixed_model': 'Mixed effects models handle correlated observations (e.g., repeated measures on same subjects) by including random effects that capture between-subject variation.',
                'interpreting_fixed_effects': 'Fixed effects coefficients represent the average change in the outcome for a one-unit change in the predictor, adjusted for other variables.',
                'interpreting_icc': f'ICC of {icc:.3f} means {icc*100:.1f}% of variance is between subjects.' if icc else None,
                'when_to_use': 'Use mixed models when you have clustered/hierarchical data, repeated measures, or nested designs.',
                'assumptions': 'Assumes normally distributed residuals, linear relationship, and correctly specified random effects structure.'
            }

            return MixedModelResult(
                model_type="Linear Mixed Effects Model",
                formula=formula,
                n_observations=n_obs,
                n_groups=n_groups,
                n_time_points=clean_data[time_variable].nunique() if time_variable else None,
                fixed_effects=fixed_eff,
                random_effects_variance=re_var,
                icc=icc,
                log_likelihood=float(result.llf) if hasattr(result, 'llf') else None,
                aic=float(result.aic) if hasattr(result, 'aic') else None,
                bic=float(result.bic) if hasattr(result, 'bic') else None,
                converged=result.converged if hasattr(result, 'converged') else True,
                summary_text=summary,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_data("Linear Mixed Model")
            result.warnings = [str(e)]
            return result

    def generalized_mixed_model(
        self,
        data: pd.DataFrame,
        outcome: str,
        fixed_effects: List[str],
        random_intercept: str,
        family: str = "binomial"
    ) -> MixedModelResult:
        """Fit a generalized linear mixed effects model.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            fixed_effects: List of fixed effect predictor names
            random_intercept: Grouping variable for random intercept
            family: Distribution family ('binomial', 'poisson', 'gamma')

        Returns:
            MixedModelResult object
        """
        # Prepare data
        all_vars = [outcome] + fixed_effects + [random_intercept]
        clean_data = data[all_vars].dropna()

        n_groups = clean_data[random_intercept].nunique()
        n_obs = len(clean_data)

        if n_obs < len(fixed_effects) + 5:
            return self._insufficient_data("Generalized Mixed Model")

        formula = f"{outcome} ~ {' + '.join(fixed_effects)}"

        try:
            # Use statsmodels BinomialBayesMixedGLM for binary outcomes
            # For other families, we'll use approximation
            if family == "binomial":
                from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

                model = BinomialBayesMixedGLM.from_formula(
                    formula,
                    vc_formulas={"a": f"0 + C({random_intercept})"},
                    data=clean_data
                )
                result = model.fit_vb()

                # Extract coefficients
                fixed_eff = {}
                for i, var in enumerate(result.model.exog_names):
                    coef = float(result.params[i])
                    se = float(result.bse[i]) if hasattr(result, 'bse') else None

                    # Calculate odds ratio for logistic
                    or_val = np.exp(coef)
                    or_lower = np.exp(coef - 1.96 * se) if se else None
                    or_upper = np.exp(coef + 1.96 * se) if se else None

                    fixed_eff[var] = {
                        'coefficient': coef,
                        'odds_ratio': or_val,
                        'std_error': se,
                        'or_ci_lower': or_lower,
                        'or_ci_upper': or_upper
                    }

                return MixedModelResult(
                    model_type="Binomial Mixed Effects Model (GLMM)",
                    formula=formula,
                    n_observations=n_obs,
                    n_groups=n_groups,
                    fixed_effects=fixed_eff,
                    converged=True,
                    summary_text=f"Fitted binomial GLMM with {n_groups} groups.",
                    educational_notes={
                        'what_is_glmm': 'Generalized Linear Mixed Models extend mixed models to non-normal outcomes (binary, count data).',
                        'interpreting_or': 'Odds ratios >1 indicate increased odds, <1 indicate decreased odds.',
                        'when_to_use': 'Use for binary/count outcomes with repeated measures or clustered data.'
                    }
                )
            else:
                return self._insufficient_data(f"GLMM with {family} family not yet implemented")

        except Exception as e:
            result = self._insufficient_data("Generalized Mixed Model")
            result.warnings = [str(e)]
            return result

    def gee(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        subject_id: str,
        time_variable: Optional[str] = None,
        correlation: str = "exchangeable",
        family: str = "gaussian"
    ) -> GEEResult:
        """Fit a Generalized Estimating Equations model.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names
            subject_id: Column identifying subjects/clusters
            time_variable: Time variable for panel structure
            correlation: Correlation structure ('exchangeable', 'ar1', 'unstructured', 'independence')
            family: Distribution family ('gaussian', 'binomial', 'poisson')

        Returns:
            GEEResult object
        """
        # Prepare data
        all_vars = [outcome] + predictors + [subject_id]
        if time_variable:
            all_vars.append(time_variable)

        clean_data = data[all_vars].dropna().copy()

        # Sort by subject and time
        if time_variable:
            clean_data = clean_data.sort_values([subject_id, time_variable])
        else:
            clean_data = clean_data.sort_values(subject_id)

        n_clusters = clean_data[subject_id].nunique()
        n_obs = len(clean_data)

        if n_obs < len(predictors) + 5:
            return self._insufficient_gee_data()

        formula = f"{outcome} ~ {' + '.join(predictors)}"

        try:
            # Select family
            if family == "gaussian":
                fam = families.Gaussian()
            elif family == "binomial":
                fam = families.Binomial()
            elif family == "poisson":
                fam = families.Poisson()
            else:
                fam = families.Gaussian()

            # Select correlation structure
            if correlation == "exchangeable":
                cov_struct = sm.cov_struct.Exchangeable()
            elif correlation == "ar1":
                cov_struct = sm.cov_struct.Autoregressive()
            elif correlation == "unstructured":
                cov_struct = sm.cov_struct.Unstructured()
            else:
                cov_struct = sm.cov_struct.Independence()

            # Fit model
            model = GEE.from_formula(
                formula,
                groups=subject_id,
                data=clean_data,
                family=fam,
                cov_struct=cov_struct
            )
            result = model.fit()

            # Extract coefficients
            coeffs = {}
            for var in result.params.index:
                coef = float(result.params[var])
                se = float(result.bse[var])
                z = float(result.tvalues[var])
                p = float(result.pvalues[var])
                ci_lower = float(result.conf_int().loc[var, 0])
                ci_upper = float(result.conf_int().loc[var, 1])

                coeffs[var] = {
                    'coefficient': coef,
                    'std_error': se,
                    'robust_std_error': se,  # GEE uses robust SEs by default
                    'z_value': z,
                    'p_value': p,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'significant': p < self.alpha
                }

                # Add odds ratio for binomial
                if family == "binomial":
                    coeffs[var]['odds_ratio'] = np.exp(coef)
                    coeffs[var]['or_ci_lower'] = np.exp(ci_lower)
                    coeffs[var]['or_ci_upper'] = np.exp(ci_upper)

            # Get working correlation matrix
            work_corr = None
            if hasattr(result, 'cov_struct') and hasattr(result.cov_struct, 'summary'):
                work_corr_summary = result.cov_struct.summary()

            # Calculate QIC
            qic = float(result.qic()[0]) if hasattr(result, 'qic') else None

            # Generate summary
            sig_predictors = [p for p in predictors if coeffs.get(p, {}).get('significant', False)]
            summary = self._generate_gee_summary(result, outcome, sig_predictors, correlation, family)

            # Educational notes
            edu_notes = {
                'what_is_gee': 'GEE is a population-averaged approach for correlated data that makes robust inferences without fully specifying the correlation structure.',
                'vs_mixed_models': 'GEE estimates population-averaged effects; mixed models estimate subject-specific effects. Choose GEE when interested in overall population effects.',
                'correlation_choice': f'Using {correlation} correlation assumes {"equal correlation between all observations" if correlation == "exchangeable" else "correlation decreases with time lag" if correlation == "ar1" else "no assumed pattern"}.',
                'robust_se': 'GEE uses sandwich (robust) standard errors that are valid even if correlation structure is misspecified.',
                'when_to_use': 'Use GEE for marginal/population-averaged inference with clustered/longitudinal data.'
            }

            return GEEResult(
                model_type="Generalized Estimating Equations",
                formula=formula,
                n_observations=n_obs,
                n_clusters=n_clusters,
                correlation_structure=correlation,
                coefficients=coeffs,
                qic=qic,
                scale=float(result.scale) if hasattr(result, 'scale') else None,
                summary_text=summary,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_gee_data()
            result.warnings = [str(e)]
            return result

    def growth_curve_model(
        self,
        data: pd.DataFrame,
        outcome: str,
        time_variable: str,
        subject_id: str,
        covariates: Optional[List[str]] = None,
        polynomial_degree: int = 1
    ) -> MixedModelResult:
        """Fit a growth curve model (latent growth model).

        Args:
            data: DataFrame with longitudinal data
            outcome: Dependent variable name
            time_variable: Time variable
            subject_id: Subject identifier
            covariates: Additional covariates
            polynomial_degree: 1=linear, 2=quadratic, 3=cubic

        Returns:
            MixedModelResult object
        """
        covariates = covariates or []

        # Prepare data
        all_vars = [outcome, time_variable, subject_id] + covariates
        clean_data = data[all_vars].dropna()

        # Build fixed effects with polynomial time terms
        time_terms = [time_variable]
        if polynomial_degree >= 2:
            clean_data[f'{time_variable}_sq'] = clean_data[time_variable] ** 2
            time_terms.append(f'{time_variable}_sq')
        if polynomial_degree >= 3:
            clean_data[f'{time_variable}_cu'] = clean_data[time_variable] ** 3
            time_terms.append(f'{time_variable}_cu')

        fixed_effects = time_terms + covariates

        # Fit as mixed model with random intercept and slope
        return self.linear_mixed_model(
            data=clean_data,
            outcome=outcome,
            fixed_effects=fixed_effects,
            random_intercept=subject_id,
            random_slope=time_variable,
            time_variable=time_variable
        )

    def _generate_mixed_model_summary(
        self,
        result,
        outcome: str,
        fixed_effects: List[str],
        grouping: str,
        icc: Optional[float]
    ) -> str:
        """Generate interpretive summary for mixed model."""
        lines = [
            f"Linear mixed model for {outcome}:",
            f"- {int(result.nobs)} observations nested within groups defined by {grouping}",
        ]

        if icc is not None:
            lines.append(f"- ICC = {icc:.3f} ({icc*100:.1f}% of variance is between-group)")

        # Report significant fixed effects
        sig_effects = []
        for var in fixed_effects:
            if hasattr(result, 'pvalues') and var in result.pvalues.index:
                if result.pvalues[var] < self.alpha:
                    coef = result.fe_params[var]
                    direction = "positive" if coef > 0 else "negative"
                    sig_effects.append(f"{var} ({direction}, p={result.pvalues[var]:.4f})")

        if sig_effects:
            lines.append(f"- Significant predictors: {', '.join(sig_effects)}")
        else:
            lines.append("- No statistically significant predictors found")

        return '\n'.join(lines)

    def _generate_gee_summary(
        self,
        result,
        outcome: str,
        sig_predictors: List[str],
        correlation: str,
        family: str
    ) -> str:
        """Generate interpretive summary for GEE."""
        lines = [
            f"GEE model for {outcome}:",
            f"- Family: {family}, Correlation: {correlation}",
            f"- {int(result.nobs)} observations in {result.model.groups.nunique()} clusters",
        ]

        if sig_predictors:
            lines.append(f"- Significant predictors: {', '.join(sig_predictors)}")
        else:
            lines.append("- No statistically significant predictors")

        return '\n'.join(lines)

    def _insufficient_data(self, method: str) -> MixedModelResult:
        """Return result indicating insufficient data."""
        return MixedModelResult(
            model_type=method,
            formula="",
            n_observations=0,
            n_groups=0,
            converged=False,
            warnings=["Insufficient data for analysis"]
        )

    def _insufficient_gee_data(self) -> GEEResult:
        """Return GEE result indicating insufficient data."""
        return GEEResult(
            model_type="GEE",
            formula="",
            n_observations=0,
            n_clusters=0,
            correlation_structure="",
            warnings=["Insufficient data for analysis"]
        )
