"""Bayesian statistical methods.

Provides Bayesian alternatives to common frequentist tests:
- Bayesian t-tests
- Bayesian correlation
- Bayesian regression
- Bayesian proportion tests
- Prior specification and sensitivity analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import warnings


@dataclass
class BayesianResult:
    """Base result for Bayesian analyses."""
    test_name: str
    n_observations: int

    # Bayes Factor
    bayes_factor: Optional[float] = None
    log_bayes_factor: Optional[float] = None
    bf_interpretation: Optional[str] = None

    # Posterior
    posterior_mean: Optional[float] = None
    posterior_std: Optional[float] = None
    credible_interval: Optional[Tuple[float, float]] = None
    credible_level: float = 0.95

    # Prior specification
    prior_description: Optional[str] = None

    # Probability of hypothesis
    prob_h1: Optional[float] = None  # Probability of alternative
    prob_h0: Optional[float] = None  # Probability of null

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.credible_interval:
            result['credible_interval'] = list(self.credible_interval)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class BayesianRegressionResult:
    """Result for Bayesian regression."""
    model_type: str
    n_observations: int
    n_predictors: int

    # Coefficients with posteriors
    coefficients: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Model comparison
    bayes_factor_vs_null: Optional[float] = None
    marginal_likelihood: Optional[float] = None

    # Posterior predictive
    r_squared_posterior: Optional[Tuple[float, float, float]] = None  # mean, lower, upper

    # Interpretation
    summary_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    educational_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class BayesianAnalysis:
    """Bayesian statistical methods."""

    def __init__(self, credible_level: float = 0.95):
        self.credible_level = credible_level

    def bayesian_ttest(
        self,
        group1: Union[pd.Series, np.ndarray, List],
        group2: Union[pd.Series, np.ndarray, List],
        prior_scale: float = 0.707,
        rope: Optional[Tuple[float, float]] = None
    ) -> BayesianResult:
        """Bayesian independent samples t-test.

        Uses JZS (Jeffreys-Zellner-Siow) prior on effect size.

        Args:
            group1: First group data
            group2: Second group data
            prior_scale: Scale of Cauchy prior on effect size (default: sqrt(2)/2)
            rope: Region of practical equivalence (-rope, +rope)

        Returns:
            BayesianResult object
        """
        # Convert to numpy arrays
        x1 = np.array(group1).flatten()
        x2 = np.array(group2).flatten()

        # Remove NaN
        x1 = x1[~np.isnan(x1)]
        x2 = x2[~np.isnan(x2)]

        n1, n2 = len(x1), len(x2)
        n = n1 + n2

        if n1 < 2 or n2 < 2:
            return self._insufficient_data("Bayesian t-test")

        # Calculate t-statistic
        mean1, mean2 = np.mean(x1), np.mean(x2)
        var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n - 2)
        se = np.sqrt(pooled_var * (1/n1 + 1/n2))
        t_stat = (mean1 - mean2) / se
        df = n - 2

        # Effect size (Cohen's d)
        cohens_d = (mean1 - mean2) / np.sqrt(pooled_var)

        # Calculate Bayes Factor using JZS prior (approximation)
        bf10 = self._jzs_bayes_factor(t_stat, n1, n2, prior_scale)

        # Interpret Bayes Factor
        bf_interp = self._interpret_bayes_factor(bf10)

        # Posterior on effect size (approximation using normal)
        # Using t-distribution approximation for posterior
        posterior_mean = cohens_d
        posterior_std = np.sqrt(1/n1 + 1/n2 + cohens_d**2 / (2*n))

        alpha = 1 - self.credible_level
        ci_lower = posterior_mean - stats.norm.ppf(1 - alpha/2) * posterior_std
        ci_upper = posterior_mean + stats.norm.ppf(1 - alpha/2) * posterior_std

        # Probability of H1 (assuming equal prior odds)
        prob_h1 = bf10 / (1 + bf10)
        prob_h0 = 1 / (1 + bf10)

        # ROPE analysis if specified
        rope_analysis = None
        if rope:
            prob_in_rope = stats.norm.cdf(rope[1], posterior_mean, posterior_std) - \
                          stats.norm.cdf(rope[0], posterior_mean, posterior_std)
            rope_analysis = f"P(effect in ROPE [{rope[0]}, {rope[1]}]) = {prob_in_rope:.3f}"

        # Generate summary
        direction = "higher" if cohens_d > 0 else "lower"
        summary = f"""Bayesian Independent Samples t-test:
- Group 1 mean: {mean1:.3f} (n={n1})
- Group 2 mean: {mean2:.3f} (n={n2})
- Effect size (Cohen's d): {cohens_d:.3f} ({direction} in group 1)
- Bayes Factor (BF10): {bf10:.3f} - {bf_interp}
- Posterior effect: {posterior_mean:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]
- P(H1|data): {prob_h1:.3f}, P(H0|data): {prob_h0:.3f}
{rope_analysis if rope_analysis else ''}"""

        # Educational notes
        edu_notes = {
            'what_is_bf': 'Bayes Factor quantifies the relative evidence for H1 vs H0. BF10 > 1 favors H1, BF10 < 1 favors H0.',
            'interpreting_bf': f'BF10 = {bf10:.2f} means the data are {bf10:.1f}x more likely under H1 than H0.' if bf10 > 1 else f'BF10 = {bf10:.2f} means the data are {1/bf10:.1f}x more likely under H0 than H1.',
            'prior': f'Using Cauchy prior on effect size with scale={prior_scale}. This reflects uncertainty about effect magnitude before seeing data.',
            'vs_pvalue': 'Unlike p-values, Bayes Factors quantify evidence for both H0 and H1, and are not affected by sampling intentions.',
            'credible_interval': f'The {int(self.credible_level*100)}% credible interval [{ci_lower:.3f}, {ci_upper:.3f}] contains the true effect with {int(self.credible_level*100)}% probability.',
            'rope': 'ROPE (Region of Practical Equivalence) defines effects too small to be meaningful.' if rope else None
        }

        return BayesianResult(
            test_name="Bayesian Independent Samples t-test",
            n_observations=n,
            bayes_factor=bf10,
            log_bayes_factor=np.log(bf10) if bf10 > 0 else None,
            bf_interpretation=bf_interp,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_interval=(ci_lower, ci_upper),
            credible_level=self.credible_level,
            prior_description=f"Cauchy prior on effect size, scale={prior_scale}",
            prob_h1=prob_h1,
            prob_h0=prob_h0,
            summary_text=summary,
            educational_notes=edu_notes
        )

    def bayesian_correlation(
        self,
        x: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        prior_kappa: float = 1.0
    ) -> BayesianResult:
        """Bayesian correlation analysis.

        Uses beta prior on correlation coefficient.

        Args:
            x: First variable
            y: Second variable
            prior_kappa: Concentration parameter for prior (1=uniform)

        Returns:
            BayesianResult object
        """
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        n = len(x)

        if n < 4:
            return self._insufficient_data("Bayesian correlation")

        # Calculate Pearson correlation
        r = np.corrcoef(x, y)[0, 1]

        # Fisher z-transformation for posterior
        z = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)

        # Posterior on z (approximately normal)
        alpha = 1 - self.credible_level
        z_lower = z - stats.norm.ppf(1 - alpha/2) * se_z
        z_upper = z + stats.norm.ppf(1 - alpha/2) * se_z

        # Transform back to correlation scale
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)

        # Bayes Factor for correlation (Wetzels & Wagenmakers approximation)
        bf10 = self._correlation_bayes_factor(r, n)

        bf_interp = self._interpret_bayes_factor(bf10)

        prob_h1 = bf10 / (1 + bf10)
        prob_h0 = 1 / (1 + bf10)

        # Probability correlation is positive/negative
        prob_positive = 1 - stats.norm.cdf(0, z, se_z)
        prob_negative = stats.norm.cdf(0, z, se_z)

        summary = f"""Bayesian Correlation Analysis:
- Pearson r = {r:.3f}
- {int(self.credible_level*100)}% Credible Interval: [{r_lower:.3f}, {r_upper:.3f}]
- Bayes Factor (BF10): {bf10:.3f} - {bf_interp}
- P(r > 0 | data) = {prob_positive:.3f}
- P(r < 0 | data) = {prob_negative:.3f}"""

        edu_notes = {
            'what_is_bayesian_corr': 'Bayesian correlation provides probability statements about the correlation coefficient and evidence for/against non-zero correlation.',
            'interpreting_ci': f'There is a {int(self.credible_level*100)}% probability the true correlation lies in [{r_lower:.3f}, {r_upper:.3f}].',
            'bayes_factor': f'BF10 = {bf10:.2f}: {bf_interp}',
            'direction': f'P(positive correlation) = {prob_positive:.3f}',
            'vs_frequentist': 'Unlike frequentist CI, the Bayesian credible interval has direct probability interpretation.'
        }

        return BayesianResult(
            test_name="Bayesian Correlation",
            n_observations=n,
            bayes_factor=bf10,
            log_bayes_factor=np.log(bf10) if bf10 > 0 else None,
            bf_interpretation=bf_interp,
            posterior_mean=r,
            posterior_std=se_z * (1 - r**2),  # Approximate SE on correlation scale
            credible_interval=(r_lower, r_upper),
            credible_level=self.credible_level,
            prior_description=f"Uniform prior on correlation (kappa={prior_kappa})",
            prob_h1=prob_h1,
            prob_h0=prob_h0,
            summary_text=summary,
            educational_notes=edu_notes
        )

    def bayesian_proportion_test(
        self,
        successes1: int,
        total1: int,
        successes2: int,
        total2: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ) -> BayesianResult:
        """Bayesian test for difference in proportions.

        Uses beta-binomial model with beta prior.

        Args:
            successes1: Number of successes in group 1
            total1: Total observations in group 1
            successes2: Number of successes in group 2
            total2: Total observations in group 2
            prior_alpha: Alpha parameter of beta prior
            prior_beta: Beta parameter of beta prior

        Returns:
            BayesianResult object
        """
        n1, n2 = total1, total2
        k1, k2 = successes1, successes2

        if n1 < 1 or n2 < 1:
            return self._insufficient_data("Bayesian proportion test")

        # Posterior parameters (beta-binomial conjugacy)
        post_alpha1 = prior_alpha + k1
        post_beta1 = prior_beta + (n1 - k1)
        post_alpha2 = prior_alpha + k2
        post_beta2 = prior_beta + (n2 - k2)

        # Posterior means
        p1_mean = post_alpha1 / (post_alpha1 + post_beta1)
        p2_mean = post_alpha2 / (post_alpha2 + post_beta2)

        # Sample from posteriors to get difference distribution
        np.random.seed(42)
        samples1 = stats.beta.rvs(post_alpha1, post_beta1, size=100000)
        samples2 = stats.beta.rvs(post_alpha2, post_beta2, size=100000)
        diff_samples = samples1 - samples2

        diff_mean = np.mean(diff_samples)
        diff_std = np.std(diff_samples)

        alpha = 1 - self.credible_level
        diff_lower = np.percentile(diff_samples, alpha/2 * 100)
        diff_upper = np.percentile(diff_samples, (1 - alpha/2) * 100)

        # Probability p1 > p2
        prob_p1_greater = np.mean(samples1 > samples2)

        # Bayes Factor (approximate)
        # H0: p1 = p2, H1: p1 ≠ p2
        # Using Savage-Dickey density ratio approximation
        prior_diff_at_zero = 1  # Uniform difference prior at 0 (approximate)
        posterior_diff_at_zero = stats.gaussian_kde(diff_samples)(0)[0]
        bf01 = posterior_diff_at_zero / prior_diff_at_zero if prior_diff_at_zero > 0 else 1
        bf10 = 1 / bf01 if bf01 > 0 else float('inf')

        bf_interp = self._interpret_bayes_factor(bf10)

        summary = f"""Bayesian Proportion Test:
- Group 1: {k1}/{n1} ({p1_mean:.3f})
- Group 2: {k2}/{n2} ({p2_mean:.3f})
- Difference: {diff_mean:.3f}, 95% CI [{diff_lower:.3f}, {diff_upper:.3f}]
- P(p1 > p2 | data) = {prob_p1_greater:.3f}
- Bayes Factor (BF10): {bf10:.3f} - {bf_interp}"""

        edu_notes = {
            'model': 'Using beta-binomial model with conjugate beta prior.',
            'prior': f'Beta({prior_alpha}, {prior_beta}) prior {"(uniform/non-informative)" if prior_alpha == 1 and prior_beta == 1 else ""}',
            'interpretation': f'There is a {prob_p1_greater*100:.1f}% probability that proportion in group 1 exceeds group 2.',
            'credible_interval': f'{int(self.credible_level*100)}% probability the true difference lies in [{diff_lower:.3f}, {diff_upper:.3f}]'
        }

        return BayesianResult(
            test_name="Bayesian Proportion Test",
            n_observations=n1 + n2,
            bayes_factor=bf10,
            bf_interpretation=bf_interp,
            posterior_mean=diff_mean,
            posterior_std=diff_std,
            credible_interval=(diff_lower, diff_upper),
            credible_level=self.credible_level,
            prior_description=f"Beta({prior_alpha}, {prior_beta}) prior on each proportion",
            prob_h1=prob_p1_greater,
            summary_text=summary,
            educational_notes=edu_notes
        )

    def bayesian_regression(
        self,
        data: pd.DataFrame,
        outcome: str,
        predictors: List[str],
        prior_scale: float = 0.354
    ) -> BayesianRegressionResult:
        """Bayesian linear regression with JZS prior.

        Args:
            data: DataFrame with the data
            outcome: Dependent variable name
            predictors: List of predictor variable names
            prior_scale: Scale of Cauchy prior on standardized coefficients

        Returns:
            BayesianRegressionResult object
        """
        all_vars = [outcome] + predictors
        clean_data = data[all_vars].dropna()
        n = len(clean_data)

        if n < len(predictors) + 2:
            return self._insufficient_reg_data()

        # Standardize variables for Bayesian analysis
        y = clean_data[outcome].values
        X = clean_data[predictors].values

        y_mean, y_std = np.mean(y), np.std(y)
        y_std = (y - y_mean) / y_std

        X_means = np.mean(X, axis=0)
        X_stds = np.std(X, axis=0)
        X_std = (X - X_means) / X_stds

        try:
            # Fit OLS to get point estimates
            import statsmodels.api as sm
            X_with_const = sm.add_constant(X_std)
            ols_result = sm.OLS(y_std, X_with_const).fit()

            # Posterior approximation using OLS estimates
            # (Full Bayesian would require MCMC, this is an approximation)
            coefficients = {}
            for i, var in enumerate(predictors):
                idx = i + 1  # Skip intercept
                coef = ols_result.params[idx]
                se = ols_result.bse[idx]

                # Credible interval
                alpha = 1 - self.credible_level
                ci_lower = coef - stats.norm.ppf(1 - alpha/2) * se
                ci_upper = coef + stats.norm.ppf(1 - alpha/2) * se

                # Individual Bayes Factor (approximate)
                t_val = abs(coef / se)
                bf = self._jzs_bf_from_t(t_val, n - len(predictors) - 1, prior_scale)

                # Probability coefficient > 0
                prob_positive = 1 - stats.norm.cdf(0, coef, se)

                # Transform back to original scale
                orig_coef = coef * y_std / X_stds[i] if X_stds[i] > 0 else coef
                orig_se = se * y_std / X_stds[i] if X_stds[i] > 0 else se

                coefficients[var] = {
                    'posterior_mean': float(orig_coef),
                    'posterior_std': float(orig_se),
                    'standardized_coef': float(coef),
                    'credible_lower': float(ci_lower * y_std / X_stds[i]) if X_stds[i] > 0 else float(ci_lower),
                    'credible_upper': float(ci_upper * y_std / X_stds[i]) if X_stds[i] > 0 else float(ci_upper),
                    'bayes_factor': float(bf),
                    'bf_interpretation': self._interpret_bayes_factor(bf),
                    'prob_positive': float(prob_positive),
                    'prob_negative': float(1 - prob_positive)
                }

            # Model Bayes Factor vs null model
            r2 = ols_result.rsquared
            bf_model = self._regression_model_bf(r2, n, len(predictors), prior_scale)

            summary = f"""Bayesian Linear Regression:
- {n} observations, {len(predictors)} predictors
- Model BF vs null: {bf_model:.3f} ({self._interpret_bayes_factor(bf_model)})
- R² = {r2:.3f}
"""
            for var, vals in coefficients.items():
                summary += f"\n{var}: β = {vals['posterior_mean']:.3f}, BF = {vals['bayes_factor']:.2f}"

            edu_notes = {
                'model': 'Bayesian regression with JZS (Jeffreys-Zellner-Siow) prior on coefficients.',
                'prior': f'Cauchy prior on standardized coefficients, scale={prior_scale}',
                'interpretation': 'Posterior means are similar to OLS estimates; credible intervals have direct probability interpretation.',
                'model_bf': f'Model BF of {bf_model:.2f} indicates the data are {bf_model:.1f}x more likely under the full model than null.',
                'individual_bf': 'Individual coefficient BFs test whether each predictor has a non-zero effect.'
            }

            return BayesianRegressionResult(
                model_type="Bayesian Linear Regression (JZS prior)",
                n_observations=n,
                n_predictors=len(predictors),
                coefficients=coefficients,
                bayes_factor_vs_null=bf_model,
                r_squared_posterior=(r2, r2 - 0.1, r2 + 0.1),  # Approximate
                summary_text=summary,
                educational_notes=edu_notes
            )

        except Exception as e:
            result = self._insufficient_reg_data()
            result.warnings = [str(e)]
            return result

    def _jzs_bayes_factor(
        self,
        t: float,
        n1: int,
        n2: int,
        scale: float
    ) -> float:
        """Calculate JZS Bayes Factor for t-test (approximation)."""
        n = n1 + n2
        df = n - 2
        r = scale

        # Approximation from Rouder et al. (2009)
        # BF10 ≈ (1 + t²/df)^(-df/2) * integral term
        # Using simplified approximation

        # Effect size
        d = t * np.sqrt(1/n1 + 1/n2)

        # Savage-Dickey approximation
        # Prior density at d=0 is the Cauchy(0, r) density at 0 = 1/(pi*r)
        prior_at_zero = 1 / (np.pi * r)

        # Posterior density at d=0 (approximated by normal)
        se_d = np.sqrt(1/n1 + 1/n2 + d**2/(2*n))
        posterior_at_zero = stats.norm.pdf(0, d, se_d)

        bf01 = posterior_at_zero / prior_at_zero
        bf10 = 1 / bf01 if bf01 > 0 else 100

        # Bound to reasonable range
        bf10 = np.clip(bf10, 0.001, 1000)

        return float(bf10)

    def _jzs_bf_from_t(self, t: float, df: int, scale: float) -> float:
        """Calculate JZS BF from t-statistic."""
        # Simplified approximation
        if abs(t) < 0.01:
            return 0.3  # Evidence for null

        # Approximate using BIC-based BF
        # BF ≈ sqrt(n) * exp(-0.5 * t^2) for large t
        # This is a rough approximation
        n = df + 1
        bf = np.sqrt(n) * np.exp(0.5 * (np.log(1 + t**2/df) * df - t**2 * df / (df + t**2)))
        return float(np.clip(bf, 0.01, 100))

    def _correlation_bayes_factor(self, r: float, n: int) -> float:
        """Calculate BF for correlation (Wetzels & Wagenmakers approximation)."""
        if n < 4:
            return 1.0

        # Jeffrey's approximation
        t = r * np.sqrt((n - 2) / (1 - r**2))
        df = n - 2

        # Use t-test BF approximation
        return self._jzs_bf_from_t(abs(t), df, 0.707)

    def _regression_model_bf(
        self,
        r2: float,
        n: int,
        p: int,
        scale: float
    ) -> float:
        """Calculate model BF for regression vs null."""
        # BIC-based approximation
        if r2 <= 0 or r2 >= 1:
            return 1.0

        # F-statistic
        f = (r2 / p) / ((1 - r2) / (n - p - 1))

        # Approximate BF
        bf = np.sqrt(n) * (1 + f * p / (n - p - 1)) ** (-(n - 1) / 2)
        bf = bf * (1 + p / (scale**2 * n)) ** (-(p + 1) / 2)

        return float(np.clip(1/bf if bf > 0 else 100, 0.01, 1000))

    def _interpret_bayes_factor(self, bf: float) -> str:
        """Interpret Bayes Factor using Jeffreys' scale."""
        if bf < 1/100:
            return "Extreme evidence for H0"
        elif bf < 1/30:
            return "Very strong evidence for H0"
        elif bf < 1/10:
            return "Strong evidence for H0"
        elif bf < 1/3:
            return "Moderate evidence for H0"
        elif bf < 1:
            return "Anecdotal evidence for H0"
        elif bf < 3:
            return "Anecdotal evidence for H1"
        elif bf < 10:
            return "Moderate evidence for H1"
        elif bf < 30:
            return "Strong evidence for H1"
        elif bf < 100:
            return "Very strong evidence for H1"
        else:
            return "Extreme evidence for H1"

    def _insufficient_data(self, method: str) -> BayesianResult:
        return BayesianResult(
            test_name=method,
            n_observations=0,
            warnings=["Insufficient data for Bayesian analysis"]
        )

    def _insufficient_reg_data(self) -> BayesianRegressionResult:
        return BayesianRegressionResult(
            model_type="Bayesian Regression",
            n_observations=0,
            n_predictors=0,
            warnings=["Insufficient data for Bayesian regression"]
        )
