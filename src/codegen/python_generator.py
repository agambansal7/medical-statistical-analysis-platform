"""Python Code Generator for Statistical Analyses."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib


class PythonCodeGenerator:
    """Generate Python code for statistical analyses."""

    def __init__(self, seed: int = 42, include_docs: bool = True, include_install: bool = True):
        self.seed = seed
        self.include_docs = include_docs
        self.include_install = include_install

    def generate(self, analysis_type: str, parameters: Dict[str, Any],
                 data_info: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        """Generate Python code for analysis."""
        from .code_generator import GeneratedCode, CodeFormat

        seed = seed or self.seed
        method = getattr(self, f'_gen_{analysis_type}', None)

        if method is None:
            method = self._gen_generic

        code, packages = method(parameters, data_info, seed)
        full_code = self._build_full_script(code, packages, analysis_type, seed)

        return GeneratedCode(
            language='python',
            code=full_code,
            format=CodeFormat.PYTHON_SCRIPT,
            analysis_type=analysis_type,
            parameters=parameters,
            packages_required=packages,
            seed=seed,
            generated_at=datetime.now().isoformat(),
            code_hash=hashlib.sha256(full_code.encode()).hexdigest()[:12]
        )

    def _build_full_script(self, analysis_code: str, packages: List[str],
                           analysis_type: str, seed: int) -> str:
        """Build complete Python script."""
        sections = []

        # Header
        sections.append(f'''"""
Statistical Analysis: {analysis_type.replace('_', ' ').title()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Seed: {seed}

This code was auto-generated for reproducibility.
"""''')

        # Package installation
        if self.include_install and packages:
            pkg_str = ' '.join(packages)
            sections.append(f'''
# Install required packages (uncomment if needed)
# !pip install {pkg_str}
''')

        # Imports
        sections.append(self._get_imports(packages))

        # Seed
        sections.append(f'''
# Set random seed for reproducibility
np.random.seed({seed})
''')

        # Data loading placeholder
        sections.append('''
# =============================================================================
# DATA LOADING
# =============================================================================
# Load your data here
# df = pd.read_csv('your_data.csv')
# df = pd.read_excel('your_data.xlsx')

# Example: Create sample data for demonstration
''')

        # Analysis code
        sections.append(f'''
# =============================================================================
# ANALYSIS
# =============================================================================
{analysis_code}
''')

        return '\n'.join(sections)

    def _get_imports(self, packages: List[str]) -> str:
        """Generate import statements."""
        imports = ['import pandas as pd', 'import numpy as np']

        if 'scipy' in packages:
            imports.append('from scipy import stats')
        if 'statsmodels' in packages:
            imports.extend([
                'import statsmodels.api as sm',
                'import statsmodels.formula.api as smf',
                'from statsmodels.stats.multicomp import pairwise_tukeyhsd'
            ])
        if 'scikit-learn' in packages:
            imports.append('from sklearn.model_selection import train_test_split, cross_val_score')
        if 'lifelines' in packages:
            imports.extend([
                'from lifelines import KaplanMeierFitter, CoxPHFitter',
                'from lifelines.statistics import logrank_test'
            ])
        if 'pingouin' in packages:
            imports.append('import pingouin as pg')
        if 'matplotlib' in packages:
            imports.extend(['import matplotlib.pyplot as plt', 'import seaborn as sns'])

        return '\n'.join(imports)

    # =========================================================================
    # BASIC TESTS
    # =========================================================================

    def _gen_ttest(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Independent samples t-test
group1 = df[df['group'] == 'control']['outcome']
group2 = df[df['group'] == 'treatment']['outcome']

# Check normality
stat1, p1 = stats.shapiro(group1)
stat2, p2 = stats.shapiro(group2)
print(f"Shapiro-Wilk test - Group 1: W={stat1:.4f}, p={p1:.4f}")
print(f"Shapiro-Wilk test - Group 2: W={stat2:.4f}, p={p2:.4f}")

# Check homogeneity of variance
levene_stat, levene_p = stats.levene(group1, group2)
print(f"Levene's test: F={levene_stat:.4f}, p={levene_p:.4f}")

# Perform t-test (Welch's by default for robustness)
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

# Effect size (Cohen's d)
pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
cohens_d = (group1.mean() - group2.mean()) / pooled_std

# Results
print(f"\\nResults:")
print(f"Group 1 mean: {group1.mean():.3f} (SD: {group1.std():.3f})")
print(f"Group 2 mean: {group2.mean():.3f} (SD: {group2.std():.3f})")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.3f}")
print(f"95% CI for difference: {stats.t.interval(0.95, len(group1)+len(group2)-2, loc=group1.mean()-group2.mean(), scale=stats.sem(group1-group2.mean()))}")
'''
        return code, ['pandas', 'numpy', 'scipy']

    def _gen_ttest_paired(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Paired samples t-test
pre = df['pre_treatment']
post = df['post_treatment']

# Check normality of differences
diff = post - pre
stat, p = stats.shapiro(diff)
print(f"Shapiro-Wilk test on differences: W={stat:.4f}, p={p:.4f}")

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(pre, post)

# Effect size (Cohen's d for paired samples)
cohens_d = diff.mean() / diff.std()

print(f"\\nResults:")
print(f"Pre mean: {pre.mean():.3f} (SD: {pre.std():.3f})")
print(f"Post mean: {post.mean():.3f} (SD: {post.std():.3f})")
print(f"Mean difference: {diff.mean():.3f} (SD: {diff.std():.3f})")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.3f}")
'''
        return code, ['pandas', 'numpy', 'scipy']

    def _gen_anova(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# One-way ANOVA
# Assuming 'group' is the grouping variable and 'outcome' is the dependent variable

# Check assumptions
groups = [df[df['group'] == g]['outcome'] for g in df['group'].unique()]

# Levene's test for homogeneity of variance
levene_stat, levene_p = stats.levene(*groups)
print(f"Levene's test: F={levene_stat:.4f}, p={levene_p:.4f}")

# One-way ANOVA
f_stat, p_value = stats.f_oneway(*groups)

# Effect size (eta-squared)
ss_between = sum(len(g) * (g.mean() - df['outcome'].mean())**2 for g in groups)
ss_total = sum((df['outcome'] - df['outcome'].mean())**2)
eta_squared = ss_between / ss_total

print(f"\\nOne-way ANOVA Results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Eta-squared: {eta_squared:.4f}")

# Post-hoc tests (Tukey HSD) if significant
if p_value < 0.05:
    print("\\nPost-hoc Tukey HSD:")
    tukey = pairwise_tukeyhsd(df['outcome'], df['group'])
    print(tukey)
'''
        return code, ['pandas', 'numpy', 'scipy', 'statsmodels']

    def _gen_chi_square(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Chi-square test of independence
# Create contingency table
contingency = pd.crosstab(df['variable1'], df['variable2'])
print("Contingency Table:")
print(contingency)

# Chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"\\nChi-square Results:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.4f}")

# Effect size (Cramer's V)
n = contingency.sum().sum()
min_dim = min(contingency.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"Cramer's V: {cramers_v:.4f}")

# Check expected frequencies
print(f"\\nExpected frequencies:")
print(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))

if (expected < 5).any():
    print("\\nWarning: Some expected frequencies < 5. Consider Fisher's exact test.")
'''
        return code, ['pandas', 'numpy', 'scipy']

    def _gen_mann_whitney(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Mann-Whitney U test (non-parametric alternative to t-test)
group1 = df[df['group'] == 'control']['outcome']
group2 = df[df['group'] == 'treatment']['outcome']

# Perform test
statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

# Effect size (rank-biserial correlation)
n1, n2 = len(group1), len(group2)
r = 1 - (2 * statistic) / (n1 * n2)

print(f"Mann-Whitney U Results:")
print(f"Group 1 median: {group1.median():.3f} (IQR: {group1.quantile(0.25):.3f}-{group1.quantile(0.75):.3f})")
print(f"Group 2 median: {group2.median():.3f} (IQR: {group2.quantile(0.25):.3f}-{group2.quantile(0.75):.3f})")
print(f"U statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Rank-biserial correlation: {r:.4f}")
'''
        return code, ['pandas', 'numpy', 'scipy']

    # =========================================================================
    # REGRESSION
    # =========================================================================

    def _gen_linear_regression(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Multiple Linear Regression
# Define outcome and predictors
outcome = 'y'
predictors = ['x1', 'x2', 'x3']

# Fit model using statsmodels formula API
formula = f"{outcome} ~ " + " + ".join(predictors)
model = smf.ols(formula, data=df).fit()

# Print summary
print(model.summary())

# Confidence intervals
print("\\n95% Confidence Intervals:")
print(model.conf_int())

# Diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(model.resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# Scale-Location
axes[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)), alpha=0.5)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Residuals|')
axes[1, 0].set_title('Scale-Location')

# Residuals histogram
axes[1, 1].hist(model.resid, bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_title('Residuals Distribution')

plt.tight_layout()
plt.savefig('regression_diagnostics.png', dpi=150)
plt.show()
'''
        return code, ['pandas', 'numpy', 'scipy', 'statsmodels', 'matplotlib']

    def _gen_logistic_regression(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Logistic Regression
outcome = 'outcome_binary'
predictors = ['x1', 'x2', 'x3']

# Fit model
formula = f"{outcome} ~ " + " + ".join(predictors)
model = smf.logit(formula, data=df).fit()

# Print summary
print(model.summary())

# Odds ratios with 95% CI
odds_ratios = np.exp(model.params)
ci = np.exp(model.conf_int())
ci.columns = ['OR_lower', 'OR_upper']
results = pd.DataFrame({
    'Odds Ratio': odds_ratios,
    'OR 95% CI Lower': ci['OR_lower'],
    'OR 95% CI Upper': ci['OR_upper'],
    'p-value': model.pvalues
})
print("\\nOdds Ratios:")
print(results)

# Model performance
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

y_pred_prob = model.predict(df)
y_pred = (y_pred_prob >= 0.5).astype(int)
y_true = df[outcome]

print(f"\\nROC-AUC: {roc_auc_score(y_true, y_pred_prob):.4f}")
print("\\nClassification Report:")
print(classification_report(y_true, y_pred))
print("\\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
'''
        return code, ['pandas', 'numpy', 'statsmodels', 'scikit-learn']

    def _gen_cox_regression(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Cox Proportional Hazards Regression
from lifelines import CoxPHFitter

# Prepare data (needs duration and event columns)
duration_col = 'time'
event_col = 'event'
covariates = ['age', 'treatment', 'stage']

# Fit Cox model
cph = CoxPHFitter()
cph.fit(df[[duration_col, event_col] + covariates],
        duration_col=duration_col,
        event_col=event_col)

# Print summary
cph.print_summary()

# Hazard ratios
print("\\nHazard Ratios with 95% CI:")
print(np.exp(cph.summary[['coef', 'coef lower 95%', 'coef upper 95%']]))

# Check proportional hazards assumption
print("\\nProportional Hazards Test:")
cph.check_assumptions(df, show_plots=True)

# Survival curves
cph.plot()
plt.title('Covariate Effects on Survival')
plt.savefig('cox_effects.png', dpi=150)
plt.show()
'''
        return code, ['pandas', 'numpy', 'lifelines', 'matplotlib']

    # =========================================================================
    # SURVIVAL
    # =========================================================================

    def _gen_kaplan_meier(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Kaplan-Meier Survival Analysis
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

duration_col = 'time'
event_col = 'event'
group_col = 'treatment'

kmf = KaplanMeierFitter()

# Plot survival curves by group
fig, ax = plt.subplots(figsize=(10, 6))

for group in df[group_col].unique():
    mask = df[group_col] == group
    kmf.fit(df.loc[mask, duration_col],
            event_observed=df.loc[mask, event_col],
            label=f'{group_col}={group}')
    kmf.plot_survival_function(ax=ax, ci_show=True)

# Median survival times
print("Median Survival Times:")
for group in df[group_col].unique():
    mask = df[group_col] == group
    kmf.fit(df.loc[mask, duration_col], df.loc[mask, event_col])
    print(f"  {group}: {kmf.median_survival_time_:.2f}")

# Log-rank test
groups = df[group_col].unique()
group1_mask = df[group_col] == groups[0]
result = logrank_test(
    df.loc[group1_mask, duration_col],
    df.loc[~group1_mask, duration_col],
    event_observed_A=df.loc[group1_mask, event_col],
    event_observed_B=df.loc[~group1_mask, event_col]
)
print(f"\\nLog-rank test: χ²={result.test_statistic:.4f}, p={result.p_value:.4f}")

plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Kaplan-Meier Survival Curves')
plt.legend()
plt.savefig('kaplan_meier.png', dpi=150)
plt.show()
'''
        return code, ['pandas', 'numpy', 'lifelines', 'matplotlib']

    # =========================================================================
    # LONGITUDINAL
    # =========================================================================

    def _gen_mixed_model(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Linear Mixed Effects Model
import statsmodels.formula.api as smf

# Define model
# outcome ~ fixed effects with random intercepts for subjects
formula = "outcome ~ time + treatment + time:treatment"
model = smf.mixedlm(formula, data=df, groups=df["subject_id"])
result = model.fit()

print(result.summary())

# Random effects
print("\\nRandom Effects Variance:")
print(f"  Subject variance: {result.cov_re.iloc[0, 0]:.4f}")
print(f"  Residual variance: {result.scale:.4f}")

# ICC (Intraclass Correlation)
icc = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale)
print(f"  ICC: {icc:.4f}")

# Predictions
df['predicted'] = result.fittedvalues

# Plot trajectories
fig, ax = plt.subplots(figsize=(10, 6))
for subject in df['subject_id'].unique()[:10]:  # Plot first 10 subjects
    subj_data = df[df['subject_id'] == subject]
    ax.plot(subj_data['time'], subj_data['outcome'], 'o-', alpha=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Outcome')
ax.set_title('Individual Trajectories (sample)')
plt.savefig('mixed_model_trajectories.png', dpi=150)
plt.show()
'''
        return code, ['pandas', 'numpy', 'statsmodels', 'matplotlib']

    def _gen_gee(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Generalized Estimating Equations (GEE)
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Define model with exchangeable correlation structure
formula = "outcome ~ time + treatment"
model = smf.gee(
    formula,
    groups="subject_id",
    data=df,
    cov_struct=sm.cov_struct.Exchangeable()  # or Independence(), AR(1), etc.
)
result = model.fit()

print(result.summary())

# Working correlation
print(f"\\nWorking correlation: {result.cov_struct.summary()}")

# Robust standard errors are used by default
print("\\n(Using robust/sandwich standard errors)")
'''
        return code, ['pandas', 'numpy', 'statsmodels']

    # =========================================================================
    # CAUSAL INFERENCE
    # =========================================================================

    def _gen_instrumental_variables(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Instrumental Variables / Two-Stage Least Squares (2SLS)
from statsmodels.sandbox.regression.gmm import IV2SLS

# Variables
outcome = 'y'           # Dependent variable
endogenous = 'x_endog'  # Endogenous regressor
instrument = 'z'        # Instrument
exogenous = ['x1', 'x2']  # Other exogenous variables

# First stage: regress endogenous on instrument + exogenous
first_stage = smf.ols(f"{endogenous} ~ {instrument} + " + " + ".join(exogenous), data=df).fit()
print("First Stage Results:")
print(first_stage.summary())
print(f"\\nFirst-stage F-statistic for instrument: {first_stage.fvalue:.2f}")
print("(Rule of thumb: F > 10 indicates strong instrument)")

# Second stage using predicted values
df['x_endog_hat'] = first_stage.fittedvalues
second_stage = smf.ols(f"{outcome} ~ x_endog_hat + " + " + ".join(exogenous), data=df).fit()
print("\\nSecond Stage (2SLS) Results:")
print(second_stage.summary())

# Alternatively, use IV2SLS directly
# Prepare matrices
y = df[outcome]
X_endog = df[[endogenous]]
X_exog = sm.add_constant(df[exogenous])
Z = sm.add_constant(df[[instrument] + exogenous])

iv_model = IV2SLS(y, X_exog, X_endog, Z).fit()
print("\\nIV2SLS Results:")
print(iv_model.summary())
'''
        return code, ['pandas', 'numpy', 'statsmodels']

    def _gen_difference_in_differences(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Difference-in-Differences (DiD) Analysis
# Data should have: outcome, treated (0/1), post (0/1)

# Create interaction term
df['treated_post'] = df['treated'] * df['post']

# DiD regression
formula = "outcome ~ treated + post + treated_post"
model = smf.ols(formula, data=df).fit()

print("Difference-in-Differences Results:")
print(model.summary())

# The DiD estimate is the coefficient on treated_post
did_estimate = model.params['treated_post']
did_se = model.bse['treated_post']
did_ci = model.conf_int().loc['treated_post']

print(f"\\nDiD Estimate: {did_estimate:.4f}")
print(f"Standard Error: {did_se:.4f}")
print(f"95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
print(f"p-value: {model.pvalues['treated_post']:.4f}")

# Visualization
import matplotlib.pyplot as plt
means = df.groupby(['treated', 'post'])['outcome'].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 6))
means.T.plot(ax=ax, marker='o', linewidth=2)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre', 'Post'])
ax.set_ylabel('Mean Outcome')
ax.set_title('Difference-in-Differences')
ax.legend(['Control', 'Treatment'])
plt.savefig('did_plot.png', dpi=150)
plt.show()
'''
        return code, ['pandas', 'numpy', 'statsmodels', 'matplotlib']

    # =========================================================================
    # MACHINE LEARNING
    # =========================================================================

    def _gen_lasso(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = f'''
# LASSO Regression with Cross-Validation
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Prepare data
X = df[['x1', 'x2', 'x3', 'x4', 'x5']]  # Features
y = df['outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LASSO with cross-validation to find optimal alpha
lasso_cv = LassoCV(cv=10, random_state={seed})
lasso_cv.fit(X_scaled, y)

print(f"Optimal alpha: {{lasso_cv.alpha_:.6f}}")
print(f"R² score: {{lasso_cv.score(X_scaled, y):.4f}}")

# Coefficients
coef_df = pd.DataFrame({{
    'Feature': X.columns,
    'Coefficient': lasso_cv.coef_,
    'Selected': lasso_cv.coef_ != 0
}})
print("\\nLASSO Coefficients:")
print(coef_df)
print(f"\\nSelected {{(coef_df['Selected']).sum()}} of {{len(coef_df)}} features")

# Cross-validation scores
cv_scores = cross_val_score(Lasso(alpha=lasso_cv.alpha_), X_scaled, y, cv=10)
print(f"\\nCross-validation R²: {{cv_scores.mean():.4f}} (±{{cv_scores.std()*2:.4f}})")

# Regularization path
alphas = lasso_cv.alphas_
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, lasso_cv.mse_path_.mean(axis=1))
plt.axvline(lasso_cv.alpha_, color='r', linestyle='--', label=f'Optimal α={{lasso_cv.alpha_:.4f}}')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('LASSO Regularization Path')
plt.legend()
plt.savefig('lasso_path.png', dpi=150)
plt.show()
'''
        return code, ['pandas', 'numpy', 'scikit-learn', 'matplotlib']

    def _gen_random_forest(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = f'''
# Random Forest Analysis
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data
X = df[['x1', 'x2', 'x3', 'x4', 'x5']]
y = df['outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state={seed})

# Fit Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state={seed}, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Performance
print("Random Forest Results:")
print(f"R² (test): {{r2_score(y_test, y_pred):.4f}}")
print(f"RMSE: {{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}}")

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=10)
print(f"CV R²: {{cv_scores.mean():.4f}} (±{{cv_scores.std()*2:.4f}})")

# Feature importance
importance_df = pd.DataFrame({{
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}}).sort_values('Importance', ascending=False)

print("\\nFeature Importances:")
print(importance_df)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=150)
plt.show()
'''
        return code, ['pandas', 'numpy', 'scikit-learn', 'matplotlib']

    # =========================================================================
    # BAYESIAN
    # =========================================================================

    def _gen_bayesian_ttest(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = f'''
# Bayesian t-test using pingouin
import pingouin as pg

group1 = df[df['group'] == 'control']['outcome']
group2 = df[df['group'] == 'treatment']['outcome']

# Bayesian independent t-test
result = pg.ttest(group1, group2, paired=False)
print("Frequentist t-test:")
print(result)

# Compute Bayes Factor
bf = pg.bayesfactor_ttest(t=result['T'].values[0],
                          nx=len(group1),
                          ny=len(group2),
                          paired=False)
print(f"\\nBayes Factor (BF10): {{bf:.4f}}")

# Interpretation
if bf > 100:
    interpretation = "Extreme evidence for H1"
elif bf > 30:
    interpretation = "Very strong evidence for H1"
elif bf > 10:
    interpretation = "Strong evidence for H1"
elif bf > 3:
    interpretation = "Moderate evidence for H1"
elif bf > 1:
    interpretation = "Anecdotal evidence for H1"
elif bf > 1/3:
    interpretation = "Anecdotal evidence for H0"
elif bf > 1/10:
    interpretation = "Moderate evidence for H0"
else:
    interpretation = "Strong evidence for H0"

print(f"Interpretation: {{interpretation}}")
'''
        return code, ['pandas', 'numpy', 'pingouin']

    # =========================================================================
    # MEDIATION
    # =========================================================================

    def _gen_mediation(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = f'''
# Mediation Analysis (Baron-Kenny with Bootstrap)
from scipy import stats

X = 'treatment'  # Independent variable
M = 'mediator'   # Mediator
Y = 'outcome'    # Dependent variable

# Path a: X -> M
path_a = smf.ols(f"{{M}} ~ {{X}}", data=df).fit()
a = path_a.params[X]
print(f"Path a (X->M): {{a:.4f}}, p={{path_a.pvalues[X]:.4f}}")

# Path b and c': M -> Y controlling for X
path_bc = smf.ols(f"{{Y}} ~ {{X}} + {{M}}", data=df).fit()
b = path_bc.params[M]
c_prime = path_bc.params[X]
print(f"Path b (M->Y|X): {{b:.4f}}, p={{path_bc.pvalues[M]:.4f}}")
print(f"Path c' (direct): {{c_prime:.4f}}, p={{path_bc.pvalues[X]:.4f}}")

# Path c: Total effect X -> Y
path_c = smf.ols(f"{{Y}} ~ {{X}}", data=df).fit()
c = path_c.params[X]
print(f"Path c (total): {{c:.4f}}, p={{path_c.pvalues[X]:.4f}}")

# Indirect effect
indirect = a * b
print(f"\\nIndirect effect (a*b): {{indirect:.4f}}")
print(f"Direct effect (c'): {{c_prime:.4f}}")
print(f"Total effect (c): {{c:.4f}}")
print(f"Proportion mediated: {{indirect/c*100:.1f}}%")

# Bootstrap confidence interval for indirect effect
n_boot = 5000
boot_indirect = []
np.random.seed({seed})

for _ in range(n_boot):
    boot_idx = np.random.choice(len(df), size=len(df), replace=True)
    boot_df = df.iloc[boot_idx]

    a_boot = smf.ols(f"{{M}} ~ {{X}}", data=boot_df).fit().params[X]
    b_boot = smf.ols(f"{{Y}} ~ {{X}} + {{M}}", data=boot_df).fit().params[M]
    boot_indirect.append(a_boot * b_boot)

ci_lower, ci_upper = np.percentile(boot_indirect, [2.5, 97.5])
print(f"\\nBootstrap 95% CI for indirect effect: [{{ci_lower:.4f}}, {{ci_upper:.4f}}]")
print(f"Significant mediation: {{not (ci_lower <= 0 <= ci_upper)}}")
'''
        return code, ['pandas', 'numpy', 'scipy', 'statsmodels']

    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================

    def _gen_e_value(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# E-Value Sensitivity Analysis
# E-value quantifies unmeasured confounding needed to explain away an effect

def compute_e_value(rr):
    """Compute E-value for a risk/odds/hazard ratio."""
    if rr >= 1:
        return rr + np.sqrt(rr * (rr - 1))
    else:
        rr_inv = 1 / rr
        return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

def e_value_ci(rr, ci_lower, ci_upper):
    """E-value for point estimate and CI bound closest to null."""
    e_point = compute_e_value(rr)

    # CI bound closest to null
    if rr >= 1:
        e_ci = compute_e_value(ci_lower) if ci_lower >= 1 else 1.0
    else:
        e_ci = compute_e_value(ci_upper) if ci_upper <= 1 else 1.0

    return e_point, e_ci

# Example: Odds ratio from logistic regression
OR = 2.5
OR_lower = 1.8
OR_upper = 3.5

e_point, e_ci = e_value_ci(OR, OR_lower, OR_upper)

print("E-Value Analysis:")
print(f"Observed OR: {OR:.2f} (95% CI: {OR_lower:.2f}-{OR_upper:.2f})")
print(f"E-value for point estimate: {e_point:.2f}")
print(f"E-value for CI bound: {e_ci:.2f}")
print()
print("Interpretation:")
print(f"An unmeasured confounder would need to be associated with both")
print(f"the exposure and outcome by a risk ratio of {e_point:.2f}-fold each")
print(f"(above and beyond measured confounders) to explain away the effect.")
'''
        return code, ['pandas', 'numpy']

    # =========================================================================
    # POWER ANALYSIS
    # =========================================================================

    def _gen_power_ttest(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        code = '''
# Power Analysis for t-test
from scipy import stats

def power_ttest(effect_size, n1, n2=None, alpha=0.05, alternative='two-sided'):
    """Calculate power for independent samples t-test."""
    if n2 is None:
        n2 = n1

    # Pooled standard error
    se = np.sqrt(1/n1 + 1/n2)

    # Non-centrality parameter
    ncp = effect_size / se

    # Degrees of freedom
    df = n1 + n2 - 2

    # Critical value
    if alternative == 'two-sided':
        crit = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
    else:
        crit = stats.t.ppf(1 - alpha, df)
        power = 1 - stats.nct.cdf(crit, df, ncp)

    return power

def sample_size_ttest(effect_size, power=0.8, alpha=0.05, ratio=1):
    """Find sample size for desired power."""
    for n in range(5, 10000):
        n2 = int(n * ratio)
        if power_ttest(effect_size, n, n2, alpha) >= power:
            return n, n2
    return None, None

# Example calculations
effect_size = 0.5  # Cohen's d (medium effect)
alpha = 0.05

print("Power Analysis for Independent Samples t-test")
print("=" * 50)
print(f"Effect size (Cohen's d): {effect_size}")
print(f"Alpha: {alpha}")
print()

# Power for various sample sizes
print("Power by sample size (per group):")
for n in [20, 30, 50, 100, 200]:
    pwr = power_ttest(effect_size, n, n, alpha)
    print(f"  n = {n}: power = {pwr:.3f}")

# Sample size for 80% power
n1, n2 = sample_size_ttest(effect_size, power=0.8)
print(f"\\nSample size for 80% power: {n1} per group")

# Power curve
sample_sizes = np.arange(10, 200, 5)
powers = [power_ttest(effect_size, n, n, alpha) for n in sample_sizes]

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, powers, linewidth=2)
plt.axhline(y=0.8, color='r', linestyle='--', label='80% power')
plt.xlabel('Sample Size (per group)')
plt.ylabel('Power')
plt.title(f'Power Curve for t-test (d={effect_size})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('power_curve.png', dpi=150)
plt.show()
'''
        return code, ['pandas', 'numpy', 'scipy', 'matplotlib']

    # =========================================================================
    # GENERIC FALLBACK
    # =========================================================================

    def _gen_generic(self, params: Dict, data_info: Dict, seed: int) -> tuple:
        """Generic code template for unsupported analyses."""
        code = '''
# Generic Statistical Analysis Template
# Modify this template for your specific analysis

# Descriptive statistics
print("Descriptive Statistics:")
print(df.describe())

# For numeric variables
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"\\n{col}:")
    print(f"  Mean: {df[col].mean():.3f}")
    print(f"  SD: {df[col].std():.3f}")
    print(f"  Median: {df[col].median():.3f}")
    print(f"  IQR: {df[col].quantile(0.25):.3f} - {df[col].quantile(0.75):.3f}")

# For categorical variables
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    print(f"\\n{col}:")
    print(df[col].value_counts())
'''
        return code, ['pandas', 'numpy']
