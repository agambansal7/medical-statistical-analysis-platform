"""
Statistical Analysis Package
============================

Comprehensive statistical methods for medical research.

Core Modules:
- descriptive: Descriptive statistics and Table 1 generation
- comparative: T-tests, ANOVA, chi-square, non-parametric tests
- correlation: Correlation analysis (Pearson, Spearman, partial)
- regression: Linear, logistic, Poisson, mixed effects regression
- survival: Kaplan-Meier, log-rank, Cox regression
- diagnostic: ROC curves, sensitivity/specificity, calibration
- agreement: Kappa, ICC, Bland-Altman, Cronbach's alpha
- effect_sizes: Cohen's d, odds ratios, NNT
- power: Sample size and power calculations
- assumptions: Normality, homogeneity, multicollinearity tests

Advanced Modules:
- propensity_score: PSM, IPW, doubly robust estimation
- advanced_survival: Competing risks, RMST, landmark analysis
- meta_analysis: Fixed/random effects, heterogeneity, publication bias
- missing_data: MICE, sensitivity analysis, pattern visualization
- subgroup_analysis: Interaction testing, forest plots, credibility

New Advanced Modules (v3.0):
- longitudinal: Mixed effects models, GEE, growth curves
- causal_inference: IV/2SLS, RDD, Difference-in-Differences
- machine_learning: LASSO, Ridge, Random Forest, Gradient Boosting
- bayesian: Bayesian t-test, correlation, regression
- mediation: Baron-Kenny, bootstrap mediation, causal mediation
- sensitivity_analysis: E-values, tipping point, influence diagnostics
- power_calculator: Enhanced power analysis for complex designs
"""

# Core modules
from .descriptive import DescriptiveStats
from .comparative import ComparativeTests
from .correlation import CorrelationAnalysis
from .regression import RegressionAnalysis
from .survival import SurvivalAnalysis
from .diagnostic import DiagnosticTests
from .agreement import AgreementAnalysis
from .effect_sizes import EffectSizeCalculator
from .power import PowerAnalysis
from .assumptions import AssumptionChecker

# Advanced modules
from .propensity_score import PropensityScoreAnalysis
from .advanced_survival import AdvancedSurvivalAnalysis
from .meta_analysis import MetaAnalysis
from .missing_data import MissingDataHandler
from .subgroup_analysis import SubgroupAnalyzer

# New advanced modules (v3.0)
from .longitudinal import LongitudinalAnalysis
from .causal_inference import CausalInference
from .machine_learning import MachineLearning
from .bayesian import BayesianAnalysis
from .mediation import MediationAnalysis
from .sensitivity_analysis import SensitivityAnalysis
from .power_calculator import PowerCalculator

__all__ = [
    # Core
    "DescriptiveStats",
    "ComparativeTests",
    "CorrelationAnalysis",
    "RegressionAnalysis",
    "SurvivalAnalysis",
    "DiagnosticTests",
    "AgreementAnalysis",
    "EffectSizeCalculator",
    "PowerAnalysis",
    "AssumptionChecker",
    # Advanced
    "PropensityScoreAnalysis",
    "AdvancedSurvivalAnalysis",
    "MetaAnalysis",
    "MissingDataHandler",
    "SubgroupAnalyzer",
    # New Advanced (v3.0)
    "LongitudinalAnalysis",
    "CausalInference",
    "MachineLearning",
    "BayesianAnalysis",
    "MediationAnalysis",
    "SensitivityAnalysis",
    "PowerCalculator",
]

__version__ = '3.0.0'
