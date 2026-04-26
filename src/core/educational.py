"""Educational Mode Module.

Provides statistical education and explanations:
- Explain why each test was chosen
- Link to relevant references
- "Learn more" content
- Interactive tutorials
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class TestExplanation:
    """Explanation for a statistical test."""
    test_name: str
    purpose: str
    when_to_use: str
    assumptions: List[str]
    interpretation: str
    effect_sizes: List[str]
    common_mistakes: List[str]
    alternatives: List[str]
    references: List[str]
    example: Optional[str] = None


@dataclass
class ConceptExplanation:
    """Explanation of a statistical concept."""
    concept: str
    definition: str
    intuition: str
    formula: Optional[str] = None
    example: Optional[str] = None
    common_misunderstandings: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


class StatisticalExplainer:
    """Provides explanations for statistical tests and concepts."""

    def __init__(self):
        self.tests = self._build_test_explanations()
        self.concepts = self._build_concept_explanations()

    def explain_test(self, test_name: str) -> Optional[TestExplanation]:
        """Get explanation for a statistical test."""
        test_key = test_name.lower().replace(' ', '_').replace('-', '_')

        # Try exact match
        if test_key in self.tests:
            return self.tests[test_key]

        # Try fuzzy match
        for key, explanation in self.tests.items():
            if test_key in key or key in test_key:
                return explanation

        return None

    def explain_concept(self, concept: str) -> Optional[ConceptExplanation]:
        """Get explanation for a statistical concept."""
        concept_key = concept.lower().replace(' ', '_').replace('-', '_')

        if concept_key in self.concepts:
            return self.concepts[concept_key]

        # Fuzzy match
        for key, explanation in self.concepts.items():
            if concept_key in key or key in concept_key:
                return explanation

        return None

    def get_test_selection_guidance(
        self,
        outcome_type: str,
        predictor_type: str,
        n_groups: int = 2,
        paired: bool = False,
        normal: bool = True
    ) -> Dict[str, Any]:
        """Get guidance on which test to use.

        Args:
            outcome_type: 'continuous', 'binary', 'categorical', 'time_to_event'
            predictor_type: 'continuous', 'binary', 'categorical'
            n_groups: Number of groups being compared
            paired: Whether observations are paired
            normal: Whether data is normally distributed

        Returns:
            Dict with recommended test and explanation
        """
        recommendation = {
            'primary_test': '',
            'alternative_test': '',
            'reasoning': '',
            'assumptions_to_check': [],
            'effect_size': ''
        }

        if outcome_type == 'continuous':
            if predictor_type in ['binary', 'categorical']:
                if n_groups == 2:
                    if paired:
                        if normal:
                            recommendation['primary_test'] = 'Paired t-test'
                            recommendation['alternative_test'] = 'Wilcoxon signed-rank test'
                            recommendation['reasoning'] = 'Comparing two related/matched continuous measurements'
                        else:
                            recommendation['primary_test'] = 'Wilcoxon signed-rank test'
                            recommendation['alternative_test'] = 'Paired t-test (if n large)'
                            recommendation['reasoning'] = 'Non-normal paired continuous data'
                    else:
                        if normal:
                            recommendation['primary_test'] = 'Independent samples t-test'
                            recommendation['alternative_test'] = 'Mann-Whitney U test'
                            recommendation['reasoning'] = 'Comparing two independent group means'
                        else:
                            recommendation['primary_test'] = 'Mann-Whitney U test'
                            recommendation['alternative_test'] = 'Welch t-test (robust to non-normality)'
                            recommendation['reasoning'] = 'Non-normal continuous data, two groups'
                    recommendation['effect_size'] = "Cohen's d"
                else:  # > 2 groups
                    if normal:
                        recommendation['primary_test'] = 'One-way ANOVA'
                        recommendation['alternative_test'] = 'Kruskal-Wallis test'
                        recommendation['reasoning'] = 'Comparing means across multiple groups'
                    else:
                        recommendation['primary_test'] = 'Kruskal-Wallis test'
                        recommendation['alternative_test'] = 'Welch ANOVA'
                        recommendation['reasoning'] = 'Non-normal data, multiple groups'
                    recommendation['effect_size'] = 'Eta-squared'

                recommendation['assumptions_to_check'] = [
                    'Normality (Shapiro-Wilk test)',
                    'Homogeneity of variances (Levene test)',
                    'Independence of observations'
                ]

            elif predictor_type == 'continuous':
                recommendation['primary_test'] = 'Pearson correlation / Linear regression'
                recommendation['alternative_test'] = 'Spearman correlation'
                recommendation['reasoning'] = 'Assessing linear relationship between two continuous variables'
                recommendation['effect_size'] = 'R-squared or r'
                recommendation['assumptions_to_check'] = [
                    'Linearity',
                    'Normality of residuals',
                    'Homoscedasticity',
                    'Independence'
                ]

        elif outcome_type == 'binary':
            if predictor_type in ['binary', 'categorical']:
                recommendation['primary_test'] = 'Chi-square test / Logistic regression'
                recommendation['alternative_test'] = "Fisher's exact test (small samples)"
                recommendation['reasoning'] = 'Comparing proportions between groups'
                recommendation['effect_size'] = 'Odds ratio'
                recommendation['assumptions_to_check'] = [
                    'Expected cell counts >= 5 (for chi-square)',
                    'Independence of observations'
                ]
            elif predictor_type == 'continuous':
                recommendation['primary_test'] = 'Logistic regression'
                recommendation['alternative_test'] = 'ROC analysis (for prediction)'
                recommendation['reasoning'] = 'Modeling binary outcome with continuous predictor'
                recommendation['effect_size'] = 'Odds ratio per unit change'
                recommendation['assumptions_to_check'] = [
                    'Linearity in the logit',
                    'No multicollinearity (if multiple predictors)',
                    'Large sample size'
                ]

        elif outcome_type == 'time_to_event':
            recommendation['primary_test'] = 'Kaplan-Meier + Log-rank test / Cox regression'
            recommendation['alternative_test'] = 'Parametric survival models'
            recommendation['reasoning'] = 'Analyzing time to event with censoring'
            recommendation['effect_size'] = 'Hazard ratio'
            recommendation['assumptions_to_check'] = [
                'Proportional hazards (for Cox)',
                'Non-informative censoring',
                'Independence'
            ]

        return recommendation

    def _build_test_explanations(self) -> Dict[str, TestExplanation]:
        """Build explanations for common tests."""
        tests = {}

        tests['independent_ttest'] = TestExplanation(
            test_name="Independent Samples t-test",
            purpose="Compare means of a continuous variable between two independent groups",
            when_to_use="When comparing two groups on a continuous outcome (e.g., treatment vs control blood pressure)",
            assumptions=[
                "Independent observations",
                "Continuous outcome variable",
                "Normal distribution in each group (or n > 30 for CLT)",
                "Homogeneity of variances (or use Welch's t-test)"
            ],
            interpretation="A significant p-value indicates the group means differ more than expected by chance. Report the mean difference and 95% CI alongside p-value.",
            effect_sizes=[
                "Cohen's d: 0.2 = small, 0.5 = medium, 0.8 = large",
                "Mean difference with 95% CI"
            ],
            common_mistakes=[
                "Using when data is clearly non-normal with small n",
                "Ignoring unequal variances",
                "Not reporting effect sizes",
                "Multiple testing without correction"
            ],
            alternatives=[
                "Mann-Whitney U test (non-parametric)",
                "Welch's t-test (unequal variances)",
                "Permutation test",
                "Bootstrap comparison"
            ],
            references=[
                "Student. The probable error of a mean. Biometrika 1908;6:1-25",
                "Welch BL. The generalization of Student's problem. Biometrika 1947;34:28-35"
            ],
            example="Comparing mean systolic blood pressure between drug (n=50, mean=130) and placebo (n=50, mean=138) groups."
        )

        tests['paired_ttest'] = TestExplanation(
            test_name="Paired t-test",
            purpose="Compare means of a continuous variable measured twice on the same subjects",
            when_to_use="Before-after designs, matched pairs, repeated measures on same subjects",
            assumptions=[
                "Paired/matched observations",
                "Differences are normally distributed",
                "Continuous outcome"
            ],
            interpretation="Tests whether the mean of the differences equals zero. Report mean difference and 95% CI.",
            effect_sizes=["Cohen's d (calculated from differences)", "Mean difference with CI"],
            common_mistakes=[
                "Using independent t-test for paired data (loses power)",
                "Not checking normality of differences"
            ],
            alternatives=["Wilcoxon signed-rank test", "Bootstrap paired comparison"],
            references=["Student. Biometrika 1908"]
        )

        tests['chi_square'] = TestExplanation(
            test_name="Chi-square Test of Independence",
            purpose="Test association between two categorical variables",
            when_to_use="Comparing proportions or testing independence in contingency tables",
            assumptions=[
                "Independent observations",
                "Expected cell counts >= 5 (use Fisher's exact if not)",
                "Categorical variables"
            ],
            interpretation="Significant result indicates variables are associated. Examine residuals or odds ratios for direction.",
            effect_sizes=[
                "Phi coefficient (2x2 tables)",
                "Cramér's V (larger tables)",
                "Odds ratio (2x2 tables)"
            ],
            common_mistakes=[
                "Using with small expected counts",
                "Not reporting effect size",
                "Confusing association with causation"
            ],
            alternatives=[
                "Fisher's exact test (small samples)",
                "G-test",
                "Logistic regression (adjusted analysis)"
            ],
            references=["Pearson K. On the criterion... Phil Mag 1900;50:157-175"]
        )

        tests['logistic_regression'] = TestExplanation(
            test_name="Logistic Regression",
            purpose="Model binary outcome as function of predictors",
            when_to_use="Binary outcome with continuous or categorical predictors; need to adjust for confounders",
            assumptions=[
                "Binary outcome",
                "Independence of observations",
                "Linear relationship between predictors and log-odds",
                "No multicollinearity",
                "Large sample size (10-20 events per predictor)"
            ],
            interpretation="Coefficients represent log-odds ratios. Exponentiate for odds ratios. OR > 1 = increased odds.",
            effect_sizes=[
                "Odds ratio with 95% CI",
                "Pseudo R-squared (McFadden, Nagelkerke)",
                "AUC for discrimination"
            ],
            common_mistakes=[
                "Too few events per predictor (overfitting)",
                "Interpreting OR as relative risk",
                "Not checking linearity assumption",
                "Including intermediate variables on causal pathway"
            ],
            alternatives=[
                "Poisson regression with robust SE (for common outcomes)",
                "Log-binomial regression (for risk ratios)",
                "Propensity score methods"
            ],
            references=[
                "Hosmer DW, Lemeshow S. Applied Logistic Regression. Wiley.",
                "Vittinghoff E. Regression Methods in Biostatistics."
            ]
        )

        tests['cox_regression'] = TestExplanation(
            test_name="Cox Proportional Hazards Regression",
            purpose="Model time-to-event data with censoring",
            when_to_use="Survival analysis, time-to-event outcomes with right censoring",
            assumptions=[
                "Proportional hazards (HR constant over time)",
                "Non-informative censoring",
                "Independence of observations",
                "Correct functional form for continuous predictors"
            ],
            interpretation="Hazard ratio (HR) represents relative instantaneous risk. HR > 1 = higher hazard/worse survival.",
            effect_sizes=["Hazard ratio with 95% CI", "C-statistic for discrimination"],
            common_mistakes=[
                "Ignoring proportional hazards violation",
                "Not censoring appropriately",
                "Including post-baseline variables",
                "Confusing HR with relative risk"
            ],
            alternatives=[
                "Kaplan-Meier + log-rank (unadjusted)",
                "RMST (if PH violated)",
                "Parametric models (Weibull, exponential)",
                "Competing risks models"
            ],
            references=[
                "Cox DR. Regression models and life-tables. JRSS B 1972;34:187-220",
                "Therneau TM, Grambsch PM. Modeling Survival Data."
            ]
        )

        return tests

    def _build_concept_explanations(self) -> Dict[str, ConceptExplanation]:
        """Build explanations for statistical concepts."""
        concepts = {}

        concepts['p_value'] = ConceptExplanation(
            concept="P-value",
            definition="The probability of observing data as extreme as or more extreme than what was observed, assuming the null hypothesis is true.",
            intuition="If the null hypothesis were true (no effect), how surprising is our result? Very low p-value = very surprising = evidence against the null.",
            formula="P(data | H0) - calculated depends on the test",
            example="A p-value of 0.03 means there's a 3% chance of seeing a difference this large if there truly were no difference.",
            common_misunderstandings=[
                "P-value is NOT the probability that the null is true",
                "P-value is NOT the probability the results are due to chance",
                "P < 0.05 does NOT mean the effect is clinically important",
                "P > 0.05 does NOT prove no effect exists"
            ],
            related_concepts=["Confidence interval", "Type I error", "Statistical significance", "Effect size"],
            references=["Wasserstein RL, Lazar NA. The ASA Statement on p-Values. Am Stat 2016;70:129-133"]
        )

        concepts['confidence_interval'] = ConceptExplanation(
            concept="Confidence Interval",
            definition="A range of plausible values for the true population parameter, calculated so that 95% of such intervals would contain the true value.",
            intuition="The 95% CI gives a range of values that are reasonably consistent with the data. Narrower CI = more precision.",
            formula="Estimate ± (critical value) × (standard error)",
            example="Mean difference = 5.2, 95% CI [2.1, 8.3] means we're confident the true difference is somewhere between 2.1 and 8.3.",
            common_misunderstandings=[
                "95% CI does NOT mean 95% probability the true value is in this interval",
                "The correct interpretation is about the procedure, not this specific interval",
                "CI depends on sample size - small samples give wide CIs"
            ],
            related_concepts=["Standard error", "P-value", "Precision"],
            references=["Morey RD et al. The fallacy of placing confidence in confidence intervals. Psychon Bull Rev 2016"]
        )

        concepts['effect_size'] = ConceptExplanation(
            concept="Effect Size",
            definition="A standardized measure of the magnitude of an effect, independent of sample size.",
            intuition="P-values tell you IF there's an effect; effect sizes tell you HOW BIG the effect is.",
            formula="Cohen's d = (M1 - M2) / SD_pooled; OR = (a/b)/(c/d) for 2x2 table",
            example="Cohen's d = 0.8 is a 'large' effect - the groups differ by 0.8 standard deviations.",
            common_misunderstandings=[
                "Large effect ≠ clinically important (depends on context)",
                "Small effect with narrow CI may still be meaningful"
            ],
            related_concepts=["Cohen's d", "Odds ratio", "Hazard ratio", "NNT"],
            references=["Cohen J. Statistical Power Analysis for the Behavioral Sciences. 1988"]
        )

        concepts['power'] = ConceptExplanation(
            concept="Statistical Power",
            definition="The probability of correctly rejecting the null hypothesis when it is false (i.e., detecting a true effect).",
            intuition="If there truly is an effect, power is your chance of finding it. Low power = high risk of missing real effects.",
            formula="Power = 1 - β, where β is Type II error rate. Depends on effect size, sample size, and alpha.",
            example="80% power means if the true effect is as assumed, you have an 80% chance of getting a significant result.",
            common_misunderstandings=[
                "Post-hoc power calculations are not meaningful",
                "Power increases with sample size and effect size",
                "Underpowered studies waste resources and can mislead"
            ],
            related_concepts=["Sample size", "Type II error", "Effect size"],
            references=["Cohen J. Statistical Power Analysis. 1988"]
        )

        concepts['confounding'] = ConceptExplanation(
            concept="Confounding",
            definition="When an extraneous variable is associated with both the exposure and the outcome, distorting the observed association.",
            intuition="A third variable creates a spurious association or hides a real one. Like how ice cream sales and drowning are both correlated with summer.",
            example="Coffee drinking appears associated with lung cancer, but this is confounded by smoking (smokers drink more coffee).",
            common_misunderstandings=[
                "Confounders must be associated with BOTH exposure AND outcome",
                "Variables on the causal pathway are not confounders",
                "You cannot adjust for unmeasured confounders"
            ],
            related_concepts=["Adjustment", "Propensity score", "Randomization", "Directed acyclic graph"],
            references=["Rothman KJ, Greenland S. Modern Epidemiology."]
        )

        return concepts


class EducationalContent:
    """Comprehensive educational content manager."""

    def __init__(self):
        self.explainer = StatisticalExplainer()
        self.tutorials = self._build_tutorials()

    def get_why_this_test(
        self,
        test_name: str,
        data_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Explain why a particular test was chosen.

        Args:
            test_name: Name of the test
            data_context: Optional context about the data

        Returns:
            Explanation string
        """
        explanation = self.explainer.explain_test(test_name)
        if not explanation:
            return f"No explanation available for {test_name}"

        lines = [
            f"## Why {explanation.test_name}?",
            "",
            f"**Purpose:** {explanation.purpose}",
            "",
            f"**When to use:** {explanation.when_to_use}",
            "",
            "**Key assumptions:**"
        ]
        for assumption in explanation.assumptions:
            lines.append(f"- {assumption}")

        lines.extend([
            "",
            f"**Interpretation:** {explanation.interpretation}",
            "",
            "**Effect sizes to report:**"
        ])
        for es in explanation.effect_sizes:
            lines.append(f"- {es}")

        if data_context:
            lines.extend([
                "",
                "**Based on your data:**"
            ])
            if data_context.get('outcome_type'):
                lines.append(f"- Outcome type: {data_context['outcome_type']}")
            if data_context.get('n_groups'):
                lines.append(f"- Number of groups: {data_context['n_groups']}")

        return '\n'.join(lines)

    def get_learn_more(self, topic: str) -> str:
        """Get comprehensive learning content on a topic."""
        # Try test explanation first
        test_exp = self.explainer.explain_test(topic)
        if test_exp:
            return self._format_test_content(test_exp)

        # Try concept explanation
        concept_exp = self.explainer.explain_concept(topic)
        if concept_exp:
            return self._format_concept_content(concept_exp)

        return f"No educational content available for '{topic}'. Try searching for: p-value, confidence interval, t-test, regression, etc."

    def _format_test_content(self, test: TestExplanation) -> str:
        """Format test explanation as learning content."""
        lines = [
            f"# {test.test_name}",
            "",
            "## Overview",
            test.purpose,
            "",
            "## When to Use",
            test.when_to_use,
            "",
            "## Assumptions",
        ]
        for a in test.assumptions:
            lines.append(f"1. {a}")

        lines.extend([
            "",
            "## Interpretation",
            test.interpretation,
            "",
            "## Effect Sizes",
        ])
        for es in test.effect_sizes:
            lines.append(f"- {es}")

        lines.extend([
            "",
            "## Common Mistakes to Avoid",
        ])
        for m in test.common_mistakes:
            lines.append(f"- {m}")

        lines.extend([
            "",
            "## Alternative Tests",
        ])
        for a in test.alternatives:
            lines.append(f"- {a}")

        if test.example:
            lines.extend([
                "",
                "## Example",
                test.example
            ])

        lines.extend([
            "",
            "## References",
        ])
        for r in test.references:
            lines.append(f"- {r}")

        return '\n'.join(lines)

    def _format_concept_content(self, concept: ConceptExplanation) -> str:
        """Format concept explanation as learning content."""
        lines = [
            f"# {concept.concept}",
            "",
            "## Definition",
            concept.definition,
            "",
            "## Intuitive Understanding",
            concept.intuition,
        ]

        if concept.formula:
            lines.extend([
                "",
                "## Formula",
                f"`{concept.formula}`"
            ])

        if concept.example:
            lines.extend([
                "",
                "## Example",
                concept.example
            ])

        if concept.common_misunderstandings:
            lines.extend([
                "",
                "## Common Misunderstandings",
            ])
            for m in concept.common_misunderstandings:
                lines.append(f"- {m}")

        if concept.related_concepts:
            lines.extend([
                "",
                "## Related Concepts",
                ", ".join(concept.related_concepts)
            ])

        lines.extend([
            "",
            "## References",
        ])
        for r in concept.references:
            lines.append(f"- {r}")

        return '\n'.join(lines)

    def _build_tutorials(self) -> Dict[str, str]:
        """Build interactive tutorials."""
        return {
            'choosing_test': 'Tutorial on choosing the right statistical test',
            'interpreting_regression': 'Tutorial on interpreting regression output',
            'power_analysis': 'Tutorial on conducting power analysis'
        }
