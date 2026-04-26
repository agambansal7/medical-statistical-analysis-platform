"""Natural Language Query Interface.

Enables natural language queries for statistical analysis:
- "Compare mortality between groups adjusting for age and sex"
- "Show me a forest plot of treatment effects by subgroup"
- "Is there effect modification by diabetes status?"
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class QueryIntent:
    """Parsed intent from natural language query."""
    action: str  # compare, show, test, calculate, etc.
    analysis_type: str  # regression, ttest, survival, visualization, etc.
    outcome: Optional[str] = None
    exposure: Optional[str] = None
    groups: Optional[List[str]] = None
    covariates: List[str] = field(default_factory=list)
    subgroups: Optional[str] = None
    visualization: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class QueryResult:
    """Result of natural language query."""
    query: str
    intent: QueryIntent
    analysis_result: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None
    explanation: str = ""
    suggestions: List[str] = field(default_factory=list)


class NaturalLanguageAnalysis:
    """Natural language interface for statistical analysis."""

    # Pattern matchers for different intents
    COMPARE_PATTERNS = [
        r'compare\s+(\w+)\s+between\s+(\w+)',
        r'difference\s+in\s+(\w+)\s+between\s+(\w+)',
        r'(\w+)\s+differences?\s+by\s+(\w+)',
        r'is\s+(\w+)\s+different\s+between\s+(\w+)',
    ]

    ASSOCIATION_PATTERNS = [
        r'association\s+between\s+(\w+)\s+and\s+(\w+)',
        r'relationship\s+between\s+(\w+)\s+and\s+(\w+)',
        r'correlation\s+between\s+(\w+)\s+and\s+(\w+)',
        r'does\s+(\w+)\s+affect\s+(\w+)',
        r'effect\s+of\s+(\w+)\s+on\s+(\w+)',
    ]

    VISUALIZATION_PATTERNS = [
        r'show\s+(?:me\s+)?(?:a\s+)?(\w+\s*\w*)\s+(?:plot|chart|graph|curve)',
        r'(?:plot|visualize|graph|draw)\s+(?:a\s+)?(\w+)',
        r'(\w+)\s+(?:plot|curve|chart)',
    ]

    SUBGROUP_PATTERNS = [
        r'effect\s+modification\s+by\s+(\w+)',
        r'interaction\s+(?:with|between|by)\s+(\w+)',
        r'subgroup\s+(?:by|analysis\s+(?:by|for))\s+(\w+)',
        r'stratif(?:y|ied)\s+by\s+(\w+)',
    ]

    ADJUSTMENT_PATTERNS = [
        r'adjust(?:ing|ed)?\s+for\s+(.+?)(?:\s+and\s+|$)',
        r'control(?:ling|led)?\s+for\s+(.+?)(?:\s+and\s+|$)',
        r'account(?:ing)?\s+for\s+(.+?)(?:\s+and\s+|$)',
    ]

    ANALYSIS_TYPE_KEYWORDS = {
        'regression': ['regression', 'adjust', 'control', 'predictor', 'model'],
        'ttest': ['compare', 't-test', 'ttest', 'difference', 'mean'],
        'anova': ['anova', 'groups', 'multiple groups'],
        'chi_square': ['chi-square', 'categorical', 'proportion', 'contingency'],
        'correlation': ['correlation', 'correlate', 'relationship', 'associated'],
        'survival': ['survival', 'kaplan', 'cox', 'hazard', 'time-to-event', 'mortality'],
        'logistic': ['odds', 'binary', 'logistic', 'risk'],
        'forest_plot': ['forest', 'subgroup', 'effect size'],
        'roc': ['roc', 'auc', 'diagnostic', 'sensitivity', 'specificity'],
    }

    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
        self.columns = list(data.columns) if data is not None else []
        self.column_types = self._infer_column_types() if data is not None else {}

    def query(self, question: str, data: Optional[pd.DataFrame] = None) -> QueryResult:
        """Process a natural language query.

        Args:
            question: Natural language question
            data: Optional DataFrame (uses stored data if not provided)

        Returns:
            QueryResult with analysis and/or visualization
        """
        if data is not None:
            self.data = data
            self.columns = list(data.columns)
            self.column_types = self._infer_column_types()

        # Parse intent
        intent = self._parse_intent(question)

        # Execute analysis
        result = None
        viz = None

        if intent.action == 'compare':
            result = self._execute_comparison(intent)
        elif intent.action == 'associate':
            result = self._execute_association(intent)
        elif intent.action == 'visualize':
            viz = self._execute_visualization(intent)
        elif intent.action == 'subgroup':
            result = self._execute_subgroup(intent)
        elif intent.action == 'test':
            result = self._execute_test(intent)

        # Generate explanation
        explanation = self._generate_explanation(intent, result, viz)

        # Generate follow-up suggestions
        suggestions = self._generate_suggestions(intent, result)

        return QueryResult(
            query=question,
            intent=intent,
            analysis_result=result,
            visualization=viz,
            explanation=explanation,
            suggestions=suggestions
        )

    def _parse_intent(self, question: str) -> QueryIntent:
        """Parse natural language question into structured intent."""
        question_lower = question.lower()

        intent = QueryIntent(
            action='unknown',
            analysis_type='unknown'
        )

        # Check for visualization request
        for pattern in self.VISUALIZATION_PATTERNS:
            match = re.search(pattern, question_lower)
            if match:
                intent.action = 'visualize'
                viz_type = match.group(1)
                intent.visualization = self._standardize_visualization(viz_type)
                intent.confidence = 0.8
                break

        # Check for comparison
        if intent.action == 'unknown':
            for pattern in self.COMPARE_PATTERNS:
                match = re.search(pattern, question_lower)
                if match:
                    intent.action = 'compare'
                    intent.outcome = self._find_column(match.group(1))
                    intent.exposure = self._find_column(match.group(2))
                    intent.confidence = 0.7
                    break

        # Check for association
        if intent.action == 'unknown':
            for pattern in self.ASSOCIATION_PATTERNS:
                match = re.search(pattern, question_lower)
                if match:
                    intent.action = 'associate'
                    var1 = self._find_column(match.group(1))
                    var2 = self._find_column(match.group(2))
                    # Determine which is outcome vs exposure based on context
                    if 'effect of' in question_lower or 'affect' in question_lower:
                        intent.exposure = var1
                        intent.outcome = var2
                    else:
                        intent.outcome = var2
                        intent.exposure = var1
                    intent.confidence = 0.7
                    break

        # Check for subgroup/interaction
        for pattern in self.SUBGROUP_PATTERNS:
            match = re.search(pattern, question_lower)
            if match:
                intent.action = 'subgroup'
                intent.subgroups = self._find_column(match.group(1))
                intent.confidence = 0.75
                break

        # Extract adjustment covariates
        for pattern in self.ADJUSTMENT_PATTERNS:
            match = re.search(pattern, question_lower)
            if match:
                covariate_text = match.group(1)
                # Split by common separators
                covariates = re.split(r',\s*|\s+and\s+', covariate_text)
                intent.covariates = [
                    self._find_column(c.strip())
                    for c in covariates
                    if self._find_column(c.strip())
                ]

        # Determine analysis type
        intent.analysis_type = self._determine_analysis_type(question_lower, intent)

        return intent

    def _find_column(self, term: str) -> Optional[str]:
        """Find matching column name in data."""
        if not self.columns:
            return term

        term_lower = term.lower().replace('_', '').replace(' ', '')

        # Exact match
        for col in self.columns:
            if col.lower() == term_lower:
                return col

        # Fuzzy match
        for col in self.columns:
            col_clean = col.lower().replace('_', '').replace(' ', '')
            if term_lower in col_clean or col_clean in term_lower:
                return col

        # Return original term if no match
        return term

    def _standardize_visualization(self, viz_type: str) -> str:
        """Standardize visualization type name."""
        viz_map = {
            'forest': 'forest_plot',
            'kaplan': 'kaplan_meier',
            'km': 'kaplan_meier',
            'survival': 'kaplan_meier',
            'roc': 'roc_curve',
            'box': 'boxplot',
            'scatter': 'scatter_plot',
            'histogram': 'histogram',
            'bar': 'bar_plot',
            'funnel': 'funnel_plot',
        }

        for key, value in viz_map.items():
            if key in viz_type.lower():
                return value

        return viz_type.replace(' ', '_')

    def _determine_analysis_type(self, question: str, intent: QueryIntent) -> str:
        """Determine appropriate analysis type."""
        # Check keywords
        for analysis, keywords in self.ANALYSIS_TYPE_KEYWORDS.items():
            if any(kw in question for kw in keywords):
                return analysis

        # Infer from variable types
        if intent.outcome and intent.exposure and self.data is not None:
            outcome_type = self.column_types.get(intent.outcome, 'unknown')
            exposure_type = self.column_types.get(intent.exposure, 'unknown')

            if outcome_type == 'binary' and exposure_type in ['binary', 'categorical']:
                return 'logistic' if intent.covariates else 'chi_square'
            elif outcome_type == 'continuous' and exposure_type == 'binary':
                return 'regression' if intent.covariates else 'ttest'
            elif outcome_type == 'continuous' and exposure_type == 'categorical':
                return 'regression' if intent.covariates else 'anova'
            elif outcome_type == 'continuous' and exposure_type == 'continuous':
                return 'regression' if intent.covariates else 'correlation'

        return 'unknown'

    def _infer_column_types(self) -> Dict[str, str]:
        """Infer variable types for all columns."""
        types = {}
        for col in self.columns:
            n_unique = self.data[col].nunique()
            dtype = str(self.data[col].dtype)

            if n_unique <= 2:
                types[col] = 'binary'
            elif n_unique <= 10 and dtype not in ['float64', 'float32']:
                types[col] = 'categorical'
            elif dtype in ['float64', 'float32', 'int64', 'int32']:
                types[col] = 'continuous'
            else:
                types[col] = 'categorical'

        return types

    def _execute_comparison(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Execute comparison analysis."""
        if self.data is None or not intent.outcome or not intent.exposure:
            return None

        # Import analysis modules
        import sys
        from pathlib import Path
        stats_path = Path(__file__).parent.parent / "stats"
        sys.path.insert(0, str(stats_path))

        try:
            if intent.analysis_type == 'ttest':
                from stats.comparative import ComparativeTests
                comp = ComparativeTests()
                groups = self.data[intent.exposure].unique()
                if len(groups) == 2:
                    g1 = self.data[self.data[intent.exposure] == groups[0]][intent.outcome]
                    g2 = self.data[self.data[intent.exposure] == groups[1]][intent.outcome]
                    result = comp.independent_ttest(g1, g2)
                    return result.to_dict()

            elif intent.analysis_type in ['regression', 'logistic']:
                from stats.regression import RegressionAnalysis
                reg = RegressionAnalysis()
                predictors = [intent.exposure] + intent.covariates

                if intent.analysis_type == 'logistic':
                    result = reg.logistic_regression(self.data, intent.outcome, predictors)
                else:
                    result = reg.linear_regression(self.data, intent.outcome, predictors)
                return result.to_dict()

        except Exception as e:
            return {'error': str(e)}

        return None

    def _execute_association(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Execute association analysis."""
        if self.data is None or not intent.outcome or not intent.exposure:
            return None

        try:
            if intent.analysis_type == 'correlation':
                from stats.correlation import CorrelationAnalysis
                corr = CorrelationAnalysis()
                result = corr.pearson_correlation(
                    self.data[intent.exposure],
                    self.data[intent.outcome]
                )
                return result.to_dict()

            # Default to regression
            return self._execute_comparison(intent)

        except Exception as e:
            return {'error': str(e)}

    def _execute_visualization(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Execute visualization request."""
        return {
            'type': intent.visualization,
            'status': 'ready',
            'parameters': {
                'outcome': intent.outcome,
                'exposure': intent.exposure,
                'subgroups': intent.subgroups
            }
        }

    def _execute_subgroup(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Execute subgroup/interaction analysis."""
        if self.data is None or not intent.subgroups:
            return None

        try:
            from stats.subgroup_analysis import SubgroupAnalyzer
            analyzer = SubgroupAnalyzer()

            if intent.outcome and intent.exposure:
                result = analyzer.analyze_subgroups(
                    self.data,
                    intent.outcome,
                    intent.exposure,
                    intent.subgroups,
                    covariates=intent.covariates
                )
                return result.to_dict()

        except Exception as e:
            return {'error': str(e)}

        return None

    def _execute_test(self, intent: QueryIntent) -> Optional[Dict[str, Any]]:
        """Execute statistical test."""
        return self._execute_comparison(intent)

    def _generate_explanation(
        self,
        intent: QueryIntent,
        result: Optional[Dict[str, Any]],
        viz: Optional[Dict[str, Any]]
    ) -> str:
        """Generate natural language explanation of results."""
        lines = []

        if intent.action == 'compare':
            lines.append(f"Comparing {intent.outcome} between {intent.exposure} groups")
            if intent.covariates:
                lines.append(f"Adjusting for: {', '.join(intent.covariates)}")

        if result:
            if 'p_value' in result:
                p = result['p_value']
                sig = "statistically significant" if p < 0.05 else "not statistically significant"
                lines.append(f"Result: {sig} (p = {p:.4f})")

            if 'odds_ratio' in result:
                or_val = result['odds_ratio']
                direction = "increased" if or_val > 1 else "decreased"
                lines.append(f"Odds ratio: {or_val:.2f} ({direction} odds)")

            if 'coefficient' in result:
                coef = result['coefficient']
                direction = "positive" if coef > 0 else "negative"
                lines.append(f"Effect: {coef:.3f} ({direction} association)")

        if viz:
            lines.append(f"Visualization: {viz.get('type', 'plot')} is ready")

        return '\n'.join(lines) if lines else "Analysis completed."

    def _generate_suggestions(
        self,
        intent: QueryIntent,
        result: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate follow-up analysis suggestions."""
        suggestions = []

        if intent.action == 'compare':
            if not intent.covariates:
                suggestions.append(f"Consider adjusting for potential confounders")
            suggestions.append(f"Check assumptions for the statistical test used")
            suggestions.append(f"Visualize the distribution with a boxplot")

        if intent.action == 'associate':
            suggestions.append("Test for non-linear relationships")
            suggestions.append("Check for influential observations")

        if result and 'p_value' in result and result['p_value'] < 0.05:
            suggestions.append("Calculate effect sizes for clinical interpretation")
            suggestions.append("Perform sensitivity analysis")

        return suggestions

    def suggest_queries(self) -> List[str]:
        """Suggest example queries based on available data."""
        suggestions = []

        if self.data is None:
            return ["Load data first to get query suggestions"]

        # Find potential outcome variables
        outcomes = [c for c, t in self.column_types.items() if t in ['binary', 'continuous']]
        exposures = [c for c, t in self.column_types.items() if t in ['binary', 'categorical']]

        if outcomes and exposures:
            suggestions.append(f"Compare {outcomes[0]} between {exposures[0]} groups")
            suggestions.append(f"What is the association between {exposures[0]} and {outcomes[0]}?")

        continuous = [c for c, t in self.column_types.items() if t == 'continuous']
        if len(continuous) >= 2:
            suggestions.append(f"Is there a correlation between {continuous[0]} and {continuous[1]}?")

        if outcomes:
            suggestions.append(f"Show me a histogram of {outcomes[0]}")

        return suggestions
