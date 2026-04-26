"""LLM Orchestrator for intelligent statistical analysis."""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import anthropic

# Handle imports from different contexts
try:
    from ..utils.config import Config
except ImportError:
    try:
        from utils.config import Config
    except ImportError:
        # Fallback: create minimal Config class
        class Config:
            def __init__(self):
                self.api_key = os.environ.get('ANTHROPIC_API_KEY')
                self.llm_model = 'claude-sonnet-4-20250514'


@dataclass
class AnalysisRecommendation:
    """Recommended statistical analysis."""
    test_name: str
    category: str
    priority: int  # 1 = primary, 2 = secondary, 3 = exploratory
    rationale: str
    assumptions: List[str]
    variables: Dict[str, str]  # role -> variable name
    parameters: Dict[str, Any]


@dataclass
class AnalysisPlan:
    """Complete analysis plan."""
    research_question: str
    research_type: str  # 'comparison', 'association', 'prediction', 'diagnostic', 'survival'
    primary_analyses: List[AnalysisRecommendation]
    secondary_analyses: List[AnalysisRecommendation]
    assumption_checks: List[str]
    visualizations: List[str]
    interpretation_notes: List[str]


class LLMOrchestrator:
    """Orchestrate statistical analysis using LLM."""

    def __init__(self, api_key: str = None):
        config = Config()
        self.api_key = api_key or config.api_key
        self.model = config.llm_model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze_research_question(self,
                                  question: str,
                                  data_profile: Dict[str, Any]) -> AnalysisPlan:
        """Analyze research question and recommend analyses.

        Args:
            question: Research question from user
            data_profile: Data profile from DataProfiler

        Returns:
            AnalysisPlan object
        """
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_analysis_prompt(question, data_profile)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return self._parse_analysis_plan(response.content[0].text, question)

    def _get_system_prompt(self) -> str:
        """Get system prompt for statistical analysis."""
        return """You are an expert biostatistician helping researchers analyze medical data.
Your role is to:

1. Understand the research question
2. Examine the data structure
3. Recommend appropriate statistical analyses
4. Explain assumptions that need to be checked
5. Suggest visualizations

IMPORTANT GUIDELINES:
- Always recommend checking assumptions before parametric tests
- Suggest both parametric and non-parametric alternatives when appropriate
- Consider multiple testing corrections for multiple comparisons
- Be conservative in recommendations - clinical research requires rigor
- Consider sample size limitations
- Recommend effect sizes alongside significance tests

When analyzing a research question, classify it as one of:
- COMPARISON: Comparing groups (e.g., treatment vs control)
- ASSOCIATION: Examining relationships between variables
- PREDICTION: Building predictive models
- DIAGNOSTIC: Evaluating diagnostic test performance
- SURVIVAL: Time-to-event analysis

Respond in JSON format with the following structure:
{
    "research_type": "COMPARISON|ASSOCIATION|PREDICTION|DIAGNOSTIC|SURVIVAL",
    "primary_analyses": [
        {
            "test_name": "Name of statistical test",
            "category": "descriptive|comparative|correlation|regression|survival|diagnostic",
            "priority": 1,
            "rationale": "Why this test is appropriate",
            "assumptions": ["assumption1", "assumption2"],
            "variables": {"outcome": "var_name", "group": "var_name"},
            "parameters": {}
        }
    ],
    "secondary_analyses": [...],
    "assumption_checks": ["normality", "homogeneity_of_variance"],
    "visualizations": ["boxplot", "histogram"],
    "interpretation_notes": ["Note about interpreting results"]
}"""

    def _build_analysis_prompt(self, question: str, data_profile: Dict) -> str:
        """Build the analysis request prompt."""

        # Extract key information from profile
        n_rows = data_profile.get('n_rows', 0)
        n_continuous = data_profile.get('n_continuous', 0)
        n_categorical = data_profile.get('n_categorical', 0)
        n_binary = data_profile.get('n_binary', 0)

        variables_info = []
        for var in data_profile.get('variables', []):
            var_desc = f"- {var['name']}: {var['statistical_type']}"
            if var['statistical_type'] == 'continuous':
                var_desc += f" (mean={var.get('mean', 'N/A'):.2f}, SD={var.get('std', 'N/A'):.2f})"
            elif var['statistical_type'] in ['categorical', 'binary']:
                cats = var.get('categories', [])
                if cats:
                    var_desc += f" (categories: {', '.join(cats[:5])})"
            variables_info.append(var_desc)

        potential_outcomes = data_profile.get('potential_outcome_columns', [])
        potential_groups = data_profile.get('potential_group_columns', [])
        warnings = data_profile.get('warnings', [])

        prompt = f"""RESEARCH QUESTION:
{question}

DATASET OVERVIEW:
- Total observations: {n_rows}
- Continuous variables: {n_continuous}
- Categorical variables: {n_categorical}
- Binary variables: {n_binary}

VARIABLES:
{chr(10).join(variables_info)}

DETECTED POTENTIAL ROLES:
- Likely outcome variables: {', '.join(potential_outcomes) or 'None detected'}
- Likely grouping variables: {', '.join(potential_groups) or 'None detected'}

DATA QUALITY WARNINGS:
{chr(10).join(['- ' + w for w in warnings]) or 'None'}

Based on this research question and data structure, recommend the most appropriate statistical analyses.
Consider the sample size, variable types, and research objectives.
Provide your recommendations in the JSON format specified."""

        return prompt

    def _parse_analysis_plan(self, response: str, question: str) -> AnalysisPlan:
        """Parse LLM response into AnalysisPlan."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = {}

            primary = [
                AnalysisRecommendation(
                    test_name=a.get('test_name', ''),
                    category=a.get('category', ''),
                    priority=a.get('priority', 1),
                    rationale=a.get('rationale', ''),
                    assumptions=a.get('assumptions', []),
                    variables=a.get('variables', {}),
                    parameters=a.get('parameters', {})
                )
                for a in data.get('primary_analyses', [])
            ]

            secondary = [
                AnalysisRecommendation(
                    test_name=a.get('test_name', ''),
                    category=a.get('category', ''),
                    priority=a.get('priority', 2),
                    rationale=a.get('rationale', ''),
                    assumptions=a.get('assumptions', []),
                    variables=a.get('variables', {}),
                    parameters=a.get('parameters', {})
                )
                for a in data.get('secondary_analyses', [])
            ]

            return AnalysisPlan(
                research_question=question,
                research_type=data.get('research_type', 'unknown'),
                primary_analyses=primary,
                secondary_analyses=secondary,
                assumption_checks=data.get('assumption_checks', []),
                visualizations=data.get('visualizations', []),
                interpretation_notes=data.get('interpretation_notes', [])
            )

        except Exception as e:
            # Return minimal plan on parse error
            return AnalysisPlan(
                research_question=question,
                research_type='unknown',
                primary_analyses=[],
                secondary_analyses=[],
                assumption_checks=[],
                visualizations=[],
                interpretation_notes=[f'Error parsing LLM response: {str(e)}']
            )

    def interpret_results(self,
                         analysis_name: str,
                         results: Dict[str, Any],
                         context: Dict[str, Any] = None) -> str:
        """Generate detailed interpretation of analysis results.

        Args:
            analysis_name: Name of the analysis performed
            results: Results dictionary from the analysis
            context: Additional context (sample sizes, variable names, etc.)

        Returns:
            Detailed interpretation text
        """
        system_prompt = """You are an expert biostatistician writing results interpretations for medical research.

Your interpretations should:
1. Explain what the test measures
2. Report the key statistics
3. State whether the result is statistically significant
4. Interpret the effect size in practical terms
5. Note any limitations or caveats
6. Be written in clear, accessible language

Use APA-style statistical reporting (e.g., "t(28) = 2.45, p = .021, d = 0.89").
Always mention clinical significance, not just statistical significance."""

        user_prompt = f"""Please interpret these statistical results:

ANALYSIS: {analysis_name}

RESULTS:
{json.dumps(results, indent=2, default=str)}

CONTEXT:
{json.dumps(context or {}, indent=2, default=str)}

Provide a clear, detailed interpretation suitable for a research paper's results section."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    def suggest_next_steps(self,
                          completed_analyses: List[str],
                          results_summary: Dict[str, Any],
                          research_question: str) -> List[str]:
        """Suggest follow-up analyses based on results.

        Args:
            completed_analyses: List of completed analysis names
            results_summary: Summary of results
            research_question: Original research question

        Returns:
            List of suggested next steps
        """
        system_prompt = """You are an expert biostatistician advising on follow-up analyses.

Based on the completed analyses and results, suggest:
1. Additional analyses that might be informative
2. Sensitivity analyses to check robustness
3. Subgroup analyses if appropriate
4. Visualizations to better communicate findings

Be specific and practical in your recommendations."""

        user_prompt = f"""RESEARCH QUESTION: {research_question}

COMPLETED ANALYSES:
{', '.join(completed_analyses)}

RESULTS SUMMARY:
{json.dumps(results_summary, indent=2, default=str)}

What follow-up analyses would you recommend? Provide a numbered list."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        # Parse numbered list from response
        text = response.content[0].text
        suggestions = []
        for line in text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering/bullets
                clean = line.lstrip('0123456789.-) ').strip()
                if clean:
                    suggestions.append(clean)

        return suggestions

    def generate_methods_section(self,
                                analyses_performed: List[Dict],
                                data_description: Dict) -> str:
        """Generate methods section for publication.

        Args:
            analyses_performed: List of analyses with parameters
            data_description: Description of the dataset

        Returns:
            Methods section text
        """
        system_prompt = """You are writing the statistical methods section of a medical research paper.

Follow these guidelines:
1. Use past tense
2. Be specific about tests used and why
3. Mention software/packages used
4. Include significance threshold
5. Describe how assumptions were checked
6. Mention any corrections for multiple testing
7. Follow STROBE/CONSORT guidelines as appropriate

Write in formal academic style appropriate for peer-reviewed journals."""

        user_prompt = f"""Write the statistical methods section based on:

DATA:
{json.dumps(data_description, indent=2, default=str)}

ANALYSES PERFORMED:
{json.dumps(analyses_performed, indent=2, default=str)}

Include all relevant methodological details."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    def answer_statistical_question(self, question: str, context: Dict = None) -> str:
        """Answer a statistical question from the user.

        Args:
            question: User's question
            context: Current analysis context

        Returns:
            Answer text
        """
        system_prompt = """You are an expert biostatistician answering questions about statistical analysis.

Be helpful, educational, and precise. If the question is about which test to use,
explain your reasoning. If it's about interpretation, be clear about what the
numbers mean in practical terms.

Always consider the medical/clinical context when relevant."""

        user_prompt = f"""Question: {question}

Context:
{json.dumps(context or {}, indent=2, default=str)}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    def validate_analysis_plan(self,
                              plan: AnalysisPlan,
                              data_profile: Dict) -> Tuple[bool, List[str]]:
        """Validate that the analysis plan is feasible with the data.

        Args:
            plan: Proposed analysis plan
            data_profile: Data profile

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        available_vars = [v['name'] for v in data_profile.get('variables', [])]

        for analysis in plan.primary_analyses + plan.secondary_analyses:
            for role, var_name in analysis.variables.items():
                if var_name not in available_vars:
                    issues.append(f"{analysis.test_name}: Variable '{var_name}' not found in data")

        # Check sample size adequacy
        n_rows = data_profile.get('n_rows', 0)
        if n_rows < 30:
            issues.append("Small sample size (n < 30) - consider non-parametric methods")

        # Check for high missing data
        for var in data_profile.get('variables', []):
            if var.get('missing_pct', 0) > 20:
                issues.append(f"High missing data in '{var['name']}' ({var['missing_pct']}%)")

        is_valid = len([i for i in issues if 'not found' in i]) == 0
        return is_valid, issues
