"""
Comprehensive Statistical Plan Generator
=========================================

This module generates detailed, publication-ready statistical analysis plans
based on the research question and data profile. The plan includes:

1. Research question classification
2. Primary and secondary analyses with full specifications
3. Assumption checks required
4. Sensitivity analyses
5. Subgroup analyses (if applicable)
6. Visualizations
7. Expected results format
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import anthropic


@dataclass
class AnalysisStep:
    """A single analysis step in the plan."""
    step_id: str
    name: str
    category: str  # descriptive, comparative, correlation, regression, survival, diagnostic, advanced
    method: str    # Specific statistical method
    description: str
    rationale: str
    variables: Dict[str, str]  # role -> variable name
    parameters: Dict[str, Any]
    assumptions: List[str]
    assumption_tests: List[Dict[str, str]]  # Tests to check assumptions
    interpretation_guidance: str
    expected_output: List[str]
    fallback_method: Optional[str] = None  # If assumptions not met
    priority: str = "primary"  # primary, secondary, sensitivity, exploratory
    depends_on: List[str] = field(default_factory=list)  # Step IDs this depends on


@dataclass
class VisualizationStep:
    """A visualization in the plan."""
    viz_id: str
    name: str
    plot_type: str
    variables: Dict[str, str]
    parameters: Dict[str, Any]
    purpose: str
    related_analysis: Optional[str] = None


@dataclass
class ComprehensiveStatisticalPlan:
    """Complete statistical analysis plan."""
    plan_id: str
    created_at: str
    research_question: str
    research_type: str
    study_design: str

    # Data summary
    sample_size: int
    outcome_variables: List[str]
    exposure_variables: List[str]
    covariates: List[str]

    # Analysis plan sections
    descriptive_analyses: List[AnalysisStep]
    primary_analyses: List[AnalysisStep]
    secondary_analyses: List[AnalysisStep]
    sensitivity_analyses: List[AnalysisStep]
    subgroup_analyses: List[AnalysisStep]
    assumption_checks: List[AnalysisStep]

    # Visualizations
    visualizations: List[VisualizationStep]

    # Methodology notes
    missing_data_strategy: str
    multiple_testing_correction: str
    significance_level: float

    # Interpretation
    clinical_significance_thresholds: Dict[str, Any]
    limitations: List[str]

    # Status
    status: str = "draft"  # draft, confirmed, executing, completed
    user_modifications: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Generate readable markdown version of the plan."""
        md = []
        md.append(f"# Statistical Analysis Plan")
        md.append(f"**Plan ID:** {self.plan_id}")
        md.append(f"**Created:** {self.created_at}")
        md.append(f"**Status:** {self.status.upper()}")
        md.append("")

        md.append("## 1. Research Question")
        md.append(f"> {self.research_question}")
        md.append("")
        md.append(f"**Research Type:** {self.research_type}")
        md.append(f"**Study Design:** {self.study_design}")
        md.append("")

        md.append("## 2. Data Overview")
        md.append(f"- **Sample Size:** {self.sample_size}")
        md.append(f"- **Outcome Variables:** {', '.join(self.outcome_variables)}")
        md.append(f"- **Exposure/Treatment Variables:** {', '.join(self.exposure_variables)}")
        md.append(f"- **Covariates:** {', '.join(self.covariates)}")
        md.append("")

        md.append("## 3. Analysis Plan")
        md.append("")

        # Descriptive
        if self.descriptive_analyses:
            md.append("### 3.1 Descriptive Statistics")
            for i, step in enumerate(self.descriptive_analyses, 1):
                md.append(f"#### {i}. {step.name}")
                md.append(f"- **Method:** {step.method}")
                md.append(f"- **Variables:** {', '.join(f'{k}={v}' for k, v in step.variables.items())}")
                md.append(f"- **Purpose:** {step.description}")
                md.append("")

        # Primary
        if self.primary_analyses:
            md.append("### 3.2 Primary Analyses")
            for i, step in enumerate(self.primary_analyses, 1):
                md.append(f"#### {i}. {step.name}")
                md.append(f"- **Method:** {step.method}")
                md.append(f"- **Rationale:** {step.rationale}")
                md.append(f"- **Variables:** {', '.join(f'{k}={v}' for k, v in step.variables.items())}")
                if step.assumptions:
                    md.append(f"- **Assumptions:** {', '.join(step.assumptions)}")
                if step.fallback_method:
                    md.append(f"- **Fallback (if assumptions violated):** {step.fallback_method}")
                md.append(f"- **Interpretation:** {step.interpretation_guidance}")
                md.append("")

        # Secondary
        if self.secondary_analyses:
            md.append("### 3.3 Secondary Analyses")
            for i, step in enumerate(self.secondary_analyses, 1):
                md.append(f"#### {i}. {step.name}")
                md.append(f"- **Method:** {step.method}")
                md.append(f"- **Purpose:** {step.description}")
                md.append("")

        # Sensitivity
        if self.sensitivity_analyses:
            md.append("### 3.4 Sensitivity Analyses")
            for i, step in enumerate(self.sensitivity_analyses, 1):
                md.append(f"#### {i}. {step.name}")
                md.append(f"- **Method:** {step.method}")
                md.append(f"- **Purpose:** {step.description}")
                md.append("")

        # Subgroup
        if self.subgroup_analyses:
            md.append("### 3.5 Subgroup Analyses")
            for i, step in enumerate(self.subgroup_analyses, 1):
                md.append(f"#### {i}. {step.name}")
                md.append(f"- **Method:** {step.method}")
                md.append(f"- **Subgroup Variable:** {step.variables.get('subgroup', 'N/A')}")
                md.append("")

        # Visualizations
        if self.visualizations:
            md.append("### 3.6 Visualizations")
            for i, viz in enumerate(self.visualizations, 1):
                md.append(f"{i}. **{viz.name}** ({viz.plot_type}) - {viz.purpose}")
            md.append("")

        md.append("## 4. Statistical Methodology")
        md.append(f"- **Missing Data Strategy:** {self.missing_data_strategy}")
        md.append(f"- **Multiple Testing Correction:** {self.multiple_testing_correction}")
        md.append(f"- **Significance Level:** α = {self.significance_level}")
        md.append("")

        if self.limitations:
            md.append("## 5. Limitations")
            for lim in self.limitations:
                md.append(f"- {lim}")
            md.append("")

        return "\n".join(md)


class ComprehensiveStatisticalPlanner:
    """
    Generates comprehensive, detailed statistical analysis plans using LLM.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"

    def generate_plan(
        self,
        research_question: str,
        data_profile: Dict[str, Any],
        user_preferences: Optional[Dict] = None
    ) -> ComprehensiveStatisticalPlan:
        """
        Generate a comprehensive statistical analysis plan.

        Parameters:
        -----------
        research_question : str
            The research question to address
        data_profile : Dict
            Data profile from DataProfiler
        user_preferences : Dict, optional
            User preferences (significance level, preferred methods, etc.)

        Returns:
        --------
        ComprehensiveStatisticalPlan
        """
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_prompt(research_question, data_profile, user_preferences)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        plan = self._parse_response(response.content[0].text, research_question, data_profile)
        return plan

    def _get_system_prompt(self) -> str:
        return """You are an expert biostatistician creating comprehensive statistical analysis plans for medical research.

Your task is to create a DETAILED, PUBLICATION-READY statistical analysis plan that includes:

1. RESEARCH CLASSIFICATION
   - Identify the type of research question (comparison, association, prediction, survival, diagnostic)
   - Identify the study design (RCT, cohort, case-control, cross-sectional)

2. VARIABLE IDENTIFICATION
   - Primary outcome variable(s)
   - Primary exposure/treatment variable(s)
   - Confounders and covariates
   - Effect modifiers

3. DESCRIPTIVE ANALYSES
   - Baseline characteristics (Table 1)
   - Distribution assessments
   - Missing data summary

4. PRIMARY ANALYSES (with full specifications)
   - Exact statistical test/model
   - All variables and their roles
   - Parameters to estimate
   - Assumptions to check
   - Fallback methods if assumptions violated
   - How to interpret results

5. SECONDARY ANALYSES
   - Additional outcomes
   - Alternative model specifications

6. SENSITIVITY ANALYSES
   - Robustness checks
   - Different analytical approaches
   - Missing data sensitivity

7. SUBGROUP ANALYSES (if appropriate)
   - Pre-specified subgroups
   - Interaction tests

8. VISUALIZATIONS
   - Specific plots needed
   - Purpose of each

9. METHODOLOGY
   - Missing data handling
   - Multiple testing correction
   - Significance level

Return your plan as a JSON object with this exact structure:
{
    "research_type": "comparison|association|prediction|survival|diagnostic",
    "study_design": "RCT|cohort|case_control|cross_sectional",
    "outcome_variables": ["var1", "var2"],
    "exposure_variables": ["var1"],
    "covariates": ["var1", "var2"],
    "descriptive_analyses": [
        {
            "step_id": "desc_1",
            "name": "Baseline Characteristics Table",
            "category": "descriptive",
            "method": "table1",
            "description": "Summary of baseline characteristics by treatment group",
            "rationale": "Standard presentation of study population",
            "variables": {"group": "treatment", "continuous": ["age", "bmi"], "categorical": ["sex", "diabetes"]},
            "parameters": {},
            "assumptions": [],
            "assumption_tests": [],
            "interpretation_guidance": "Report means (SD) or medians (IQR) for continuous, n (%) for categorical",
            "expected_output": ["table"],
            "priority": "primary"
        }
    ],
    "primary_analyses": [
        {
            "step_id": "primary_1",
            "name": "Treatment Effect on Primary Outcome",
            "category": "comparative",
            "method": "independent_ttest",
            "description": "Compare mean outcome between treatment groups",
            "rationale": "Primary outcome is continuous, comparing two independent groups",
            "variables": {"outcome": "outcome_var", "group": "treatment"},
            "parameters": {"equal_var": true, "alpha": 0.05},
            "assumptions": ["normality", "homogeneity_of_variance", "independence"],
            "assumption_tests": [
                {"test": "shapiro_wilk", "for": "normality"},
                {"test": "levene", "for": "homogeneity_of_variance"}
            ],
            "interpretation_guidance": "Report mean difference, 95% CI, p-value, and Cohen's d",
            "expected_output": ["mean_difference", "ci_95", "p_value", "cohens_d"],
            "fallback_method": "mann_whitney_u",
            "priority": "primary",
            "depends_on": []
        }
    ],
    "secondary_analyses": [...],
    "sensitivity_analyses": [...],
    "subgroup_analyses": [...],
    "assumption_checks": [...],
    "visualizations": [
        {
            "viz_id": "viz_1",
            "name": "Outcome Distribution by Group",
            "plot_type": "boxplot",
            "variables": {"x": "treatment", "y": "outcome"},
            "parameters": {},
            "purpose": "Visual comparison of outcome distributions",
            "related_analysis": "primary_1"
        }
    ],
    "missing_data_strategy": "Complete case analysis with sensitivity using multiple imputation",
    "multiple_testing_correction": "Bonferroni for primary outcomes, FDR for exploratory",
    "significance_level": 0.05,
    "clinical_significance_thresholds": {"mean_difference": 5, "odds_ratio": 1.5},
    "limitations": ["Observational design limits causal inference", "Single-center study"]
}

Be SPECIFIC and DETAILED. Include exact variable names from the data profile.
Consider sample size limitations and data quality issues.
Recommend appropriate methods for the specific data characteristics."""

    def _build_prompt(
        self,
        research_question: str,
        data_profile: Dict,
        user_preferences: Optional[Dict]
    ) -> str:
        """Build the analysis request prompt."""

        # Extract profile information
        n_rows = data_profile.get('n_rows', 0)

        variables_info = []
        continuous_vars = []
        categorical_vars = []
        binary_vars = []

        for var in data_profile.get('variables', []):
            var_name = var.get('name', 'unknown')
            var_type = var.get('statistical_type', 'unknown')

            var_desc = f"- {var_name}: {var_type}"

            if var_type == 'continuous':
                continuous_vars.append(var_name)
                mean = var.get('mean', 'N/A')
                std = var.get('std', 'N/A')
                missing = var.get('missing_pct', 0)
                var_desc += f" (mean={mean:.2f}, SD={std:.2f}, {missing:.1f}% missing)"
            elif var_type == 'categorical':
                categorical_vars.append(var_name)
                cats = var.get('categories', [])
                n_cats = len(cats)
                missing = var.get('missing_pct', 0)
                var_desc += f" ({n_cats} categories: {', '.join(str(c) for c in cats[:5])}, {missing:.1f}% missing)"
            elif var_type == 'binary':
                binary_vars.append(var_name)
                missing = var.get('missing_pct', 0)
                var_desc += f" (binary, {missing:.1f}% missing)"

            variables_info.append(var_desc)

        # Detected potential roles
        potential_outcomes = data_profile.get('potential_outcome_columns', [])
        potential_groups = data_profile.get('potential_group_columns', [])

        # Warnings
        warnings = data_profile.get('warnings', [])

        # User preferences
        prefs_text = ""
        if user_preferences:
            prefs_text = f"""
USER PREFERENCES:
- Significance level: {user_preferences.get('alpha', 0.05)}
- Preferred methods: {', '.join(user_preferences.get('preferred_methods', ['standard']))}
- Focus areas: {', '.join(user_preferences.get('focus_areas', ['primary analysis']))}
"""

        prompt = f"""Create a comprehensive statistical analysis plan for the following:

RESEARCH QUESTION:
{research_question}

DATASET SUMMARY:
- Sample size: {n_rows} observations
- Continuous variables ({len(continuous_vars)}): {', '.join(continuous_vars[:10])}
- Categorical variables ({len(categorical_vars)}): {', '.join(categorical_vars[:10])}
- Binary variables ({len(binary_vars)}): {', '.join(binary_vars[:10])}

VARIABLE DETAILS:
{chr(10).join(variables_info)}

DETECTED POTENTIAL VARIABLE ROLES:
- Likely outcome variables: {', '.join(potential_outcomes) if potential_outcomes else 'Not auto-detected'}
- Likely grouping/treatment variables: {', '.join(potential_groups) if potential_groups else 'Not auto-detected'}

DATA QUALITY NOTES:
{chr(10).join(['- ' + w for w in warnings]) if warnings else '- No major issues detected'}
{prefs_text}

Generate a COMPREHENSIVE statistical analysis plan in the JSON format specified.
Include ALL details needed to execute each analysis.
Be specific about which variables to use for each analysis.
Consider the sample size ({n_rows}) when recommending methods.
"""
        return prompt

    def _parse_response(
        self,
        response: str,
        research_question: str,
        data_profile: Dict
    ) -> ComprehensiveStatisticalPlan:
        """Parse LLM response into ComprehensiveStatisticalPlan."""

        import uuid
        from datetime import datetime

        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = {}

            # Parse analysis steps
            def parse_steps(steps_data: List[Dict]) -> List[AnalysisStep]:
                steps = []
                for s in steps_data:
                    steps.append(AnalysisStep(
                        step_id=s.get('step_id', str(uuid.uuid4())[:8]),
                        name=s.get('name', 'Unnamed Analysis'),
                        category=s.get('category', 'unknown'),
                        method=s.get('method', 'unknown'),
                        description=s.get('description', ''),
                        rationale=s.get('rationale', ''),
                        variables=s.get('variables', {}),
                        parameters=s.get('parameters', {}),
                        assumptions=s.get('assumptions', []),
                        assumption_tests=s.get('assumption_tests', []),
                        interpretation_guidance=s.get('interpretation_guidance', ''),
                        expected_output=s.get('expected_output', []),
                        fallback_method=s.get('fallback_method'),
                        priority=s.get('priority', 'secondary'),
                        depends_on=s.get('depends_on', [])
                    ))
                return steps

            # Parse visualizations
            def parse_viz(viz_data: List[Dict]) -> List[VisualizationStep]:
                vizs = []
                for v in viz_data:
                    vizs.append(VisualizationStep(
                        viz_id=v.get('viz_id', str(uuid.uuid4())[:8]),
                        name=v.get('name', 'Unnamed Plot'),
                        plot_type=v.get('plot_type', 'unknown'),
                        variables=v.get('variables', {}),
                        parameters=v.get('parameters', {}),
                        purpose=v.get('purpose', ''),
                        related_analysis=v.get('related_analysis')
                    ))
                return vizs

            plan = ComprehensiveStatisticalPlan(
                plan_id=str(uuid.uuid4())[:12],
                created_at=datetime.now().isoformat(),
                research_question=research_question,
                research_type=data.get('research_type', 'unknown'),
                study_design=data.get('study_design', 'unknown'),
                sample_size=data_profile.get('n_rows', 0),
                outcome_variables=data.get('outcome_variables', []),
                exposure_variables=data.get('exposure_variables', []),
                covariates=data.get('covariates', []),
                descriptive_analyses=parse_steps(data.get('descriptive_analyses', [])),
                primary_analyses=parse_steps(data.get('primary_analyses', [])),
                secondary_analyses=parse_steps(data.get('secondary_analyses', [])),
                sensitivity_analyses=parse_steps(data.get('sensitivity_analyses', [])),
                subgroup_analyses=parse_steps(data.get('subgroup_analyses', [])),
                assumption_checks=parse_steps(data.get('assumption_checks', [])),
                visualizations=parse_viz(data.get('visualizations', [])),
                missing_data_strategy=data.get('missing_data_strategy', 'Complete case analysis'),
                multiple_testing_correction=data.get('multiple_testing_correction', 'None specified'),
                significance_level=data.get('significance_level', 0.05),
                clinical_significance_thresholds=data.get('clinical_significance_thresholds', {}),
                limitations=data.get('limitations', []),
                status='draft'
            )

            return plan

        except Exception as e:
            # Return minimal plan on error
            return ComprehensiveStatisticalPlan(
                plan_id=str(uuid.uuid4())[:12],
                created_at=datetime.now().isoformat(),
                research_question=research_question,
                research_type='unknown',
                study_design='unknown',
                sample_size=data_profile.get('n_rows', 0),
                outcome_variables=[],
                exposure_variables=[],
                covariates=[],
                descriptive_analyses=[],
                primary_analyses=[],
                secondary_analyses=[],
                sensitivity_analyses=[],
                subgroup_analyses=[],
                assumption_checks=[],
                visualizations=[],
                missing_data_strategy='Not specified',
                multiple_testing_correction='Not specified',
                significance_level=0.05,
                clinical_significance_thresholds={},
                limitations=[f'Error generating plan: {str(e)}'],
                status='error'
            )

    def modify_plan(
        self,
        plan: ComprehensiveStatisticalPlan,
        modifications: Dict[str, Any]
    ) -> ComprehensiveStatisticalPlan:
        """
        Apply user modifications to the plan.

        Parameters:
        -----------
        plan : ComprehensiveStatisticalPlan
            The current plan
        modifications : Dict
            User modifications like:
            - add_analysis: {step details}
            - remove_analysis: step_id
            - modify_analysis: {step_id, changes}
            - change_method: {step_id, new_method}
            - add_subgroup: variable
            - change_significance: new_alpha

        Returns:
        --------
        Modified ComprehensiveStatisticalPlan
        """
        # Track modifications
        plan.user_modifications.append({
            'timestamp': datetime.now().isoformat(),
            'modifications': modifications
        })

        # Add analysis
        if 'add_analysis' in modifications:
            new_step = modifications['add_analysis']
            step = AnalysisStep(
                step_id=new_step.get('step_id', f"user_{len(plan.primary_analyses)}"),
                name=new_step.get('name', 'User-added Analysis'),
                category=new_step.get('category', 'comparative'),
                method=new_step.get('method', ''),
                description=new_step.get('description', ''),
                rationale=new_step.get('rationale', 'User requested'),
                variables=new_step.get('variables', {}),
                parameters=new_step.get('parameters', {}),
                assumptions=new_step.get('assumptions', []),
                assumption_tests=[],
                interpretation_guidance=new_step.get('interpretation_guidance', ''),
                expected_output=[],
                priority=new_step.get('priority', 'secondary')
            )

            if new_step.get('priority') == 'primary':
                plan.primary_analyses.append(step)
            else:
                plan.secondary_analyses.append(step)

        # Remove analysis
        if 'remove_analysis' in modifications:
            step_id = modifications['remove_analysis']
            plan.primary_analyses = [s for s in plan.primary_analyses if s.step_id != step_id]
            plan.secondary_analyses = [s for s in plan.secondary_analyses if s.step_id != step_id]
            plan.sensitivity_analyses = [s for s in plan.sensitivity_analyses if s.step_id != step_id]

        # Modify analysis
        if 'modify_analysis' in modifications:
            mod = modifications['modify_analysis']
            step_id = mod.get('step_id')
            changes = mod.get('changes', {})

            for analyses in [plan.primary_analyses, plan.secondary_analyses, plan.sensitivity_analyses]:
                for step in analyses:
                    if step.step_id == step_id:
                        for key, value in changes.items():
                            if hasattr(step, key):
                                setattr(step, key, value)

        # Change significance level
        if 'change_significance' in modifications:
            plan.significance_level = modifications['change_significance']

        # Add subgroup analysis
        if 'add_subgroup' in modifications:
            subgroup_var = modifications['add_subgroup']
            # Find primary outcome
            if plan.primary_analyses:
                primary = plan.primary_analyses[0]
                plan.subgroup_analyses.append(AnalysisStep(
                    step_id=f"subgroup_{subgroup_var}",
                    name=f"Subgroup Analysis by {subgroup_var}",
                    category="subgroup",
                    method="subgroup_analysis",
                    description=f"Examine treatment effect heterogeneity across {subgroup_var} levels",
                    rationale="User-requested subgroup analysis",
                    variables={
                        **primary.variables,
                        'subgroup': subgroup_var
                    },
                    parameters={'interaction_test': True},
                    assumptions=[],
                    assumption_tests=[],
                    interpretation_guidance="Report effect estimates within each subgroup and interaction p-value",
                    expected_output=['subgroup_effects', 'interaction_p'],
                    priority="subgroup"
                ))

        return plan

    def confirm_plan(self, plan: ComprehensiveStatisticalPlan) -> ComprehensiveStatisticalPlan:
        """Mark plan as confirmed and ready for execution."""
        plan.status = 'confirmed'
        return plan
