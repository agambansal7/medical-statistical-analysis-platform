"""Visualization endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, Any, List, Optional
import json

from services import session_manager, analysis_service
from models import VisualizationRequest, VisualizationResponse

router = APIRouter(prefix="/viz", tags=["Visualization"])


@router.post("/create")
async def create_visualization(request: VisualizationRequest):
    """Create a visualization.

    Available plot types:
    - histogram: Distribution of a single variable
    - boxplot: Box plot for group comparison
    - scatter: Scatter plot with optional regression
    - correlation_heatmap: Correlation matrix heatmap
    - bar: Bar chart
    - qq: Q-Q plot for normality
    - paired: Paired comparison (spaghetti plot)
    """
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    # Merge variables and options
    params = {**request.variables}
    if request.options:
        params.update(request.options)

    result = analysis_service.create_visualization(
        session.data,
        request.plot_type,
        params
    )

    if result.get("success"):
        return JSONResponse(content={
            "success": True,
            "image": result["image"],
            "format": result.get("format", "png")
        })
    else:
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))


@router.get("/types")
async def get_visualization_types():
    """Get available visualization types."""
    return JSONResponse(content={
        "visualization_types": {
            "histogram": {
                "name": "Histogram",
                "description": "Distribution of a continuous variable",
                "required": ["variable"],
                "optional": ["title", "bins", "show_normal"]
            },
            "boxplot": {
                "name": "Box Plot",
                "description": "Compare distributions across groups",
                "required": ["value_col"],
                "optional": ["group_col", "title", "show_points"]
            },
            "scatter": {
                "name": "Scatter Plot",
                "description": "Relationship between two variables",
                "required": ["x_col", "y_col"],
                "optional": ["hue", "show_regression", "title"]
            },
            "correlation_heatmap": {
                "name": "Correlation Heatmap",
                "description": "Matrix of correlations",
                "optional": ["variables", "method", "title"]
            },
            "bar": {
                "name": "Bar Chart",
                "description": "Categorical comparisons",
                "required": ["x_col"],
                "optional": ["y_col", "hue", "title", "horizontal"]
            },
            "qq": {
                "name": "Q-Q Plot",
                "description": "Assess normality",
                "required": ["variable"],
                "optional": ["title"]
            },
            "paired": {
                "name": "Paired Plot",
                "description": "Before-after comparison",
                "required": ["before", "after"],
                "optional": ["labels", "title"]
            },
            "violin": {
                "name": "Violin Plot",
                "description": "Distribution shape by group",
                "required": ["value_col", "group_col"],
                "optional": ["title", "split", "hue"]
            }
        }
    })


@router.get("/suggestions/{session_id}")
async def get_visualization_suggestions(session_id: str):
    """Get suggested visualizations based on data profile."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.data_profile:
        raise HTTPException(status_code=400, detail="No data profile available")

    suggestions = []

    # Get variable types
    continuous_vars = [v["name"] for v in session.data_profile.get("variables", [])
                       if v["statistical_type"] == "continuous"]
    categorical_vars = [v["name"] for v in session.data_profile.get("variables", [])
                        if v["statistical_type"] in ["categorical", "binary"]]

    # Suggest histograms for continuous variables
    for var in continuous_vars[:5]:
        suggestions.append({
            "type": "histogram",
            "description": f"Distribution of {var}",
            "params": {"variable": var}
        })

    # Suggest boxplots for continuous vs categorical
    if continuous_vars and categorical_vars:
        for cont in continuous_vars[:3]:
            for cat in categorical_vars[:2]:
                suggestions.append({
                    "type": "boxplot",
                    "description": f"{cont} by {cat}",
                    "params": {"value_col": cont, "group_col": cat}
                })

    # Suggest correlation heatmap if multiple continuous
    if len(continuous_vars) >= 3:
        suggestions.append({
            "type": "correlation_heatmap",
            "description": "Correlation matrix of continuous variables",
            "params": {"variables": continuous_vars[:10]}
        })

    # Suggest scatter plots for pairs of continuous
    if len(continuous_vars) >= 2:
        for i, var1 in enumerate(continuous_vars[:3]):
            for var2 in continuous_vars[i+1:4]:
                suggestions.append({
                    "type": "scatter",
                    "description": f"{var1} vs {var2}",
                    "params": {"x_col": var1, "y_col": var2}
                })

    return JSONResponse(content={
        "success": True,
        "suggestions": suggestions[:15]  # Limit to 15
    })


@router.post("/interactive/create")
async def create_interactive_visualization(request: Dict[str, Any]):
    """Create an interactive Plotly visualization.

    Request body:
    {
        "session_id": "session-uuid",
        "plot_type": "forest_plot" | "km_curve" | "regression" | "scatter" | "boxplot" | "histogram",
        "params": {
            // Plot-specific parameters
        },
        "options": {
            "title": "Plot Title",
            "width": 800,
            "height": 600,
            "theme": "plotly_white"
        }
    }

    Returns Plotly JSON specification that can be rendered client-side.
    """
    session_id = request.get("session_id")
    plot_type = request.get("plot_type")
    params = request.get("params", {})
    options = request.get("options", {})

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        # Generate interactive plot based on type
        if plot_type == "forest_plot":
            fig_json = _create_interactive_forest_plot(session.data, params, options)
        elif plot_type == "km_curve":
            fig_json = _create_interactive_km_curve(session.data, params, options)
        elif plot_type == "regression":
            fig_json = _create_interactive_regression_plot(session.data, params, options)
        elif plot_type == "scatter":
            fig_json = _create_interactive_scatter(session.data, params, options)
        elif plot_type == "boxplot":
            fig_json = _create_interactive_boxplot(session.data, params, options)
        elif plot_type == "histogram":
            fig_json = _create_interactive_histogram(session.data, params, options)
        elif plot_type == "correlation_heatmap":
            fig_json = _create_interactive_heatmap(session.data, params, options)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown plot type: {plot_type}")

        return JSONResponse(content={
            "success": True,
            "plot_type": plot_type,
            "plotly_json": fig_json,
            "interactive": True
        })

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Plotly not installed. Run: pip install plotly"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _create_interactive_forest_plot(df, params: Dict, options: Dict) -> str:
    """Create publication-quality interactive forest plot with Plotly."""
    import plotly.graph_objects as go
    import numpy as np

    # Get analysis results from params
    variables = params.get("variables", [])
    estimates = params.get("estimates", [])
    ci_lower = params.get("ci_lower", [])
    ci_upper = params.get("ci_upper", [])
    p_values = params.get("p_values", [])

    # Handle empty data
    if not variables:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for forest plot.<br>Run a regression analysis first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#6b7280')
        )
        fig.update_layout(
            width=600, height=300,
            template='plotly_white'
        )
        return fig.to_json()

    # Ensure all arrays are the same length
    n = len(variables)
    if len(estimates) < n:
        estimates = list(estimates) + [1.0] * (n - len(estimates))
    if len(ci_lower) < n:
        ci_lower = list(ci_lower) + [0.5] * (n - len(ci_lower))
    if len(ci_upper) < n:
        ci_upper = list(ci_upper) + [2.0] * (n - len(ci_upper))
    if len(p_values) < n:
        p_values = list(p_values) + [1.0] * (n - len(p_values))

    # Use numeric y positions for consistency
    y_positions = list(range(n))

    # Determine colors based on significance and direction
    colors = []
    symbols = []
    for est, low, high, p in zip(estimates, ci_lower, ci_upper, p_values):
        if p < 0.05:
            if high < 1:
                colors.append('#059669')  # Green - protective
                symbols.append('diamond')
            elif low > 1:
                colors.append('#dc2626')  # Red - harmful
                symbols.append('diamond')
            else:
                colors.append('#1e40af')  # Blue - significant but crosses 1
                symbols.append('diamond')
        else:
            colors.append('#6b7280')  # Gray - not significant
            symbols.append('square')

    # Create figure
    fig = go.Figure()

    # Add horizontal CI lines and point estimates for each variable
    for i, (var, est, low, high, p, color, symbol) in enumerate(
            zip(variables, estimates, ci_lower, ci_upper, p_values, colors, symbols)):

        # Format p-value for display
        if p < 0.001:
            p_str = '<0.001'
        elif p < 0.01:
            p_str = f'{p:.3f}'
        else:
            p_str = f'{p:.2f}'

        sig_text = '**' if p < 0.01 else '*' if p < 0.05 else ''

        # Add CI line (horizontal)
        fig.add_trace(go.Scatter(
            x=[low, high],
            y=[i, i],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add CI caps (vertical lines at ends)
        cap_height = 0.2
        fig.add_trace(go.Scatter(
            x=[low, low],
            y=[i - cap_height, i + cap_height],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[high, high],
            y=[i - cap_height, i + cap_height],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add point estimate
        fig.add_trace(go.Scatter(
            x=[est],
            y=[i],
            mode='markers',
            marker=dict(
                size=14,
                color=color,
                symbol=symbol,
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hovertemplate=(
                f'<b>{var}</b><br>'
                f'Estimate: {est:.2f}<br>'
                f'95% CI: [{low:.2f}, {high:.2f}]<br>'
                f'P-value: {p_str}{sig_text}<extra></extra>'
            )
        ))

    # Add reference line at 1
    fig.add_vline(
        x=1,
        line=dict(color='#374151', width=2, dash='solid'),
        annotation_text="Null",
        annotation_position="top"
    )

    # Calculate x-axis range
    all_vals = list(ci_lower) + list(ci_upper)
    valid_vals = [v for v in all_vals if v and v > 0]
    x_min = min(valid_vals) * 0.5 if valid_vals else 0.1
    x_max = max(valid_vals) * 1.5 if valid_vals else 10

    # Add annotation table with estimates on the right
    label = options.get("label", "HR/OR")
    annotations = []

    # Header annotations
    annotations.extend([
        dict(x=1.12, y=1.05, xref='paper', yref='paper',
             text=f'<b>{label}</b>', showarrow=False, font=dict(size=11)),
        dict(x=1.25, y=1.05, xref='paper', yref='paper',
             text='<b>95% CI</b>', showarrow=False, font=dict(size=11)),
        dict(x=1.38, y=1.05, xref='paper', yref='paper',
             text='<b>P-value</b>', showarrow=False, font=dict(size=11))
    ])

    # Data row annotations
    for i, (est, low, high, p) in enumerate(zip(estimates, ci_lower, ci_upper, p_values)):
        y_pos = 1 - (i + 0.5) / (n + 0.5) * 0.92
        p_str = '<0.001' if p < 0.001 else f'{p:.3f}' if p < 0.01 else f'{p:.2f}'
        font_color = colors[i]

        annotations.extend([
            dict(x=1.12, y=y_pos, xref='paper', yref='paper',
                 text=f'{est:.2f}', showarrow=False, font=dict(size=10, color=font_color)),
            dict(x=1.25, y=y_pos, xref='paper', yref='paper',
                 text=f'({low:.2f}-{high:.2f})', showarrow=False, font=dict(size=10, color=font_color)),
            dict(x=1.38, y=y_pos, xref='paper', yref='paper',
                 text=p_str, showarrow=False, font=dict(size=10, color=font_color))
        ])

    # Add "Favors" annotations
    annotations.extend([
        dict(x=0.15, y=-0.1, xref='paper', yref='paper',
             text='← Lower Risk', showarrow=False, font=dict(size=10, color='#059669')),
        dict(x=0.85, y=-0.1, xref='paper', yref='paper',
             text='Higher Risk →', showarrow=False, font=dict(size=10, color='#dc2626'))
    ])

    # Update layout
    fig.update_layout(
        title=dict(
            text=options.get("title", "Forest Plot: Effect Estimates with 95% CI"),
            font=dict(size=16, color='#1f2937'),
            x=0.4
        ),
        xaxis=dict(
            title=dict(text=options.get("x_label", "Hazard Ratio / Odds Ratio"),
                       font=dict(size=12, color='#374151')),
            type='log' if options.get("log_scale", True) else 'linear',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title='',
            tickvals=y_positions,
            ticktext=variables,
            showgrid=False,
            zeroline=False,
            autorange='reversed'
        ),
        width=options.get("width", 900),
        height=options.get("height", max(400, n * 45 + 150)),
        template='plotly_white',
        hovermode='closest',
        margin=dict(l=180, r=180, t=80, b=80),
        annotations=annotations,
        plot_bgcolor='rgba(248,250,252,0.5)'
    )

    # Add alternating row backgrounds
    for i in range(n):
        if i % 2 == 0:
            fig.add_hrect(
                y0=i - 0.4, y1=i + 0.4,
                fillcolor='rgba(241,245,249,0.5)',
                line_width=0,
                layer='below'
            )

    return fig.to_json()


def _create_interactive_km_curve(df, params: Dict, options: Dict) -> str:
    """Create publication-quality interactive Kaplan-Meier curve with Plotly."""
    import plotly.graph_objects as go
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import numpy as np

    time_col = params.get("time")
    event_col = params.get("event")
    group_col = params.get("group")

    fig = go.Figure()

    # Publication-quality color palette
    colors = ['#2E86AB', '#E94F37', '#1B998B', '#F18F01', '#8338EC', '#FF006E']

    annotations = []
    log_rank_text = ""

    if group_col and group_col in df.columns:
        groups = sorted(df[group_col].dropna().unique())

        # Perform log-rank test if we have 2 groups
        if len(groups) == 2:
            try:
                g1 = df[df[group_col] == groups[0]]
                g2 = df[df[group_col] == groups[1]]
                result = logrank_test(g1[time_col], g2[time_col],
                                     event_observed_A=g1[event_col],
                                     event_observed_B=g2[event_col])
                p_val = result.p_value
                p_str = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
                log_rank_text = f"Log-rank p = {p_str}"
            except:
                pass

        for i, group in enumerate(groups):
            subset = df[df[group_col] == group].dropna(subset=[time_col, event_col])
            n_total = len(subset)
            n_events = int(subset[event_col].sum())

            kmf = KaplanMeierFitter()
            kmf.fit(subset[time_col], event_observed=subset[event_col], label=str(group))

            # Get survival data
            timeline = kmf.survival_function_.index.tolist()
            survival_prob = kmf.survival_function_[str(group)].tolist()
            ci_lower = kmf.confidence_interval_[f'{group}_lower_0.95'].tolist()
            ci_upper = kmf.confidence_interval_[f'{group}_upper_0.95'].tolist()

            color = colors[i % len(colors)]

            # Add survival curve with step interpolation
            fig.add_trace(go.Scatter(
                x=timeline,
                y=survival_prob,
                mode='lines',
                name=f'{group} (n={n_total}, events={n_events})',
                line=dict(color=color, width=3, shape='hv'),
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.1f}<br>Survival: %{y:.1%}<extra></extra>'
            ))

            # Add confidence interval with matching color
            fig.add_trace(go.Scatter(
                x=timeline + timeline[::-1],
                y=ci_upper + ci_lower[::-1],
                fill='toself',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add censoring marks
            censored = subset[subset[event_col] == 0]
            if len(censored) > 0:
                censor_times = censored[time_col].values
                censor_probs = [kmf.predict(t) for t in censor_times]
                fig.add_trace(go.Scatter(
                    x=censor_times,
                    y=censor_probs,
                    mode='markers',
                    marker=dict(symbol='line-ns', size=10, color=color, line=dict(width=2)),
                    showlegend=False,
                    hovertemplate='Censored at %{x:.1f}<extra></extra>'
                ))
    else:
        clean = df.dropna(subset=[time_col, event_col])
        kmf = KaplanMeierFitter()
        kmf.fit(clean[time_col], event_observed=clean[event_col])

        timeline = kmf.survival_function_.index.tolist()
        survival_prob = kmf.survival_function_['KM_estimate'].tolist()
        ci = kmf.confidence_interval_survival_function_
        ci_lower = ci.iloc[:, 0].tolist()
        ci_upper = ci.iloc[:, 1].tolist()

        color = colors[0]
        n_total = len(clean)
        n_events = int(clean[event_col].sum())

        fig.add_trace(go.Scatter(
            x=timeline,
            y=survival_prob,
            mode='lines',
            name=f'Overall (n={n_total}, events={n_events})',
            line=dict(color=color, width=3, shape='hv'),
            hovertemplate='Time: %{x:.1f}<br>Survival: %{y:.1%}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=timeline + timeline[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add log-rank annotation if available
    if log_rank_text:
        annotations.append(dict(
            x=0.98, y=0.02,
            xref='paper', yref='paper',
            text=f'<b>{log_rank_text}</b>',
            showarrow=False,
            font=dict(size=14, color='#333'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc',
            borderwidth=1,
            borderpad=6
        ))

    # Enhanced layout for publication quality
    fig.update_layout(
        title=dict(
            text=options.get("title", "Kaplan-Meier Survival Curves"),
            font=dict(size=18, color='#1a1a1a', family='Arial Black'),
            x=0.5, xanchor='center'
        ),
        xaxis=dict(
            title=dict(text=options.get("x_label", "Time"), font=dict(size=14, color='#333')),
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
            zeroline=True, zerolinewidth=1, zerolinecolor='#333',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text=options.get("y_label", "Survival Probability"), font=dict(size=14, color='#333')),
            range=[-0.02, 1.05], tickformat='.0%',
            showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            tickfont=dict(size=12)
        ),
        width=options.get("width", 900),
        height=options.get("height", 600),
        template='plotly_white',
        hovermode='closest',
        legend=dict(
            yanchor="bottom", y=0.02, xanchor="left", x=0.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc', borderwidth=1,
            font=dict(size=12)
        ),
        annotations=annotations,
        margin=dict(t=60, r=40, b=60, l=60),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig.to_json()


def _create_interactive_regression_plot(df, params: Dict, options: Dict) -> str:
    """Create interactive regression plot with residuals."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    from scipy import stats

    x_col = params.get("x")
    y_col = params.get("y")

    # Fit regression
    clean = df[[x_col, y_col]].dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean[x_col], clean[y_col])

    # Calculate predictions and residuals
    predictions = intercept + slope * clean[x_col]
    residuals = clean[y_col] - predictions

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Regression Plot', 'Residuals'),
        horizontal_spacing=0.1
    )

    # Regression plot
    fig.add_trace(go.Scatter(
        x=clean[x_col],
        y=clean[y_col],
        mode='markers',
        name='Data',
        marker=dict(color='#4f46e5', opacity=0.6),
        hovertemplate=f'{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>'
    ), row=1, col=1)

    # Regression line
    x_range = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
    y_pred = intercept + slope * x_range

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name=f'y = {slope:.3f}x + {intercept:.3f}',
        line=dict(color='#dc2626', width=2)
    ), row=1, col=1)

    # Residuals plot
    fig.add_trace(go.Scatter(
        x=predictions,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='#059669', opacity=0.6),
        hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # Update layout
    fig.update_layout(
        title=options.get("title", f"Regression: {y_col} ~ {x_col}"),
        width=options.get("width", 1000),
        height=options.get("height", 450),
        template=options.get("theme", "plotly_white"),
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=f'R² = {r_value**2:.3f}, p = {p_value:.4f}',
                showarrow=False,
                font=dict(size=12),
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1
            )
        ]
    )

    return fig.to_json()


def _create_interactive_scatter(df, params: Dict, options: Dict) -> str:
    """Create interactive scatter plot."""
    import plotly.express as px

    x_col = params.get("x")
    y_col = params.get("y")
    color_col = params.get("color")
    size_col = params.get("size")

    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        size=size_col,
        hover_data=df.columns.tolist()[:5],
        title=options.get("title", f"{y_col} vs {x_col}"),
        template=options.get("theme", "plotly_white"),
        trendline="ols" if options.get("show_trendline", False) else None
    )

    fig.update_layout(
        width=options.get("width", 800),
        height=options.get("height", 500)
    )

    return fig.to_json()


def _create_interactive_boxplot(df, params: Dict, options: Dict) -> str:
    """Create interactive box plot."""
    import plotly.express as px

    y_col = params.get("y") or params.get("value_col")
    x_col = params.get("x") or params.get("group_col")
    color_col = params.get("color")

    fig = px.box(
        df, x=x_col, y=y_col,
        color=color_col,
        points="outliers" if not options.get("show_all_points", False) else "all",
        title=options.get("title", f"{y_col} by {x_col}"),
        template=options.get("theme", "plotly_white"),
        notched=options.get("notched", False)
    )

    fig.update_layout(
        width=options.get("width", 800),
        height=options.get("height", 500)
    )

    return fig.to_json()


def _create_interactive_histogram(df, params: Dict, options: Dict) -> str:
    """Create interactive histogram."""
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy import stats

    variable = params.get("variable")
    color_col = params.get("color")
    nbins = options.get("bins", 30)

    fig = px.histogram(
        df, x=variable,
        color=color_col,
        nbins=nbins,
        title=options.get("title", f"Distribution of {variable}"),
        template=options.get("theme", "plotly_white"),
        marginal="box" if options.get("show_marginal", True) else None
    )

    # Add normal curve overlay if requested
    if options.get("show_normal", False):
        data = df[variable].dropna()
        mu, std = data.mean(), data.std()
        x_range = np.linspace(data.min(), data.max(), 100)
        y_normal = stats.norm.pdf(x_range, mu, std)

        # Scale to match histogram
        bin_width = (data.max() - data.min()) / nbins
        y_normal = y_normal * len(data) * bin_width

        fig.add_trace(go.Scatter(
            x=x_range, y=y_normal,
            mode='lines',
            name='Normal',
            line=dict(color='red', width=2)
        ))

    fig.update_layout(
        width=options.get("width", 800),
        height=options.get("height", 500)
    )

    return fig.to_json()


def _create_interactive_heatmap(df, params: Dict, options: Dict) -> str:
    """Create interactive correlation heatmap."""
    import plotly.express as px
    import plotly.graph_objects as go

    variables = params.get("variables")
    if variables:
        subset = df[variables].select_dtypes(include=['number'])
    else:
        subset = df.select_dtypes(include=['number'])

    corr_matrix = subset.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=options.get("title", "Correlation Matrix"),
        width=options.get("width", 800),
        height=options.get("height", 700),
        template=options.get("theme", "plotly_white"),
        xaxis=dict(tickangle=45)
    )

    return fig.to_json()


@router.get("/interactive/types")
async def get_interactive_plot_types():
    """Get available interactive plot types."""
    return JSONResponse(content={
        "interactive_types": {
            "forest_plot": {
                "name": "Interactive Forest Plot",
                "description": "Zoomable, hoverable forest plot for effect estimates",
                "params": ["variables", "estimates", "ci_lower", "ci_upper", "p_values"]
            },
            "km_curve": {
                "name": "Interactive Kaplan-Meier",
                "description": "Survival curves with hover details and zoom",
                "params": ["time", "event", "group"]
            },
            "regression": {
                "name": "Interactive Regression",
                "description": "Scatter with regression line and residual plot",
                "params": ["x", "y"]
            },
            "scatter": {
                "name": "Interactive Scatter",
                "description": "Scatter plot with color/size encoding",
                "params": ["x", "y", "color", "size"]
            },
            "boxplot": {
                "name": "Interactive Box Plot",
                "description": "Box plot with individual points",
                "params": ["y", "x", "color"]
            },
            "histogram": {
                "name": "Interactive Histogram",
                "description": "Distribution with marginal plots",
                "params": ["variable", "color"]
            },
            "correlation_heatmap": {
                "name": "Interactive Heatmap",
                "description": "Correlation matrix with hover values",
                "params": ["variables"]
            }
        }
    })
