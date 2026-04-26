import React, { useState, useEffect } from 'react';
import { useStore } from '../hooks/useStore';
import { analysisApi, vizApi, chatApi, codeApi } from '../services/api';
import {
  FileText,
  Image,
  ChevronDown,
  ChevronRight,
  Loader2,
  CheckCircle,
  XCircle,
  FileTextIcon,
  Download,
  Copy,
  BarChart3,
  Activity,
  Code,
} from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import CodePanel from './CodePanel';

// Simple markdown renderer component
const SimpleMarkdown: React.FC<{ content: string }> = ({ content }) => {
  const renderLine = (line: string, index: number): React.ReactNode => {
    if (line.startsWith('### ')) {
      return <h3 key={index} className="text-lg font-medium text-slate-200 mb-2 mt-4">{renderInline(line.slice(4))}</h3>;
    }
    if (line.startsWith('## ')) {
      return <h2 key={index} className="text-xl font-semibold text-slate-100 mb-3 mt-5 border-b border-slate-700 pb-2">{renderInline(line.slice(3))}</h2>;
    }
    if (line.startsWith('# ')) {
      return <h1 key={index} className="text-2xl font-bold text-slate-100 mb-4 mt-6">{renderInline(line.slice(2))}</h1>;
    }
    if (line.match(/^[-*]\s/)) {
      return <li key={index} className="ml-4 mb-1 list-disc text-slate-300">{renderInline(line.slice(2))}</li>;
    }
    if (line.match(/^\d+\.\s/)) {
      const text = line.replace(/^\d+\.\s/, '');
      return <li key={index} className="ml-4 mb-1 list-decimal text-slate-300">{renderInline(text)}</li>;
    }
    if (line.trim() === '') {
      return <div key={index} className="h-2" />;
    }
    return <p key={index} className="text-slate-300 mb-3 leading-relaxed">{renderInline(line)}</p>;
  };

  const renderInline = (text: string): React.ReactNode => {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="font-semibold text-slate-100">{part.slice(2, -2)}</strong>;
      }
      const italicParts = part.split(/(\*[^*]+\*)/g);
      return italicParts.map((iPart, j) => {
        if (iPart.startsWith('*') && iPart.endsWith('*') && !iPart.startsWith('**')) {
          return <em key={`${i}-${j}`} className="italic">{iPart.slice(1, -1)}</em>;
        }
        return <span key={`${i}-${j}`}>{iPart}</span>;
      });
    });
  };

  const lines = content.split('\n');
  return <div className="space-y-1">{lines.map((line, index) => renderLine(line, index))}</div>;
};

interface ResultsPanelProps {
  showVisualizations?: boolean;
}

const interpretPValue = (p: number): { text: string; significant: boolean } => {
  if (p < 0.001) return { text: 'Highly significant (p < 0.001)', significant: true };
  if (p < 0.01) return { text: 'Very significant (p < 0.01)', significant: true };
  if (p < 0.05) return { text: 'Significant (p < 0.05)', significant: true };
  if (p < 0.1) return { text: 'Marginally significant (p < 0.1)', significant: false };
  return { text: 'Not significant (p ≥ 0.1)', significant: false };
};

const interpretHR = (hr: number): string => {
  if (hr > 1) return `${((hr - 1) * 100).toFixed(1)}% increased risk`;
  if (hr < 1) return `${((1 - hr) * 100).toFixed(1)}% reduced risk`;
  return 'No effect';
};

const interpretOR = (or: number): string => {
  if (or > 1) return `${((or - 1) * 100).toFixed(1)}% higher odds`;
  if (or < 1) return `${((1 - or) * 100).toFixed(1)}% lower odds`;
  return 'No association';
};

const renderMarkdownBold = (text: string): React.ReactNode => {
  if (!text) return null;
  return text.split('**').map((part, i) =>
    i % 2 === 1 ? <strong key={i} className="text-slate-100">{part}</strong> : <span key={i}>{part}</span>
  );
};

export default function ResultsPanel({ showVisualizations = false }: ResultsPanelProps) {
  const { sessionId, analysisResults, resultsReport, setResultsReport, analysisPlan, dataProfile } = useStore();
  const [history, setHistory] = useState<Record<string, unknown>[]>([]);
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());
  const [visualizations, setVisualizations] = useState<Map<string, string>>(new Map());
  const [vizSuggestions, setVizSuggestions] = useState<unknown[]>([]);
  const [loading, setLoading] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [showCodeFor, setShowCodeFor] = useState<string | null>(null);
  const [analysisCode, setAnalysisCode] = useState<Map<string, { python?: string; r?: string; packages?: { python?: string[]; r?: string[] } }>>(new Map());

  useEffect(() => {
    const loadHistory = async () => {
      if (!sessionId) return;
      try {
        const data = await analysisApi.getHistory(sessionId);
        setHistory(data);
        const ids = data.map((a: any, i: number) => a.analysis_id || String(i));
        setExpandedResults(new Set(ids));
      } catch (error) {
        console.error('Failed to load history', error);
      }
    };
    loadHistory();
  }, [sessionId, analysisResults]);

  useEffect(() => {
    const loadVizSuggestions = async () => {
      if (!sessionId || !showVisualizations) return;
      try {
        const suggestions = await vizApi.getSuggestions(sessionId);
        setVizSuggestions(suggestions);
      } catch (error) {
        console.error('Failed to load viz suggestions', error);
      }
    };
    loadVizSuggestions();
  }, [sessionId, showVisualizations]);

  const generateReport = async () => {
    if (!sessionId) {
      toast.error('No session found');
      return;
    }
    if (history.length === 0) {
      toast.error('No analyses to generate report from');
      return;
    }
    setGeneratingReport(true);
    try {
      const response = await chatApi.generateReport(sessionId);
      if (response.success && response.report) {
        setResultsReport(response.report);
        setShowReport(true);
        toast.success('Results section generated!');
      } else {
        toast.error(response.message || 'Failed to generate report');
      }
    } catch (error: any) {
      console.error('Failed to generate report:', error);
      toast.error(error.response?.data?.detail || 'Failed to generate report');
    } finally {
      setGeneratingReport(false);
    }
  };

  const copyReportToClipboard = () => {
    if (resultsReport) {
      navigator.clipboard.writeText(resultsReport);
      toast.success('Report copied to clipboard!');
    }
  };

  const downloadReport = () => {
    if (resultsReport) {
      const blob = new Blob([resultsReport], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'results_section.md';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('Report downloaded!');
    }
  };

  const toggleExpand = (id: string) => {
    const newExpanded = new Set(expandedResults);
    if (newExpanded.has(id)) {
      newExpanded.delete(id);
    } else {
      newExpanded.add(id);
    }
    setExpandedResults(newExpanded);
  };

  const loadCodeForAnalysis = async (analysisId: string, analysisType: string) => {
    if (analysisCode.has(analysisId)) {
      setShowCodeFor(showCodeFor === analysisId ? null : analysisId);
      return;
    }

    try {
      const result = await codeApi.generate(sessionId!, analysisType, {});
      if (result.success) {
        const newCodeMap = new Map(analysisCode);
        newCodeMap.set(analysisId, {
          python: result.python_code,
          r: result.r_code,
          packages: {
            python: result.python_packages,
            r: result.r_packages,
          },
        });
        setAnalysisCode(newCodeMap);
        setShowCodeFor(analysisId);
      }
    } catch (error) {
      console.error('Failed to load code:', error);
      toast.error('Failed to load reproducible code');
    }
  };

  const createVisualization = async (type: string, params: Record<string, unknown>) => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const result = await vizApi.create(sessionId, type, params as Record<string, string>);
      if (result.success && result.image) {
        const key = `${type}-${JSON.stringify(params)}`;
        setVisualizations(new Map(visualizations.set(key, result.image)));
        toast.success('Visualization created!');
      } else {
        toast.error(result.message || 'Failed to create visualization');
      }
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to create visualization');
      console.error('Visualization error:', error);
    }
    setLoading(false);
  };

  // Parse recommended visualization name into type and parameters
  const parseRecommendedViz = (vizName: string): { type: string; params: Record<string, string>; label: string; description: string } => {
    const name = vizName.toLowerCase();

    // Find group variable from data profile
    const groupVar = dataProfile?.potential_groups?.[0] || 'race';

    // Find outcome/time/event variables from data profile
    const allVars = dataProfile?.variables?.map(v => v.name) || [];
    const outcomeVars = dataProfile?.potential_outcomes || [];

    // Find time-related variable
    const timeVar = allVars.find(v =>
      v.toLowerCase().includes('time') ||
      v.toLowerCase().includes('follow') ||
      v.toLowerCase().includes('duration')
    ) || 'time_to_event';

    // Find event/mortality variable
    const eventVar = allVars.find(v =>
      v.toLowerCase().includes('death') ||
      v.toLowerCase().includes('mortality') ||
      v.toLowerCase().includes('event') ||
      v.toLowerCase() === 'died'
    ) || 'mortality';

    // Find quality of life variable
    const qolVar = allVars.find(v =>
      v.toLowerCase().includes('kccq') ||
      v.toLowerCase().includes('quality') ||
      v.toLowerCase().includes('qol') ||
      v.toLowerCase().includes('score')
    ) || outcomeVars[0] || 'outcome';

    // Kaplan-Meier curves
    if (name.includes('kaplan') || name.includes('km') || name.includes('survival_curve')) {
      return {
        type: 'kaplan_meier',
        params: { time: timeVar, event: eventVar, group: groupVar },
        label: 'Kaplan-Meier Survival Curves',
        description: `Survival curves by ${groupVar}`
      };
    }

    // Forest plot
    if (name.includes('forest')) {
      if (name.includes('hazard') || name.includes('hr') || name.includes('cox')) {
        return {
          type: 'forest_plot_hr',
          params: { analysis_type: 'cox', group: groupVar },
          label: 'Forest Plot: Hazard Ratios',
          description: 'Adjusted hazard ratios with 95% CI'
        };
      }
      return {
        type: 'forest_plot_or',
        params: { analysis_type: 'logistic', group: groupVar },
        label: 'Forest Plot: Odds Ratios',
        description: 'Adjusted odds ratios with 95% CI'
      };
    }

    // Box plot
    if (name.includes('boxplot') || name.includes('box_plot')) {
      return {
        type: 'boxplot',
        params: { value_col: qolVar, group_col: groupVar },
        label: `Box Plot: ${qolVar}`,
        description: `Distribution by ${groupVar}`
      };
    }

    // Bar chart
    if (name.includes('bar_chart') || name.includes('barchart') || name.includes('bar')) {
      const compVar = allVars.find(v =>
        v.toLowerCase().includes('complication') ||
        v.toLowerCase().includes('adverse') ||
        v.toLowerCase().includes('readmission')
      ) || eventVar;
      return {
        type: 'bar_chart',
        params: { variable: compVar, group: groupVar },
        label: `Bar Chart: ${compVar}`,
        description: `Comparison across ${groupVar}`
      };
    }

    // Histogram
    if (name.includes('histogram') || name.includes('distribution')) {
      return {
        type: 'histogram',
        params: { variable: outcomeVars[0] || qolVar },
        label: 'Histogram',
        description: 'Distribution of continuous variable'
      };
    }

    // Scatter plot
    if (name.includes('scatter')) {
      return {
        type: 'scatter',
        params: { x_col: outcomeVars[0] || 'age', y_col: outcomeVars[1] || qolVar },
        label: 'Scatter Plot',
        description: 'Relationship between variables'
      };
    }

    // Violin plot
    if (name.includes('violin')) {
      return {
        type: 'violin',
        params: { value_col: qolVar, group_col: groupVar },
        label: `Violin Plot: ${qolVar}`,
        description: `Distribution shape by ${groupVar}`
      };
    }

    // Default - try to parse the name
    const parts = name.split('_');
    const vizType = parts[0] || 'bar';
    return {
      type: vizType,
      params: { group: groupVar, variable: eventVar },
      label: vizName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      description: 'Custom visualization'
    };
  };

  // Generate all recommended visualizations
  const generateAllVisualizations = async () => {
    if (!sessionId || !analysisPlan?.visualizations) return;
    setLoading(true);

    for (const vizName of analysisPlan.visualizations) {
      const { type, params } = parseRecommendedViz(vizName);
      try {
        const result = await vizApi.create(sessionId, type, params);
        if (result.success && result.image) {
          const key = `${vizName}`;
          setVisualizations(prev => new Map(prev.set(key, result.image)));
        }
      } catch (error) {
        console.error(`Failed to create ${vizName}:`, error);
      }
    }

    setLoading(false);
    toast.success('Visualizations generated!');
  };

  const formatValue = (value: unknown): string => {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'number') {
      if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(3);
      return value.toFixed(4);
    }
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
  };

  const renderKaplanMeierResults = (analysis: any) => {
    const results = analysis.results || {};
    const kmResults = results.km_results || {};
    const logRank = results.log_rank || {};

    return (
      <div className="space-y-6">
        {/* KM Curve Plot - Primary Visual */}
        {results.plot_base64 && (
          <div className="glass-card p-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
            <h4 className="font-semibold text-slate-100 mb-3 flex items-center space-x-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              <span>Kaplan-Meier Survival Curves</span>
            </h4>
            <div className="bg-white rounded-lg p-2">
              <img
                src={`data:image/png;base64,${results.plot_base64}`}
                alt="Kaplan-Meier Survival Curves"
                className="w-full max-w-4xl mx-auto rounded-lg shadow-lg"
              />
            </div>
          </div>
        )}

        {/* Summary Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(kmResults).map(([groupName, groupData]: [string, any]) => (
            <div key={groupName} className="stat-card">
              <div className="text-sm text-cyan-400 font-medium">{groupName}</div>
              <div className="text-xl font-bold text-slate-100">
                {groupData.median_survival != null ? `${groupData.median_survival.toFixed(1)}` : 'NR'}
              </div>
              <div className="text-xs text-slate-500">Median Survival</div>
              <div className="mt-2 text-xs text-slate-400">
                n={groupData.n_observations}, events={groupData.n_events}
              </div>
            </div>
          ))}
        </div>

        {/* Log-Rank Test Result */}
        {logRank.p_value !== undefined && (
          <div className={clsx(
            'glass-card p-4',
            logRank.p_value < 0.05 ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-slate-700'
          )}>
            <div className="flex items-start space-x-3">
              {logRank.p_value < 0.05 ? (
                <CheckCircle className="w-5 h-5 text-emerald-400 mt-0.5" />
              ) : (
                <XCircle className="w-5 h-5 text-slate-500 mt-0.5" />
              )}
              <div>
                <h4 className={clsx('font-semibold mb-1', logRank.p_value < 0.05 ? 'text-emerald-400' : 'text-slate-300')}>
                  Log-Rank Test: p = {logRank.p_value < 0.001 ? '<0.001' : logRank.p_value.toFixed(4)}
                </h4>
                <p className="text-sm text-slate-400">
                  {logRank.p_value < 0.05
                    ? 'Statistically significant difference in survival between groups.'
                    : 'No statistically significant difference in survival between groups.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Interpretation */}
        {(analysis.interpretation || results.interpretation) && (
          <div className="glass-card p-4 border-cyan-500/30 bg-cyan-500/5">
            <h4 className="font-semibold text-cyan-400 mb-2">Clinical Interpretation</h4>
            <div className="text-sm text-slate-300 whitespace-pre-wrap">
              {renderMarkdownBold(analysis.interpretation || results.interpretation)}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCoxResults = (analysis: any) => {
    const results = analysis.results || {};
    const coefficients = results.coefficients || {};

    return (
      <div className="space-y-6">
        {results.plot_base64 && (
          <div className="glass-card p-4">
            <h4 className="font-semibold text-slate-100 mb-3">Forest Plot: Hazard Ratios</h4>
            <img src={`data:image/png;base64,${results.plot_base64}`} alt="Cox Regression Forest Plot" className="w-full max-w-3xl mx-auto rounded-lg" />
          </div>
        )}
        <div className="grid grid-cols-3 gap-4">
          <div className="stat-card blue">
            <div className="text-sm text-blue-400 font-medium">Observations</div>
            <div className="text-2xl font-bold text-slate-100">{results.n_observations || 0}</div>
          </div>
          <div className="stat-card purple">
            <div className="text-sm text-red-400 font-medium">Events</div>
            <div className="text-2xl font-bold text-slate-100">{results.n_events || 0}</div>
          </div>
          <div className="stat-card green">
            <div className="text-sm text-emerald-400 font-medium">Censored</div>
            <div className="text-2xl font-bold text-slate-100">{(results.n_observations || 0) - (results.n_events || 0)}</div>
          </div>
        </div>
        {Object.keys(coefficients).length > 0 && (
          <div>
            <h4 className="font-semibold text-slate-100 mb-3">Hazard Ratios</h4>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Variable</th>
                    <th className="text-right">HR</th>
                    <th className="text-right">95% CI</th>
                    <th className="text-right">p-value</th>
                    <th>Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(coefficients).map(([varName, stats]: [string, any]) => {
                    const pInterp = interpretPValue(stats.p_value);
                    return (
                      <tr key={varName} className={pInterp.significant ? 'bg-emerald-500/10' : ''}>
                        <td className="font-medium text-slate-200">{varName}</td>
                        <td className="text-right font-mono">{stats.hazard_ratio?.toFixed(3)}</td>
                        <td className="text-right font-mono text-slate-400">({stats.hr_ci_lower?.toFixed(2)} - {stats.hr_ci_upper?.toFixed(2)})</td>
                        <td className="text-right">
                          <span className={clsx('badge', pInterp.significant ? 'badge-green' : 'bg-slate-700 text-slate-300')}>
                            {stats.p_value < 0.001 ? '<0.001' : stats.p_value?.toFixed(3)}
                          </span>
                        </td>
                        <td className="text-sm text-slate-400">{pInterp.significant ? interpretHR(stats.hazard_ratio) : 'Not significant'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {analysis.interpretation && (
          <div className="glass-card p-4 border-cyan-500/30 bg-cyan-500/5">
            <h4 className="font-semibold text-cyan-400 mb-2">Interpretation</h4>
            <div className="text-sm text-slate-300 whitespace-pre-wrap">{renderMarkdownBold(analysis.interpretation)}</div>
          </div>
        )}
      </div>
    );
  };

  const renderLogisticResults = (analysis: any) => {
    const results = analysis.results || {};
    const coefficients = results.coefficients || {};

    // Filter out intercept and problematic coefficients
    const filteredCoefficients = Object.entries(coefficients).filter(([varName, stats]: [string, any]) => {
      // Skip intercept
      if (varName.toLowerCase().includes('intercept') || varName.toLowerCase() === 'const') return false;
      // Skip coefficients with extreme OR values (perfect separation)
      const or = stats.odds_ratio || Math.exp(stats.coefficient || 0);
      if (or > 1000 || or < 0.001 || !isFinite(or)) return false;
      // Skip coefficients with absurdly wide CIs
      const ciLower = stats.or_ci_lower || Math.exp(stats.ci_lower || 0);
      const ciUpper = stats.or_ci_upper || Math.exp(stats.ci_upper || 0);
      if (ciUpper / ciLower > 10000) return false;
      return true;
    });

    return (
      <div className="space-y-6">
        {results.plot_base64 && (
          <div className="glass-card p-4">
            <h4 className="font-semibold text-slate-100 mb-3">Forest Plot: Odds Ratios</h4>
            <img src={`data:image/png;base64,${results.plot_base64}`} alt="Forest Plot" className="w-full max-w-3xl mx-auto rounded-lg" />
          </div>
        )}
        <div className="grid grid-cols-4 gap-4">
          <div className="stat-card blue"><div className="text-sm text-blue-400">Observations</div><div className="text-2xl font-bold text-slate-100">{results.n_observations || 0}</div></div>
          <div className="stat-card purple"><div className="text-sm text-purple-400">Predictors</div><div className="text-2xl font-bold text-slate-100">{results.n_predictors || 0}</div></div>
          <div className="stat-card orange"><div className="text-sm text-orange-400">Pseudo R²</div><div className="text-2xl font-bold text-slate-100">{(results.pseudo_r2 || 0).toFixed(3)}</div></div>
          <div className="stat-card green"><div className="text-sm text-emerald-400">AUC</div><div className="text-2xl font-bold text-slate-100">{(results.auc || 0).toFixed(3)}</div></div>
        </div>
        {filteredCoefficients.length > 0 && (
          <div>
            <h4 className="font-semibold text-slate-100 mb-3">Odds Ratios</h4>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead><tr><th>Variable</th><th className="text-right">OR</th><th className="text-right">95% CI</th><th className="text-right">p-value</th><th>Interpretation</th></tr></thead>
                <tbody>
                  {filteredCoefficients.map(([varName, stats]: [string, any]) => {
                    const pInterp = interpretPValue(stats.p_value);
                    const or = stats.odds_ratio || Math.exp(stats.coefficient || 0);
                    // Use OR-scale CIs if available, otherwise convert from log-scale
                    const ciLower = stats.or_ci_lower || Math.exp(stats.ci_lower || 0);
                    const ciUpper = stats.or_ci_upper || Math.exp(stats.ci_upper || 0);
                    return (
                      <tr key={varName} className={pInterp.significant ? 'bg-emerald-500/10' : ''}>
                        <td className="font-medium text-slate-200">{varName}</td>
                        <td className="text-right font-mono">{or?.toFixed(2)}</td>
                        <td className="text-right font-mono text-slate-400">({ciLower?.toFixed(2)} - {ciUpper?.toFixed(2)})</td>
                        <td className="text-right"><span className={clsx('badge', pInterp.significant ? 'badge-green' : 'bg-slate-700 text-slate-300')}>{stats.p_value < 0.001 ? '<0.001' : stats.p_value?.toFixed(3)}</span></td>
                        <td className="text-sm text-slate-400">{pInterp.significant ? interpretOR(or) : 'Not significant'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {analysis.interpretation && (
          <div className="glass-card p-4 border-purple-500/30 bg-purple-500/5">
            <h4 className="font-semibold text-purple-400 mb-2">Interpretation</h4>
            <div className="text-sm text-slate-300 whitespace-pre-wrap">{renderMarkdownBold(analysis.interpretation)}</div>
          </div>
        )}
      </div>
    );
  };

  const renderChiSquareResults = (analysis: any) => {
    const results = analysis.results || {};
    const pInterp = interpretPValue(results.p_value);
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-3 gap-4">
          <div className="stat-card blue"><div className="text-sm text-blue-400">Chi-Square</div><div className="text-2xl font-bold text-slate-100">{results.statistic?.toFixed(3)}</div></div>
          <div className="stat-card purple"><div className="text-sm text-purple-400">DF</div><div className="text-2xl font-bold text-slate-100">{results.df || results.degrees_of_freedom}</div></div>
          <div className={clsx('stat-card', pInterp.significant ? 'green' : '')}><div className={clsx('text-sm', pInterp.significant ? 'text-emerald-400' : 'text-slate-400')}>p-value</div><div className={clsx('text-2xl font-bold', pInterp.significant ? 'text-emerald-400' : 'text-slate-100')}>{results.p_value < 0.001 ? '<0.001' : results.p_value?.toFixed(4)}</div></div>
        </div>
        {(results.cramers_v || results.effect_size) && (
          <div className="stat-card orange"><div className="text-sm text-orange-400">Effect Size (Cramér's V)</div><div className="text-2xl font-bold text-slate-100">{(results.cramers_v || results.effect_size)?.toFixed(3)}<span className="text-sm font-normal ml-2 text-slate-400">({(results.cramers_v || results.effect_size) < 0.1 ? 'Negligible' : (results.cramers_v || results.effect_size) < 0.3 ? 'Small' : (results.cramers_v || results.effect_size) < 0.5 ? 'Medium' : 'Large'})</span></div></div>
        )}
        <div className={clsx('glass-card p-4', pInterp.significant ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-slate-700')}>
          <div className="flex items-start space-x-3">
            {pInterp.significant ? <CheckCircle className="w-5 h-5 text-emerald-400 mt-0.5" /> : <XCircle className="w-5 h-5 text-slate-500 mt-0.5" />}
            <div>
              <h4 className={clsx('font-semibold mb-1', pInterp.significant ? 'text-emerald-400' : 'text-slate-300')}>{pInterp.text}</h4>
              <p className="text-sm text-slate-400">{pInterp.significant ? 'There is a statistically significant association between the variables.' : 'No statistically significant association was found.'}</p>
            </div>
          </div>
        </div>
        {analysis.interpretation && (
          <div className="glass-card p-4 border-cyan-500/30 bg-cyan-500/5">
            <h4 className="font-semibold text-cyan-400 mb-2">Detailed Interpretation</h4>
            <div className="text-sm text-slate-300 whitespace-pre-wrap">{renderMarkdownBold(analysis.interpretation)}</div>
          </div>
        )}
      </div>
    );
  };

  const renderTable1Results = (analysis: any) => {
    const results = analysis.results || {};
    const tableHtml = results.table_html;

    return (
      <div className="space-y-4">
        {tableHtml && (
          <div className="table-html-container">
            <div
              className="analysis-result-content"
              dangerouslySetInnerHTML={{ __html: tableHtml }}
            />
          </div>
        )}
        {!tableHtml && results.table_data && (
          <div className="overflow-x-auto">
            <pre className="text-xs text-slate-300 bg-slate-800/50 p-4 rounded-xl overflow-x-auto">
              {JSON.stringify(results.table_data, null, 2)}
            </pre>
          </div>
        )}
        {analysis.interpretation && (
          <div className="glass-card p-4 border-cyan-500/30 bg-cyan-500/5">
            <h4 className="font-semibold text-cyan-400 mb-2">Interpretation</h4>
            <div className="text-sm text-slate-300 whitespace-pre-wrap">{renderMarkdownBold(analysis.interpretation)}</div>
          </div>
        )}
      </div>
    );
  };

  const renderGenericResults = (analysis: any) => {
    const results = analysis.results || {};

    // Check if this is a Table 1 result
    if (results.table_html || results.table_data) {
      return renderTable1Results(analysis);
    }

    return (
      <div className="space-y-4">
        {/* Display any plots */}
        {results.plot_base64 && (
          <div className="glass-card p-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
            <h4 className="font-semibold text-slate-100 mb-3">Visualization</h4>
            <div className="bg-white rounded-lg p-2">
              <img
                src={`data:image/png;base64,${results.plot_base64}`}
                alt="Analysis Plot"
                className="w-full max-w-4xl mx-auto rounded-lg shadow-lg"
              />
            </div>
          </div>
        )}
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(results).map(([key, value]) => {
            if (typeof value === 'object' && value !== null) return null;
            if (key === 'plot_base64' || key === 'plot_error') return null;
            return (
              <div key={key} className="stat-card">
                <div className="text-sm text-slate-400 font-medium">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                <div className="text-lg font-bold text-slate-100">{formatValue(value)}</div>
              </div>
            );
          })}
        </div>
        {analysis.interpretation && (
          <div className="glass-card p-4 border-cyan-500/30 bg-cyan-500/5">
            <h4 className="font-semibold text-cyan-400 mb-2">Interpretation</h4>
            <div className="text-sm text-slate-300 whitespace-pre-wrap">{renderMarkdownBold(analysis.interpretation)}</div>
          </div>
        )}
      </div>
    );
  };

  const renderResults = (analysis: any) => {
    const type = analysis.analysis_type?.toLowerCase() || '';
    if (type.includes('table1') || type.includes('table_1') || type.includes('baseline')) return renderTable1Results(analysis);
    if (type.includes('kaplan') || type.includes('km') || type === 'log_rank') return renderKaplanMeierResults(analysis);
    if (type.includes('cox')) return renderCoxResults(analysis);
    if (type.includes('logistic')) return renderLogisticResults(analysis);
    if (type.includes('chi') || type.includes('square')) return renderChiSquareResults(analysis);
    return renderGenericResults(analysis);
  };

  if (showVisualizations) {
    return (
      <div className="h-full overflow-y-auto p-6">
        <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2 text-slate-100">
          <Image className="w-6 h-6 text-indigo-400" />
          <span>Visualizations</span>
        </h2>
        {vizSuggestions.length > 0 && (
          <div className="mb-8">
            <h3 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-4">Suggested Visualizations</h3>
            <div className="grid grid-cols-2 gap-4">
              {vizSuggestions.slice(0, 8).map((suggestion: any, i) => (
                <button key={i} onClick={() => createVisualization(suggestion.type, suggestion.params)} disabled={loading}
                  className="p-4 glass-card text-left hover:bg-slate-800/50 transition-colors disabled:opacity-50">
                  <div className="font-medium text-slate-200">{suggestion.type}</div>
                  <div className="text-sm text-slate-400">{suggestion.description}</div>
                </button>
              ))}
            </div>
          </div>
        )}
        {visualizations.size > 0 && (
          <div>
            <h3 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-4">Generated Figures</h3>
            <div className="space-y-6">
              {Array.from(visualizations.entries()).map(([key, base64]) => (
                <div key={key} className="glass-card p-4">
                  <img src={`data:image/png;base64,${base64}`} alt="Visualization" className="w-full rounded-lg" />
                </div>
              ))}
            </div>
          </div>
        )}
        {loading && <div className="flex items-center justify-center py-12"><Loader2 className="w-8 h-8 text-indigo-400 animate-spin" /></div>}
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold flex items-center space-x-2 text-slate-100">
          <FileText className="w-6 h-6 text-indigo-400" />
          <span>Analysis Results</span>
        </h2>
        {history.length > 0 && (
          <div className="flex items-center space-x-2">
            {resultsReport && (
              <button onClick={() => setShowReport(!showReport)} className={clsx('btn-ghost text-sm', showReport && 'bg-slate-800')}>
                {showReport ? 'Hide Report' : 'View Report'}
              </button>
            )}
            <button onClick={generateReport} disabled={generatingReport} className="btn-primary text-sm flex items-center gap-2">
              {generatingReport ? <><Loader2 className="w-4 h-4 animate-spin" /><span>Generating...</span></> : <><FileTextIcon className="w-4 h-4" /><span>Generate Report</span></>}
            </button>
          </div>
        )}
      </div>

      {showReport && resultsReport && (
        <div className="mb-8 glass-card overflow-hidden">
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white flex items-center space-x-2"><FileTextIcon className="w-5 h-5" /><span>Results Section</span></h3>
            <div className="flex items-center space-x-2">
              <button onClick={copyReportToClipboard} className="p-2 text-white/80 hover:text-white hover:bg-white/10 rounded-lg"><Copy className="w-4 h-4" /></button>
              <button onClick={downloadReport} className="p-2 text-white/80 hover:text-white hover:bg-white/10 rounded-lg"><Download className="w-4 h-4" /></button>
            </div>
          </div>
          <div className="p-6"><SimpleMarkdown content={resultsReport} /></div>
        </div>
      )}

      {history.length === 0 ? (
        <div className="text-center py-16">
          <div className="relative mx-auto w-20 h-20 mb-6">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl blur-xl opacity-20" />
            <div className="relative w-20 h-20 bg-slate-800 rounded-2xl flex items-center justify-center border border-slate-700">
              <BarChart3 className="w-10 h-10 text-slate-600" />
            </div>
          </div>
          <h3 className="text-lg font-medium text-slate-300 mb-2">No analyses yet</h3>
          <p className="text-sm text-slate-500">Ask a research question in the chat to get started</p>
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((analysis: any, i) => (
            <div key={analysis.analysis_id || i} className="glass-card overflow-hidden">
              {analysis.test_name && (
                <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-3">
                  <h3 className="text-lg font-semibold text-white">{analysis.test_name}</h3>
                  {analysis.rationale && <p className="text-indigo-200 text-sm mt-1">{analysis.rationale}</p>}
                </div>
              )}
              <button onClick={() => toggleExpand(analysis.analysis_id || String(i))} className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-800/30 transition-colors">
                <div className="flex items-center space-x-3">
                  <div className={clsx('w-3 h-3 rounded-full', analysis.results?.p_value < 0.05 ? 'bg-emerald-500' : 'bg-slate-600')} />
                  <div>
                    <div className="font-medium text-slate-200">{analysis.analysis_type?.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</div>
                    <div className="text-sm text-slate-500">{analysis.timestamp ? new Date(analysis.timestamp).toLocaleString() : ''}</div>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  {analysis.results?.p_value !== undefined && (
                    <span className={clsx('badge', analysis.results.p_value < 0.05 ? 'badge-green' : 'bg-slate-700 text-slate-300')}>
                      p = {analysis.results.p_value < 0.001 ? '<0.001' : analysis.results.p_value?.toFixed(4)}
                    </span>
                  )}
                  {expandedResults.has(analysis.analysis_id || String(i)) ? <ChevronDown className="w-5 h-5 text-slate-400" /> : <ChevronRight className="w-5 h-5 text-slate-400" />}
                </div>
              </button>
              {expandedResults.has(analysis.analysis_id || String(i)) && (
                <div className="px-6 py-6 border-t border-slate-700/50 bg-slate-900/30">
                  {renderResults(analysis)}

                  {/* Code Transparency Section */}
                  <div className="mt-6 pt-4 border-t border-slate-700/50">
                    <button
                      onClick={() => loadCodeForAnalysis(analysis.analysis_id || String(i), analysis.analysis_type)}
                      className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-cyan-400
                                 bg-cyan-500/10 border border-cyan-500/30 rounded-xl
                                 hover:bg-cyan-500/20 transition-colors"
                    >
                      <Code className="w-4 h-4" />
                      <span>{showCodeFor === (analysis.analysis_id || String(i)) ? 'Hide Code' : 'View Reproducible Code'}</span>
                    </button>

                    {showCodeFor === (analysis.analysis_id || String(i)) && analysisCode.has(analysis.analysis_id || String(i)) && (
                      <div className="mt-4">
                        <CodePanel
                          pythonCode={analysisCode.get(analysis.analysis_id || String(i))?.python}
                          rCode={analysisCode.get(analysis.analysis_id || String(i))?.r}
                          analysisType={analysis.analysis_type}
                          packages={analysisCode.get(analysis.analysis_id || String(i))?.packages}
                          isCollapsible={false}
                          defaultExpanded={true}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Recommended Visualizations Section */}
      {analysisPlan?.visualizations && analysisPlan.visualizations.length > 0 && (
        <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-100 flex items-center space-x-2">
              <Image className="w-5 h-5 text-indigo-400" />
              <span>Recommended Visualizations</span>
            </h3>
            <button
              onClick={generateAllVisualizations}
              disabled={loading}
              className="btn-primary text-sm flex items-center gap-2"
            >
              {loading ? (
                <><Loader2 className="w-4 h-4 animate-spin" /><span>Generating...</span></>
              ) : (
                <><Image className="w-4 h-4" /><span>Generate All</span></>
              )}
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {analysisPlan.visualizations.map((vizName, i) => {
              const { label, description, type, params } = parseRecommendedViz(vizName);
              const isGenerated = visualizations.has(vizName);
              return (
                <div key={i} className="glass-card p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-slate-200 truncate">{label}</div>
                      <div className="text-sm text-slate-400 mt-1">{description}</div>
                    </div>
                    <button
                      onClick={() => createVisualization(type, params)}
                      disabled={loading}
                      className={clsx(
                        'ml-3 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex-shrink-0',
                        isGenerated
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/30'
                      )}
                    >
                      {isGenerated ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : loading ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        'Generate'
                      )}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Generated Visualizations */}
      {visualizations.size > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" />
            <span>Generated Figures ({visualizations.size})</span>
          </h3>
          <div className="space-y-4">
            {Array.from(visualizations.entries()).map(([key, base64]) => (
              <div key={key} className="glass-card overflow-hidden">
                <div className="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
                  <span className="font-medium text-slate-200">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                  <button
                    onClick={() => {
                      const link = document.createElement('a');
                      link.href = `data:image/png;base64,${base64}`;
                      link.download = `${key}.png`;
                      link.click();
                    }}
                    className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors"
                  >
                    <Download className="w-4 h-4 text-slate-400" />
                  </button>
                </div>
                <div className="p-4">
                  <img src={`data:image/png;base64,${base64}`} alt={key} className="w-full rounded-lg" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
        </div>
      )}
    </div>
  );
}
