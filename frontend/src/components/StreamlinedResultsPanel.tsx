import { useState, useEffect } from 'react';
import { useStore } from '../hooks/useStore';
import { analysisApi, chatApi, codeApi } from '../services/api';
import {
  FileText,
  ChevronDown,
  ChevronRight,
  Loader2,
  Download,
  Copy,
  Code,
  BarChart3,
  Info,
} from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import CodePanel from './CodePanel';
import InteractivePlot from './InteractivePlot';

interface AnalysisResult {
  analysis_id?: string;
  analysis_type: string;
  test_name?: string;
  rationale?: string;
  timestamp?: string;
  results?: {
    p_value?: number;
    statistic?: number;
    plot_base64?: string;
    plotly_json?: string;
    interpretation?: string;
    coefficients?: Record<string, any>;
    km_results?: Record<string, any>;
    log_rank?: { p_value?: number };
    n_observations?: number;
    n_events?: number;
    auc?: number;
    pseudo_r2?: number;
    table_html?: string;
    [key: string]: any;
  };
  interpretation?: string;
}

// Compact stat badge component
function StatBadge({
  label,
  value,
  variant = 'default',
  small = false
}: {
  label: string;
  value: string | number;
  variant?: 'default' | 'success' | 'warning' | 'info';
  small?: boolean;
}) {
  const colors = {
    default: 'bg-slate-700/50 text-slate-300',
    success: 'bg-emerald-500/20 text-emerald-400',
    warning: 'bg-amber-500/20 text-amber-400',
    info: 'bg-indigo-500/20 text-indigo-400',
  };

  return (
    <div className={clsx(
      'flex items-center gap-2 px-3 py-1.5 rounded-lg',
      colors[variant],
      small && 'text-xs'
    )}>
      <span className="text-slate-500">{label}:</span>
      <span className="font-mono font-medium">{value}</span>
    </div>
  );
}


export default function StreamlinedResultsPanel() {
  const { sessionId, analysisResults, resultsReport, setResultsReport } = useStore();
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [showCodeFor, setShowCodeFor] = useState<string | null>(null);
  const [analysisCode, setAnalysisCode] = useState<Map<string, any>>(new Map());

  useEffect(() => {
    const loadHistory = async () => {
      if (!sessionId) return;
      setLoading(true);
      try {
        const data = await analysisApi.getHistory(sessionId);
        setHistory(data);
        // Auto-expand only the most recent result
        if (data.length > 0) {
          const latestId = data[0]?.analysis_id || '0';
          setExpandedResults(new Set([latestId]));
        }
      } catch (error) {
        console.error('Failed to load history', error);
      } finally {
        setLoading(false);
      }
    };
    loadHistory();
  }, [sessionId, analysisResults]);

  const toggleExpand = (id: string) => {
    setExpandedResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const generateReport = async () => {
    if (!sessionId || history.length === 0) {
      toast.error('No analyses to generate report from');
      return;
    }
    setGeneratingReport(true);
    try {
      const response = await chatApi.generateReport(sessionId);
      if (response.success && response.report) {
        setResultsReport(response.report);
        setShowReport(true);
        toast.success('Report generated!');
      }
    } catch (error: any) {
      toast.error('Failed to generate report');
    } finally {
      setGeneratingReport(false);
    }
  };

  const formatPValue = (p: number | undefined): string => {
    if (p === undefined) return '—';
    if (p < 0.001) return '<0.001';
    if (p < 0.01) return p.toFixed(3);
    return p.toFixed(4);
  };

  const isSignificant = (p: number | undefined): boolean => {
    return p !== undefined && p < 0.05;
  };

  const loadCode = async (analysisId: string, analysisType: string) => {
    if (analysisCode.has(analysisId)) {
      setShowCodeFor(showCodeFor === analysisId ? null : analysisId);
      return;
    }
    try {
      const result = await codeApi.generate(sessionId!, analysisType, {});
      if (result.success) {
        setAnalysisCode(prev => new Map(prev.set(analysisId, {
          python: result.python_code,
          r: result.r_code,
          packages: { python: result.python_packages, r: result.r_packages },
        })));
        setShowCodeFor(analysisId);
      }
    } catch (error) {
      toast.error('Failed to load code');
    }
  };

  // Render compact analysis result
  const renderAnalysisResult = (analysis: AnalysisResult) => {
    const results = analysis.results || {};
    const type = analysis.analysis_type?.toLowerCase() || '';
    const pValue = results.p_value ?? results.log_rank?.p_value;
    const significant = isSignificant(pValue);

    // Determine key metrics based on analysis type
    let keyMetrics: Array<{ label: string; value: string; variant?: 'default' | 'success' | 'warning' | 'info' }> = [];

    if (type.includes('kaplan') || type.includes('km') || type === 'log_rank') {
      keyMetrics = [
        { label: 'n', value: String(results.n_observations || '—') },
        { label: 'Events', value: String(results.n_events || '—') },
        { label: 'Log-rank p', value: formatPValue(results.log_rank?.p_value), variant: isSignificant(results.log_rank?.p_value) ? 'success' : 'default' },
      ];
    } else if (type.includes('cox')) {
      keyMetrics = [
        { label: 'n', value: String(results.n_observations || '—') },
        { label: 'Events', value: String(results.n_events || '—') },
      ];
    } else if (type.includes('logistic')) {
      keyMetrics = [
        { label: 'n', value: String(results.n_observations || '—') },
        { label: 'AUC', value: results.auc?.toFixed(3) || '—', variant: 'info' },
        { label: 'R²', value: results.pseudo_r2?.toFixed(3) || '—' },
      ];
    } else if (type.includes('chi')) {
      keyMetrics = [
        { label: 'χ²', value: results.statistic?.toFixed(2) || '—' },
        { label: 'df', value: String(results.df || results.degrees_of_freedom || '—') },
      ];
    }

    return (
      <div className="space-y-4">
        {/* Key metrics bar */}
        {keyMetrics.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {keyMetrics.map((m, i) => (
              <StatBadge key={i} label={m.label} value={m.value} variant={m.variant} small />
            ))}
            {pValue !== undefined && (
              <StatBadge
                label="p"
                value={formatPValue(pValue)}
                variant={significant ? 'success' : 'default'}
                small
              />
            )}
          </div>
        )}

        {/* Visualization - primary focus */}
        {(results.plotly_json || results.plot_base64) && (
          <InteractivePlot
            plotlyJson={results.plotly_json}
            base64Image={results.plot_base64}
            compact
          />
        )}

        {/* Table 1 HTML */}
        {results.table_html && (
          <div className="overflow-x-auto bg-white rounded-xl p-4">
            <div dangerouslySetInnerHTML={{ __html: results.table_html }} />
          </div>
        )}

        {/* Coefficient table for regression - simplified */}
        {results.coefficients && Object.keys(results.coefficients).length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-2 px-3 text-slate-400 font-medium">Variable</th>
                  <th className="text-right py-2 px-3 text-slate-400 font-medium">
                    {type.includes('cox') ? 'HR' : 'OR'}
                  </th>
                  <th className="text-right py-2 px-3 text-slate-400 font-medium">95% CI</th>
                  <th className="text-right py-2 px-3 text-slate-400 font-medium">p</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results.coefficients)
                  .filter(([name, _]) => !name.toLowerCase().includes('intercept') && name.toLowerCase() !== 'const')
                  .slice(0, 10)
                  .map(([name, stats]: [string, any]) => {
                    const estimate = stats.hazard_ratio || stats.odds_ratio || Math.exp(stats.coefficient || 0);
                    const ciLower = stats.hr_ci_lower || stats.or_ci_lower || Math.exp(stats.ci_lower || 0);
                    const ciUpper = stats.hr_ci_upper || stats.or_ci_upper || Math.exp(stats.ci_upper || 0);
                    const pVal = stats.p_value;
                    const isSig = isSignificant(pVal);

                    // Skip invalid values
                    if (!isFinite(estimate) || estimate > 1000 || estimate < 0.001) return null;

                    return (
                      <tr key={name} className={clsx('border-b border-slate-800', isSig && 'bg-emerald-500/5')}>
                        <td className="py-2 px-3 text-slate-200">{name}</td>
                        <td className="py-2 px-3 text-right font-mono text-slate-300">
                          {estimate.toFixed(2)}
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-slate-500">
                          ({ciLower.toFixed(2)}–{ciUpper.toFixed(2)})
                        </td>
                        <td className={clsx(
                          'py-2 px-3 text-right font-mono',
                          isSig ? 'text-emerald-400' : 'text-slate-400'
                        )}>
                          {formatPValue(pVal)}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        )}

        {/* KM group summary */}
        {results.km_results && Object.keys(results.km_results).length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(results.km_results).map(([group, data]: [string, any]) => (
              <div key={group} className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                <div className="text-xs text-slate-500 uppercase tracking-wider">{group}</div>
                <div className="text-lg font-semibold text-slate-200 mt-1">
                  {data.median_survival != null ? `${data.median_survival.toFixed(1)}` : 'NR'}
                </div>
                <div className="text-xs text-slate-500">
                  median survival (n={data.n_observations}, {data.n_events} events)
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Interpretation - collapsible, concise */}
        {(analysis.interpretation || results.interpretation) && (
          <details className="group">
            <summary className="flex items-center gap-2 cursor-pointer text-sm text-slate-400 hover:text-slate-300">
              <Info className="w-4 h-4" />
              <span>Interpretation</span>
              <ChevronRight className="w-4 h-4 transition-transform group-open:rotate-90" />
            </summary>
            <div className="mt-2 pl-6 text-sm text-slate-400 leading-relaxed">
              {(analysis.interpretation || results.interpretation || '').slice(0, 500)}
              {(analysis.interpretation || results.interpretation || '').length > 500 && '...'}
            </div>
          </details>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur-sm border-b border-slate-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Results</h2>
              <p className="text-sm text-slate-500">
                {history.length} {history.length === 1 ? 'analysis' : 'analyses'} completed
              </p>
            </div>
          </div>
          {history.length > 0 && (
            <button
              onClick={generateReport}
              disabled={generatingReport}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              {generatingReport ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <FileText className="w-4 h-4" />
              )}
              Generate Report
            </button>
          )}
        </div>
      </div>

      {/* Report section */}
      {showReport && resultsReport && (
        <div className="mx-6 mt-6 p-6 bg-slate-800/50 rounded-xl border border-slate-700/50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-slate-200">Generated Report</h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  navigator.clipboard.writeText(resultsReport);
                  toast.success('Copied!');
                }}
                className="p-2 hover:bg-slate-700 rounded-lg text-slate-400"
              >
                <Copy className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  const blob = new Blob([resultsReport], { type: 'text/markdown' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = 'results.md';
                  a.click();
                }}
                className="p-2 hover:bg-slate-700 rounded-lg text-slate-400"
              >
                <Download className="w-4 h-4" />
              </button>
              <button
                onClick={() => setShowReport(false)}
                className="text-sm text-slate-400 hover:text-slate-300"
              >
                Hide
              </button>
            </div>
          </div>
          <div className="prose prose-invert prose-sm max-w-none">
            <pre className="whitespace-pre-wrap text-slate-300 text-sm">{resultsReport}</pre>
          </div>
        </div>
      )}

      {/* Results list */}
      <div className="p-6 space-y-4">
        {history.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-800 flex items-center justify-center">
              <BarChart3 className="w-8 h-8 text-slate-600" />
            </div>
            <h3 className="text-slate-300 font-medium mb-2">No results yet</h3>
            <p className="text-sm text-slate-500">
              Ask a research question in the chat to run analyses
            </p>
          </div>
        ) : (
          history.map((analysis, i) => {
            const id = analysis.analysis_id || String(i);
            const isExpanded = expandedResults.has(id);
            const pValue = analysis.results?.p_value ?? analysis.results?.log_rank?.p_value;
            const significant = isSignificant(pValue);

            return (
              <div key={id} className="rounded-xl border border-slate-700/50 overflow-hidden bg-slate-800/20">
                {/* Analysis header */}
                <button
                  onClick={() => toggleExpand(id)}
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-800/30 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {/* Significance indicator */}
                    <div className={clsx(
                      'w-2 h-2 rounded-full flex-shrink-0',
                      significant ? 'bg-emerald-500' : 'bg-slate-600'
                    )} />

                    {/* Analysis info */}
                    <div className="text-left">
                      <div className="font-medium text-slate-200">
                        {analysis.test_name || analysis.analysis_type?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </div>
                      {analysis.rationale && (
                        <div className="text-xs text-slate-500 mt-0.5 line-clamp-1">
                          {analysis.rationale}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    {/* Quick p-value badge */}
                    {pValue !== undefined && (
                      <span className={clsx(
                        'px-2 py-1 rounded text-xs font-mono',
                        significant
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : 'bg-slate-700/50 text-slate-400'
                      )}>
                        p = {formatPValue(pValue)}
                      </span>
                    )}

                    {/* Expand icon */}
                    {isExpanded ? (
                      <ChevronDown className="w-5 h-5 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-slate-400" />
                    )}
                  </div>
                </button>

                {/* Expanded content */}
                {isExpanded && (
                  <div className="px-4 pb-4 pt-2 border-t border-slate-700/50">
                    {renderAnalysisResult(analysis)}

                    {/* Code section */}
                    <div className="mt-4 pt-4 border-t border-slate-700/30">
                      <button
                        onClick={() => loadCode(id, analysis.analysis_type)}
                        className="flex items-center gap-2 text-sm text-cyan-400 hover:text-cyan-300"
                      >
                        <Code className="w-4 h-4" />
                        {showCodeFor === id ? 'Hide Code' : 'View Reproducible Code'}
                      </button>

                      {showCodeFor === id && analysisCode.has(id) && (
                        <div className="mt-3">
                          <CodePanel
                            pythonCode={analysisCode.get(id)?.python}
                            rCode={analysisCode.get(id)?.r}
                            analysisType={analysis.analysis_type}
                            packages={analysisCode.get(id)?.packages}
                            isCollapsible={false}
                            defaultExpanded={true}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
