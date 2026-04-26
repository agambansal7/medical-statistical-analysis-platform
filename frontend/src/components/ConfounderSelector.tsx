import React, { useState, useEffect } from 'react';
import { useStore } from '../hooks/useStore';
import { chatApi } from '../services/api';
import {
  CheckCircle,
  XCircle,
  Loader2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Info,
  Filter,
  Zap,
  X,
} from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';

interface Confounder {
  variable: string;
  n_valid: number;
  missing_pct: number;
  type: string;
  test?: string;
  statistic?: number;
  p_value?: number;
  significant?: boolean;
  test_error?: string;
  univariate_or?: number;
  univariate_hr?: number;
  univariate_ci_lower?: number;
  univariate_ci_upper?: number;
}

interface ConfounderSelectorProps {
  outcome: string;
  outcomeType: 'binary' | 'continuous' | 'survival';
  mainPredictor?: string;
  timeCol?: string;
  eventCol?: string;
  onAnalysisComplete?: (result: any) => void;
  onClose?: () => void;
}

export default function ConfounderSelector({
  outcome,
  outcomeType,
  mainPredictor,
  timeCol,
  eventCol,
  onAnalysisComplete,
  onClose
}: ConfounderSelectorProps) {
  const { sessionId, addAnalysisResult } = useStore();
  const [confounders, setConfounders] = useState<Confounder[]>([]);
  const [selectedConfounders, setSelectedConfounders] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const [filterSignificant, setFilterSignificant] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadConfounders();
  }, [sessionId, outcome, outcomeType]);

  const loadConfounders = async () => {
    if (!sessionId) return;

    setLoading(true);
    try {
      // Pass timeCol and eventCol to exclude them from the variable list
      const response = await chatApi.getConfounders(
        sessionId,
        outcome,
        outcomeType,
        timeCol,
        eventCol
      );
      if (response.success) {
        setConfounders(response.confounders);
        // Auto-select significant confounders
        const significant = response.confounders
          .filter(c => c.significant)
          .map(c => c.variable);
        setSelectedConfounders(new Set(significant));
      }
    } catch (error) {
      console.error('Failed to load confounders:', error);
      toast.error('Failed to load variable analysis');
    } finally {
      setLoading(false);
    }
  };

  const toggleConfounder = (variable: string) => {
    const newSelected = new Set(selectedConfounders);
    if (newSelected.has(variable)) {
      newSelected.delete(variable);
    } else {
      newSelected.add(variable);
    }
    setSelectedConfounders(newSelected);
  };

  const selectAll = () => {
    const filtered = getFilteredConfounders();
    setSelectedConfounders(new Set(filtered.map(c => c.variable)));
  };

  const selectSignificant = () => {
    setSelectedConfounders(new Set(confounders.filter(c => c.significant).map(c => c.variable)));
  };

  const selectClinicallyRelevant = () => {
    // Select common clinical confounders
    const clinicalVars = ['age', 'sex', 'gender', 'bmi', 'diabetes', 'hypertension', 'smoking', 'ckd', 'race', 'ethnicity'];
    const relevant = confounders.filter(c =>
      clinicalVars.some(cv => c.variable.toLowerCase().includes(cv))
    );
    setSelectedConfounders(prev => new Set([...prev, ...relevant.map(r => r.variable)]));
  };

  const clearAll = () => {
    setSelectedConfounders(new Set());
  };

  const getFilteredConfounders = () => {
    let filtered = confounders;

    if (filterSignificant) {
      filtered = filtered.filter(c => c.significant);
    }

    if (searchTerm) {
      filtered = filtered.filter(c =>
        c.variable.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    return filtered;
  };

  const runAdjustedAnalysis = async () => {
    if (!sessionId) return;

    const analysisType = outcomeType === 'survival' ? 'cox_regression' : 'logistic_regression';

    setRunning(true);
    try {
      const response = await chatApi.runAdjustedAnalysis(sessionId, {
        analysis_type: analysisType,
        outcome: outcomeType === 'survival' ? '' : outcome,
        predictors: mainPredictor ? [mainPredictor] : [],
        confounders: Array.from(selectedConfounders),
        time: timeCol,
        event: eventCol
      });

      if (response.success) {
        toast.success('Adjusted analysis completed!');
        if (onAnalysisComplete) {
          onAnalysisComplete(response);
        }
        addAnalysisResult(response as any);
      } else {
        toast.error(response.error || 'Analysis failed');
      }
    } catch (error: any) {
      console.error('Analysis failed:', error);
      toast.error(error.response?.data?.detail || 'Analysis failed');
    } finally {
      setRunning(false);
    }
  };

  const formatPValue = (p: number | undefined) => {
    if (p === undefined) return '-';
    if (p < 0.001) return '<0.001';
    return p.toFixed(3);
  };

  const filteredConfounders = getFilteredConfounders();
  const significantCount = confounders.filter(c => c.significant).length;

  if (loading) {
    return (
      <div className="glass-card p-6">
        <div className="flex items-center justify-center space-x-3">
          <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
          <span className="text-slate-300">Analyzing variables with univariate tests...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Filter className="w-5 h-5" />
              Select Variables for Adjusted Analysis
            </h3>
            <p className="text-indigo-200 text-sm mt-1">
              {selectedConfounders.size} of {confounders.length} variables selected
              {significantCount > 0 && (
                <span className="ml-2">
                  • <span className="text-emerald-300">{significantCount} significant</span>
                </span>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setExpanded(!expanded)}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              {expanded ? (
                <ChevronUp className="w-5 h-5 text-white" />
              ) : (
                <ChevronDown className="w-5 h-5 text-white" />
              )}
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-white" />
              </button>
            )}
          </div>
        </div>
      </div>

      {expanded && (
        <>
          {/* Info Banner */}
          <div className="bg-cyan-500/10 border-b border-cyan-500/20 px-6 py-3">
            <div className="flex items-start space-x-3">
              <Info className="w-5 h-5 text-cyan-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-cyan-300">
                <strong>Univariate Analysis Results:</strong> Each variable shows its p-value from
                testing association with <code className="bg-cyan-500/20 px-1.5 py-0.5 rounded text-cyan-200">{outcome || 'outcome'}</code>.
                Significant variables (p &lt; 0.05) are highlighted in green.
                Select variables to include in your adjusted model.
              </div>
            </div>
          </div>

          {/* Search and Filters */}
          <div className="px-6 py-4 border-b border-slate-700/50 space-y-3">
            {/* Search */}
            <div className="relative">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search variables..."
                className="w-full px-4 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500"
              />
            </div>

            {/* Quick Actions */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-slate-500 uppercase tracking-wider">Quick select:</span>
              <button
                onClick={selectSignificant}
                className="px-3 py-1.5 text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded-lg hover:bg-emerald-500/30 transition-colors flex items-center gap-1"
              >
                <Zap className="w-3 h-3" />
                Significant (p&lt;0.05)
              </button>
              <button
                onClick={selectClinicallyRelevant}
                className="px-3 py-1.5 text-xs bg-purple-500/20 text-purple-400 border border-purple-500/30 rounded-lg hover:bg-purple-500/30 transition-colors"
              >
                + Clinical Confounders
              </button>
              <button
                onClick={selectAll}
                className="px-3 py-1.5 text-xs bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors"
              >
                Select All
              </button>
              <button
                onClick={clearAll}
                className="px-3 py-1.5 text-xs bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors"
              >
                Clear All
              </button>
              <div className="ml-auto">
                <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterSignificant}
                    onChange={(e) => setFilterSignificant(e.target.checked)}
                    className="rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500 focus:ring-offset-slate-900"
                  />
                  Show significant only
                </label>
              </div>
            </div>
          </div>

          {/* Variables Table */}
          <div className="overflow-x-auto max-h-80 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-800/80 sticky top-0 z-10">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-slate-300 w-12"></th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-300">Variable</th>
                  <th className="px-4 py-3 text-center font-semibold text-slate-300 w-24">Type</th>
                  <th className="px-4 py-3 text-center font-semibold text-slate-300 w-24">Missing</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-300 w-28">Test</th>
                  <th className="px-4 py-3 text-center font-semibold text-slate-300 w-28">
                    Univariate<br/>P-value
                  </th>
                  <th className="px-4 py-3 text-center font-semibold text-slate-300 w-16">Sig.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {filteredConfounders.map((conf) => (
                  <tr
                    key={conf.variable}
                    className={clsx(
                      'cursor-pointer transition-colors',
                      selectedConfounders.has(conf.variable)
                        ? 'bg-indigo-500/20 hover:bg-indigo-500/30'
                        : 'hover:bg-slate-800/50',
                      conf.significant && 'border-l-2 border-l-emerald-500'
                    )}
                    onClick={() => toggleConfounder(conf.variable)}
                  >
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selectedConfounders.has(conf.variable)}
                        onChange={() => toggleConfounder(conf.variable)}
                        onClick={(e) => e.stopPropagation()}
                        className="rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500 focus:ring-offset-slate-900"
                      />
                    </td>
                    <td className="px-4 py-3">
                      <span className={clsx(
                        'font-medium',
                        selectedConfounders.has(conf.variable) ? 'text-slate-100' : 'text-slate-300'
                      )}>
                        {conf.variable}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={clsx(
                        'px-2 py-0.5 rounded-full text-xs font-medium',
                        conf.type === 'continuous'
                          ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                          : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                      )}>
                        {conf.type}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={clsx(
                        'text-sm',
                        conf.missing_pct > 20 ? 'text-red-400' :
                        conf.missing_pct > 10 ? 'text-amber-400' : 'text-slate-400'
                      )}>
                        {conf.missing_pct.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-400 text-xs">
                      {conf.test || '-'}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={clsx(
                        'font-mono text-sm font-medium',
                        conf.significant ? 'text-emerald-400' : 'text-slate-400'
                      )}>
                        {formatPValue(conf.p_value)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      {conf.p_value !== undefined ? (
                        conf.significant ? (
                          <CheckCircle className="w-5 h-5 text-emerald-500 mx-auto" />
                        ) : (
                          <XCircle className="w-5 h-5 text-slate-600 mx-auto" />
                        )
                      ) : (
                        <span className="text-slate-600">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filteredConfounders.length === 0 && (
            <div className="px-6 py-8 text-center text-slate-400">
              No variables match your filter criteria.
            </div>
          )}

          {/* Footer */}
          <div className="px-6 py-4 bg-slate-800/50 border-t border-slate-700/50 flex items-center justify-between">
            <div className="text-sm text-slate-400">
              <strong className="text-slate-200">{selectedConfounders.size}</strong> variables selected for adjustment
              {mainPredictor && (
                <span className="ml-3 text-indigo-400">
                  Main predictor: <code className="bg-indigo-500/20 px-1.5 py-0.5 rounded">{mainPredictor}</code>
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              {onClose && (
                <button
                  onClick={onClose}
                  className="px-4 py-2 text-slate-300 bg-slate-700 border border-slate-600 rounded-lg hover:bg-slate-600 transition-colors"
                >
                  Cancel
                </button>
              )}
              <button
                onClick={runAdjustedAnalysis}
                disabled={running || selectedConfounders.size === 0}
                className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-indigo-500/25"
              >
                {running ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Running Analysis...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    <span>Run Adjusted Analysis</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
