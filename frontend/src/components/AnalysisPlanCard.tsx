import { useState } from 'react';
import {
  ChevronRight,
  Play,
  Check,
  X,
  AlertCircle,
  Loader2,
  BarChart3,
  TrendingUp,
  Activity,
  FileText,
  Trash2,
} from 'lucide-react';
import clsx from 'clsx';
import AdjustmentStrategySelector, { AdjustmentStrategy } from './AdjustmentStrategySelector';

interface AnalysisConfig {
  id: string;
  test_name: string;
  analysis_type: string;
  rationale: string;
  outcome?: string;
  outcome_type?: 'binary' | 'continuous' | 'survival';
  predictor?: string;
  time_col?: string;
  event_col?: string;
  parameters?: Record<string, any>;
  requires_adjustment?: boolean;
}

interface AnalysisPlanCardProps {
  analysis: AnalysisConfig;
  sessionId: string;
  index: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  onRun: (analysis: AnalysisConfig, confounders: string[]) => void;
  onRemove: (id: string) => void;
  onToggleInclude: (id: string, include: boolean) => void;
  isIncluded: boolean;
  result?: any;
}

const ANALYSIS_ICONS: Record<string, typeof BarChart3> = {
  table1: FileText,
  chi_square: BarChart3,
  ttest: TrendingUp,
  logistic_regression: TrendingUp,
  cox_regression: Activity,
  kaplan_meier: Activity,
  linear_regression: TrendingUp,
  anova: BarChart3,
  default: BarChart3,
};

export default function AnalysisPlanCard({
  analysis,
  sessionId,
  index,
  status,
  onRun,
  onRemove,
  onToggleInclude,
  isIncluded,
  result,
}: AnalysisPlanCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [adjustmentStrategy, setAdjustmentStrategy] = useState<AdjustmentStrategy>('clinical');
  const [selectedConfounders, setSelectedConfounders] = useState<string[]>([]);

  const needsAdjustment = analysis.requires_adjustment ||
    ['logistic_regression', 'cox_regression', 'linear_regression'].includes(analysis.analysis_type);

  const Icon = ANALYSIS_ICONS[analysis.analysis_type] || ANALYSIS_ICONS.default;

  const handleStrategyChange = (strategy: AdjustmentStrategy, variables: string[]) => {
    setAdjustmentStrategy(strategy);
    setSelectedConfounders(variables);
  };

  const handleRun = () => {
    onRun(analysis, needsAdjustment ? selectedConfounders : []);
  };

  const getStatusBadge = () => {
    switch (status) {
      case 'running':
        return (
          <span className="flex items-center gap-1.5 px-2 py-1 bg-indigo-500/20 text-indigo-400 rounded-lg text-xs">
            <Loader2 className="w-3 h-3 animate-spin" />
            Running
          </span>
        );
      case 'completed':
        return (
          <span className="flex items-center gap-1.5 px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-lg text-xs">
            <Check className="w-3 h-3" />
            Complete
          </span>
        );
      case 'failed':
        return (
          <span className="flex items-center gap-1.5 px-2 py-1 bg-red-500/20 text-red-400 rounded-lg text-xs">
            <AlertCircle className="w-3 h-3" />
            Failed
          </span>
        );
      case 'skipped':
        return (
          <span className="flex items-center gap-1.5 px-2 py-1 bg-slate-500/20 text-slate-400 rounded-lg text-xs">
            <X className="w-3 h-3" />
            Skipped
          </span>
        );
      default:
        return null;
    }
  };

  const formatAnalysisType = (type: string) => {
    return type
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <div
      className={clsx(
        'border rounded-xl overflow-hidden transition-all',
        isIncluded
          ? 'border-slate-700/50 bg-slate-800/30'
          : 'border-slate-700/30 bg-slate-800/10 opacity-60',
        status === 'completed' && 'border-emerald-500/30',
        status === 'failed' && 'border-red-500/30'
      )}
    >
      {/* Header */}
      <div
        className={clsx(
          'flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors',
          isIncluded ? 'hover:bg-slate-700/30' : 'hover:bg-slate-700/20'
        )}
        onClick={() => setExpanded(!expanded)}
      >
        {/* Include Checkbox */}
        <input
          type="checkbox"
          checked={isIncluded}
          onChange={(e) => {
            e.stopPropagation();
            onToggleInclude(analysis.id, e.target.checked);
          }}
          className="rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500 focus:ring-offset-slate-900"
        />

        {/* Index */}
        <span className="w-6 h-6 flex items-center justify-center bg-slate-700/50 rounded-lg text-xs text-slate-400 font-medium">
          {index + 1}
        </span>

        {/* Icon */}
        <div className={clsx(
          'w-8 h-8 rounded-lg flex items-center justify-center',
          status === 'completed' ? 'bg-emerald-500/20' :
          status === 'failed' ? 'bg-red-500/20' :
          'bg-indigo-500/20'
        )}>
          <Icon className={clsx(
            'w-4 h-4',
            status === 'completed' ? 'text-emerald-400' :
            status === 'failed' ? 'text-red-400' :
            'text-indigo-400'
          )} />
        </div>

        {/* Title */}
        <div className="flex-1 min-w-0">
          <div className="font-medium text-slate-200 truncate">
            {analysis.test_name}
          </div>
          <div className="text-xs text-slate-500 truncate">
            {formatAnalysisType(analysis.analysis_type)}
            {analysis.outcome && ` • ${analysis.outcome}`}
          </div>
        </div>

        {/* Status Badge */}
        {getStatusBadge()}

        {/* Expand Arrow */}
        <ChevronRight className={clsx(
          'w-4 h-4 text-slate-500 transition-transform',
          expanded && 'rotate-90'
        )} />
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="border-t border-slate-700/50 px-4 py-3 space-y-4">
          {/* Rationale */}
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">
              Rationale
            </div>
            <p className="text-sm text-slate-300">
              {analysis.rationale}
            </p>
          </div>

          {/* Parameters */}
          {analysis.parameters && Object.keys(analysis.parameters).length > 0 && (
            <div>
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">
                Parameters
              </div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(analysis.parameters).map(([key, value]) => (
                  <span
                    key={key}
                    className="px-2 py-1 bg-slate-700/50 rounded-lg text-xs text-slate-300"
                  >
                    <span className="text-slate-500">{key}:</span> {String(value)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Adjustment Strategy (for regression analyses) */}
          {needsAdjustment && isIncluded && status === 'pending' && (
            <div>
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">
                Confounder Adjustment
              </div>
              <AdjustmentStrategySelector
                sessionId={sessionId}
                outcome={analysis.outcome || ''}
                outcomeType={analysis.outcome_type || 'binary'}
                mainPredictor={analysis.predictor || ''}
                timeCol={analysis.time_col}
                eventCol={analysis.event_col}
                onStrategyChange={handleStrategyChange}
                initialStrategy={adjustmentStrategy}
              />
            </div>
          )}

          {/* Result Summary (if completed) */}
          {status === 'completed' && result && (
            <div>
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">
                Result
              </div>
              <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                {result.p_value !== undefined && (
                  <div className="text-sm">
                    <span className="text-slate-400">p-value: </span>
                    <span className={clsx(
                      'font-mono font-medium',
                      result.p_value < 0.05 ? 'text-emerald-400' : 'text-slate-300'
                    )}>
                      {result.p_value < 0.001 ? '<0.001' : result.p_value.toFixed(4)}
                    </span>
                    {result.p_value < 0.05 && (
                      <span className="ml-2 text-emerald-400 text-xs">Significant</span>
                    )}
                  </div>
                )}
                {result.interpretation && (
                  <p className="text-sm text-slate-300 mt-2">
                    {result.interpretation.slice(0, 200)}...
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between pt-2 border-t border-slate-700/30">
            <button
              onClick={() => onRemove(analysis.id)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Remove
            </button>

            {status === 'pending' && isIncluded && (
              <button
                onClick={handleRun}
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg text-sm font-medium hover:from-indigo-500 hover:to-purple-500 transition-colors shadow-lg shadow-indigo-500/25"
              >
                <Play className="w-3.5 h-3.5" />
                Run Analysis
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
