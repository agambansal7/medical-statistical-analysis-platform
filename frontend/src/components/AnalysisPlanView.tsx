import { useState, useMemo } from 'react';
import {
  PlayCircle,
  ChevronDown,
  ChevronUp,
  Sparkles,
  AlertTriangle,
  Settings2,
  Loader2,
  CheckCircle,
} from 'lucide-react';
import clsx from 'clsx';
import AnalysisPlanCard from './AnalysisPlanCard';
import { useStore } from '../hooks/useStore';
import { analysisApi } from '../services/api';
import toast from 'react-hot-toast';

interface Analysis {
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

interface AnalysisPlanViewProps {
  plan: {
    research_question: string;
    research_type: string;
    primary_analyses: Analysis[];
    secondary_analyses?: Analysis[];
    sensitivity_analyses?: Analysis[];
  };
  onConfirm: () => void;
  onCancel: () => void;
  onExecutionComplete: (results: any[]) => void;
}

type AnalysisStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export default function AnalysisPlanView({
  plan,
  onConfirm,
  onCancel,
  onExecutionComplete,
}: AnalysisPlanViewProps) {
  const { sessionId, addAnalysisResult, setAnalysisProgress, setIsAnalysisRunning } = useStore();

  const [includedAnalyses, setIncludedAnalyses] = useState<Set<string>>(() => {
    // Include all analyses by default
    const allIds = [
      ...plan.primary_analyses.map(a => a.id || a.test_name),
      ...(plan.secondary_analyses || []).map(a => a.id || a.test_name),
      ...(plan.sensitivity_analyses || []).map(a => a.id || a.test_name),
    ];
    return new Set(allIds);
  });

  const [analysisStatus, setAnalysisStatus] = useState<Record<string, AnalysisStatus>>({});
  const [analysisResults, setAnalysisResults] = useState<Record<string, any>>({});
  const [confounderSelections, setConfounderSelections] = useState<Record<string, string[]>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [showSecondary, setShowSecondary] = useState(true);
  const [showSensitivity, setShowSensitivity] = useState(false);

  // Ensure each analysis has an ID
  const normalizeAnalyses = (analyses: Analysis[]) => {
    return analyses.map((a, i) => ({
      ...a,
      id: a.id || `${a.analysis_type}_${i}`,
    }));
  };

  const primaryAnalyses = useMemo(() => normalizeAnalyses(plan.primary_analyses), [plan.primary_analyses]);
  const secondaryAnalyses = useMemo(() => normalizeAnalyses(plan.secondary_analyses || []), [plan.secondary_analyses]);
  const sensitivityAnalyses = useMemo(() => normalizeAnalyses(plan.sensitivity_analyses || []), [plan.sensitivity_analyses]);

  const allAnalyses = [...primaryAnalyses, ...secondaryAnalyses, ...sensitivityAnalyses];
  const includedCount = includedAnalyses.size;
  const completedCount = Object.values(analysisStatus).filter(s => s === 'completed').length;
  const failedCount = Object.values(analysisStatus).filter(s => s === 'failed').length;

  const toggleInclude = (id: string, include: boolean) => {
    const newIncluded = new Set(includedAnalyses);
    if (include) {
      newIncluded.add(id);
    } else {
      newIncluded.delete(id);
    }
    setIncludedAnalyses(newIncluded);
  };

  const removeAnalysis = (id: string) => {
    setIncludedAnalyses(prev => {
      const newSet = new Set(prev);
      newSet.delete(id);
      return newSet;
    });
  };

  const runSingleAnalysis = async (analysis: Analysis, confounders: string[]) => {
    if (!sessionId) return;

    const id = analysis.id;
    setAnalysisStatus(prev => ({ ...prev, [id]: 'running' }));
    setConfounderSelections(prev => ({ ...prev, [id]: confounders }));

    try {
      const params = {
        ...analysis.parameters,
        confounders: confounders.length > 0 ? confounders : undefined,
      };

      const result = await analysisApi.run(sessionId, analysis.analysis_type, params);

      setAnalysisStatus(prev => ({ ...prev, [id]: 'completed' }));
      setAnalysisResults(prev => ({ ...prev, [id]: result }));
      addAnalysisResult(result);
      toast.success(`${analysis.test_name} completed`);
    } catch (error: any) {
      setAnalysisStatus(prev => ({ ...prev, [id]: 'failed' }));
      toast.error(`${analysis.test_name} failed: ${error.message}`);
    }
  };

  const runAllAnalyses = async () => {
    if (!sessionId) return;

    setIsRunning(true);
    setIsAnalysisRunning(true);

    const toRun = allAnalyses.filter(a => includedAnalyses.has(a.id) && analysisStatus[a.id] !== 'completed');
    const results: any[] = [];

    // Set up progress tracking
    setAnalysisProgress(toRun.map(a => ({
      id: a.id,
      name: a.test_name,
      status: 'pending' as const,
    })));

    for (const analysis of toRun) {
      setAnalysisStatus(prev => ({ ...prev, [analysis.id]: 'running' }));

      try {
        const confounders = confounderSelections[analysis.id] || [];
        const params = {
          ...analysis.parameters,
          confounders: confounders.length > 0 ? confounders : undefined,
        };

        const result = await analysisApi.run(sessionId, analysis.analysis_type, params);

        setAnalysisStatus(prev => ({ ...prev, [analysis.id]: 'completed' }));
        setAnalysisResults(prev => ({ ...prev, [analysis.id]: result }));
        results.push({ analysis, result });
        addAnalysisResult(result);
      } catch (error: any) {
        setAnalysisStatus(prev => ({ ...prev, [analysis.id]: 'failed' }));
        results.push({ analysis, error: error.message });
      }
    }

    setIsRunning(false);
    setIsAnalysisRunning(false);
    onExecutionComplete(results);

    const successCount = results.filter(r => !r.error).length;
    if (successCount === toRun.length) {
      toast.success(`All ${successCount} analyses completed successfully!`);
    } else {
      toast.success(`Completed ${successCount}/${toRun.length} analyses`);
    }
  };

  const renderAnalysisSection = (
    title: string,
    analyses: Analysis[],
    isExpanded: boolean,
    onToggle: () => void,
    color: string
  ) => {
    if (analyses.length === 0) return null;

    const sectionCompleted = analyses.every(a =>
      !includedAnalyses.has(a.id) || analysisStatus[a.id] === 'completed'
    );

    return (
      <div className="space-y-3">
        <button
          onClick={onToggle}
          className="w-full flex items-center justify-between px-4 py-2 bg-slate-800/50 rounded-lg hover:bg-slate-800 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className={clsx(
              'w-2 h-2 rounded-full',
              sectionCompleted ? 'bg-emerald-500' : `bg-${color}-500`
            )} />
            <span className="font-medium text-slate-200">{title}</span>
            <span className="text-sm text-slate-500">
              ({analyses.filter(a => includedAnalyses.has(a.id)).length}/{analyses.length})
            </span>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </button>

        {isExpanded && (
          <div className="space-y-2 pl-4">
            {analyses.map((analysis, index) => (
              <AnalysisPlanCard
                key={analysis.id}
                analysis={analysis}
                sessionId={sessionId || ''}
                index={index}
                status={analysisStatus[analysis.id] || 'pending'}
                onRun={runSingleAnalysis}
                onRemove={removeAnalysis}
                onToggleInclude={toggleInclude}
                isIncluded={includedAnalyses.has(analysis.id)}
                result={analysisResults[analysis.id]}
              />
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="glass-card overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Sparkles className="w-5 h-5 text-indigo-200" />
              <h3 className="text-lg font-semibold text-white">Analysis Plan</h3>
            </div>
            <p className="text-indigo-200 text-sm max-w-xl">
              {plan.research_question}
            </p>
          </div>
          <span className="px-3 py-1 bg-white/20 rounded-full text-xs text-white">
            {plan.research_type}
          </span>
        </div>
      </div>

      {/* Progress Summary */}
      {(completedCount > 0 || failedCount > 0) && (
        <div className="px-6 py-3 bg-slate-800/50 border-b border-slate-700/50">
          <div className="flex items-center gap-4">
            {completedCount > 0 && (
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <CheckCircle className="w-4 h-4" />
                <span>{completedCount} completed</span>
              </div>
            )}
            {failedCount > 0 && (
              <div className="flex items-center gap-2 text-sm text-red-400">
                <AlertTriangle className="w-4 h-4" />
                <span>{failedCount} failed</span>
              </div>
            )}
            <div className="flex-1" />
            <div className="text-sm text-slate-400">
              {includedCount} analyses selected
            </div>
          </div>
        </div>
      )}

      {/* Analysis Sections */}
      <div className="p-6 space-y-6">
        {/* Primary Analyses - Always expanded */}
        <div className="space-y-3">
          <div className="flex items-center gap-3 px-4 py-2 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
            <span className="w-2 h-2 rounded-full bg-indigo-500" />
            <span className="font-medium text-indigo-300">Primary Analyses</span>
            <span className="text-sm text-indigo-400/70">
              ({primaryAnalyses.filter(a => includedAnalyses.has(a.id)).length}/{primaryAnalyses.length})
            </span>
          </div>
          <div className="space-y-2 pl-4">
            {primaryAnalyses.map((analysis, index) => (
              <AnalysisPlanCard
                key={analysis.id}
                analysis={analysis}
                sessionId={sessionId || ''}
                index={index}
                status={analysisStatus[analysis.id] || 'pending'}
                onRun={runSingleAnalysis}
                onRemove={removeAnalysis}
                onToggleInclude={toggleInclude}
                isIncluded={includedAnalyses.has(analysis.id)}
                result={analysisResults[analysis.id]}
              />
            ))}
          </div>
        </div>

        {/* Secondary Analyses */}
        {renderAnalysisSection(
          'Secondary Analyses',
          secondaryAnalyses,
          showSecondary,
          () => setShowSecondary(!showSecondary),
          'purple'
        )}

        {/* Sensitivity Analyses */}
        {renderAnalysisSection(
          'Sensitivity Analyses',
          sensitivityAnalyses,
          showSensitivity,
          () => setShowSensitivity(!showSensitivity),
          'amber'
        )}
      </div>

      {/* Footer Actions */}
      <div className="px-6 py-4 bg-slate-800/50 border-t border-slate-700/50 flex items-center justify-between">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-slate-400 hover:text-slate-300 hover:bg-slate-700/50 rounded-lg transition-colors"
        >
          Cancel
        </button>

        <div className="flex items-center gap-3">
          <button
            onClick={onConfirm}
            className="px-4 py-2 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-500/10 rounded-lg transition-colors"
          >
            <Settings2 className="w-4 h-4 inline mr-2" />
            Modify Plan
          </button>

          <button
            onClick={runAllAnalyses}
            disabled={isRunning || includedCount === 0}
            className={clsx(
              'flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-all',
              isRunning || includedCount === 0
                ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-500 hover:to-purple-500 shadow-lg shadow-indigo-500/25'
            )}
          >
            {isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <PlayCircle className="w-4 h-4" />
                Run All ({includedCount})
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
