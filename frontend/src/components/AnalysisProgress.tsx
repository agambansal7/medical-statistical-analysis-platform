import { useEffect, useState } from 'react';
import {
  CheckCircle,
  Circle,
  Loader2,
  XCircle,
  Clock,
  TrendingUp,
} from 'lucide-react';
import clsx from 'clsx';

export interface AnalysisStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration?: number;
  error?: string;
}

interface AnalysisProgressProps {
  steps: AnalysisStep[];
  isRunning: boolean;
  onCancel?: () => void;
}

export default function AnalysisProgress({ steps, isRunning, onCancel }: AnalysisProgressProps) {
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    if (!isRunning) {
      setElapsedTime(0);
      return;
    }

    const interval = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [isRunning]);

  const completedCount = steps.filter(s => s.status === 'completed').length;
  const failedCount = steps.filter(s => s.status === 'failed').length;
  const progress = steps.length > 0 ? (completedCount / steps.length) * 100 : 0;

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const getStatusIcon = (status: AnalysisStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'running':
        return <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Circle className="w-4 h-4 text-slate-600" />;
    }
  };

  if (!isRunning && steps.length === 0) return null;

  return (
    <div className="glass-card overflow-hidden animate-fade-in">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <TrendingUp className="w-5 h-5 text-indigo-400" />
              {isRunning && (
                <span className="absolute -top-1 -right-1 w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
              )}
            </div>
            <div>
              <h3 className="font-semibold text-slate-100">
                {isRunning ? 'Running Analyses' : 'Analysis Complete'}
              </h3>
              <p className="text-xs text-slate-400">
                {completedCount}/{steps.length} completed
                {failedCount > 0 && ` • ${failedCount} failed`}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            {isRunning && (
              <div className="flex items-center space-x-2 text-sm text-slate-400">
                <Clock className="w-4 h-4" />
                <span>{formatTime(elapsedTime)}</span>
              </div>
            )}
            {isRunning && onCancel && (
              <button
                onClick={onCancel}
                className="px-3 py-1 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-3">
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full transition-all duration-500',
                failedCount > 0
                  ? 'bg-gradient-to-r from-emerald-500 via-yellow-500 to-red-500'
                  : 'bg-gradient-to-r from-indigo-500 to-purple-500'
              )}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Steps List */}
      <div className="p-4 space-y-2 max-h-64 overflow-y-auto">
        {steps.map((step, _index) => (
          <div
            key={step.id}
            className={clsx(
              'flex items-center justify-between p-3 rounded-xl transition-all duration-300',
              step.status === 'running' && 'bg-indigo-500/10 border border-indigo-500/30',
              step.status === 'completed' && 'bg-emerald-500/5',
              step.status === 'failed' && 'bg-red-500/10 border border-red-500/30',
              step.status === 'pending' && 'opacity-50'
            )}
          >
            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                {getStatusIcon(step.status)}
              </div>
              <div>
                <span className={clsx(
                  'text-sm font-medium',
                  step.status === 'completed' && 'text-emerald-300',
                  step.status === 'running' && 'text-indigo-300',
                  step.status === 'failed' && 'text-red-300',
                  step.status === 'pending' && 'text-slate-400'
                )}>
                  {step.name}
                </span>
                {step.error && (
                  <p className="text-xs text-red-400 mt-0.5">{step.error}</p>
                )}
              </div>
            </div>
            {step.duration !== undefined && step.status === 'completed' && (
              <span className="text-xs text-slate-500">{step.duration.toFixed(1)}s</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
