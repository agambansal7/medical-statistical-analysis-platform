import { useState } from 'react';
import {
  Table2,
  TrendingUp,
  Activity,
  BarChart3,
  FileText,
  Download,
  Settings,
  Sparkles,
  ChevronRight,
  Users,
  PieChart,
  GitBranch,
  Calculator,
} from 'lucide-react';
import clsx from 'clsx';
import toast from 'react-hot-toast';

interface QuickAction {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
  category: 'descriptive' | 'comparative' | 'regression' | 'survival' | 'export';
  analysisType?: string;
  color: string;
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'table1',
    label: 'Table 1',
    icon: Table2,
    description: 'Baseline characteristics table',
    category: 'descriptive',
    analysisType: 'table1',
    color: 'blue',
  },
  {
    id: 'compare_groups',
    label: 'Compare Groups',
    icon: Users,
    description: 'Compare outcomes between groups',
    category: 'comparative',
    analysisType: 'group_comparison',
    color: 'purple',
  },
  {
    id: 'survival',
    label: 'Survival Analysis',
    icon: Activity,
    description: 'Kaplan-Meier curves & Cox regression',
    category: 'survival',
    analysisType: 'survival_analysis',
    color: 'cyan',
  },
  {
    id: 'regression',
    label: 'Regression',
    icon: TrendingUp,
    description: 'Logistic or linear regression',
    category: 'regression',
    analysisType: 'regression',
    color: 'emerald',
  },
  {
    id: 'visualize',
    label: 'Visualize',
    icon: BarChart3,
    description: 'Generate charts and plots',
    category: 'export',
    color: 'orange',
  },
  {
    id: 'export_report',
    label: 'Export Report',
    icon: FileText,
    description: 'Generate results section',
    category: 'export',
    color: 'pink',
  },
];

interface QuickActionBarProps {
  onAction: (actionId: string, analysisType?: string) => void;
  disabled?: boolean;
  compact?: boolean;
}

export default function QuickActionBar({
  onAction,
  disabled = false,
  compact = false,
}: QuickActionBarProps) {
  const [hoveredAction, setHoveredAction] = useState<string | null>(null);

  const getColorClasses = (color: string, isHovered: boolean) => {
    const colors: Record<string, { bg: string; text: string; border: string; hover: string }> = {
      blue: {
        bg: 'bg-blue-500/10',
        text: 'text-blue-400',
        border: 'border-blue-500/30',
        hover: 'hover:bg-blue-500/20',
      },
      purple: {
        bg: 'bg-purple-500/10',
        text: 'text-purple-400',
        border: 'border-purple-500/30',
        hover: 'hover:bg-purple-500/20',
      },
      cyan: {
        bg: 'bg-cyan-500/10',
        text: 'text-cyan-400',
        border: 'border-cyan-500/30',
        hover: 'hover:bg-cyan-500/20',
      },
      emerald: {
        bg: 'bg-emerald-500/10',
        text: 'text-emerald-400',
        border: 'border-emerald-500/30',
        hover: 'hover:bg-emerald-500/20',
      },
      orange: {
        bg: 'bg-orange-500/10',
        text: 'text-orange-400',
        border: 'border-orange-500/30',
        hover: 'hover:bg-orange-500/20',
      },
      pink: {
        bg: 'bg-pink-500/10',
        text: 'text-pink-400',
        border: 'border-pink-500/30',
        hover: 'hover:bg-pink-500/20',
      },
    };

    return colors[color] || colors.blue;
  };

  if (compact) {
    return (
      <div className="flex items-center space-x-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-700">
        {QUICK_ACTIONS.map((action) => {
          const colors = getColorClasses(action.color, hoveredAction === action.id);
          return (
            <button
              key={action.id}
              onClick={() => onAction(action.id, action.analysisType)}
              disabled={disabled}
              onMouseEnter={() => setHoveredAction(action.id)}
              onMouseLeave={() => setHoveredAction(null)}
              className={clsx(
                'flex items-center space-x-2 px-3 py-2 rounded-xl border transition-all whitespace-nowrap',
                colors.bg,
                colors.border,
                colors.hover,
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <action.icon className={clsx('w-4 h-4', colors.text)} />
              <span className={clsx('text-sm font-medium', colors.text)}>{action.label}</span>
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-indigo-400" />
          <h3 className="font-semibold text-slate-100">Quick Actions</h3>
        </div>
        <span className="text-xs text-slate-500">One-click analyses</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {QUICK_ACTIONS.map((action) => {
          const colors = getColorClasses(action.color, hoveredAction === action.id);
          return (
            <button
              key={action.id}
              onClick={() => onAction(action.id, action.analysisType)}
              disabled={disabled}
              onMouseEnter={() => setHoveredAction(action.id)}
              onMouseLeave={() => setHoveredAction(null)}
              className={clsx(
                'group flex flex-col items-start p-4 rounded-xl border transition-all text-left',
                colors.bg,
                colors.border,
                colors.hover,
                disabled && 'opacity-50 cursor-not-allowed'
              )}
            >
              <div className="flex items-center justify-between w-full mb-2">
                <div
                  className={clsx(
                    'w-10 h-10 rounded-xl flex items-center justify-center transition-transform group-hover:scale-110',
                    colors.bg
                  )}
                >
                  <action.icon className={clsx('w-5 h-5', colors.text)} />
                </div>
                <ChevronRight
                  className={clsx(
                    'w-4 h-4 text-slate-600 transition-transform group-hover:translate-x-1',
                    colors.text
                  )}
                />
              </div>
              <h4 className={clsx('font-medium mb-1', colors.text)}>{action.label}</h4>
              <p className="text-xs text-slate-500 line-clamp-2">{action.description}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Floating quick action button
export function QuickActionButton({
  onClick,
  disabled = false,
}: {
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'fixed bottom-6 right-6 w-14 h-14 rounded-full shadow-lg transition-all',
        'bg-gradient-to-r from-indigo-600 to-purple-600 text-white',
        'hover:shadow-xl hover:shadow-indigo-500/30 hover:scale-105',
        'flex items-center justify-center',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      <Sparkles className="w-6 h-6" />
    </button>
  );
}
