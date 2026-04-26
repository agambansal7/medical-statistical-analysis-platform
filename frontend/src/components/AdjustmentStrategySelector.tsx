import { useState, useEffect } from 'react';
import {
  ChevronDown,
  Check,
  Info,
  Loader2,
  X,
  Sparkles,
  Shield,
  TrendingUp,
  Users,
  Settings2,
} from 'lucide-react';
import clsx from 'clsx';
import { chatApi } from '../services/api';

export type AdjustmentStrategy =
  | 'none'
  | 'clinical'
  | 'significant'
  | 'change_in_estimate'
  | 'minimal'
  | 'custom';

interface VariableInfo {
  name: string;
  type: 'continuous' | 'categorical';
  missing_pct: number;
  p_value?: number;
  effect_change?: number; // % change in main effect when included
}

interface AdjustmentStrategySelectorProps {
  sessionId: string;
  outcome: string;
  outcomeType: 'binary' | 'continuous' | 'survival';
  mainPredictor: string;
  timeCol?: string;
  eventCol?: string;
  onStrategyChange: (strategy: AdjustmentStrategy, variables: string[]) => void;
  initialStrategy?: AdjustmentStrategy;
  compact?: boolean;
}

const STRATEGY_CONFIG = {
  none: {
    label: 'Unadjusted',
    description: 'No adjustment for confounders',
    icon: X,
    color: 'slate',
  },
  clinical: {
    label: 'Clinical Confounders',
    description: 'Age, sex, and common comorbidities',
    icon: Users,
    color: 'blue',
  },
  significant: {
    label: 'Statistically Significant',
    description: 'Variables with p < 0.1 in univariate analysis',
    icon: TrendingUp,
    color: 'emerald',
  },
  change_in_estimate: {
    label: 'Change-in-Estimate',
    description: 'Variables that change main effect by >10%',
    icon: Shield,
    color: 'purple',
  },
  minimal: {
    label: 'Minimal Sufficient',
    description: 'AI-recommended minimal adjustment set',
    icon: Sparkles,
    color: 'amber',
  },
  custom: {
    label: 'Custom Selection',
    description: 'Manually select variables',
    icon: Settings2,
    color: 'indigo',
  },
};

export default function AdjustmentStrategySelector({
  sessionId,
  outcome,
  outcomeType,
  mainPredictor: _mainPredictor,
  timeCol,
  eventCol,
  onStrategyChange,
  initialStrategy = 'clinical',
  compact = false,
}: AdjustmentStrategySelectorProps) {
  const [strategy, setStrategy] = useState<AdjustmentStrategy>(initialStrategy);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [variables, setVariables] = useState<VariableInfo[]>([]);
  const [selectedVars, setSelectedVars] = useState<Set<string>>(new Set());
  const [showCustom, setShowCustom] = useState(false);

  // Clinical variable patterns
  const CLINICAL_PATTERNS = [
    'age', 'sex', 'gender', 'bmi', 'diabetes', 'hypertension', 'smoking',
    'ckd', 'egfr', 'creatinine', 'heart_failure', 'chf', 'cad', 'mi',
    'stroke', 'copd', 'atrial_fibrillation', 'af', 'race', 'ethnicity'
  ];

  useEffect(() => {
    if (expanded && variables.length === 0) {
      loadVariables();
    }
  }, [expanded]);

  useEffect(() => {
    // When strategy changes, compute selected variables
    computeSelectedVariables();
  }, [strategy, variables]);

  const loadVariables = async () => {
    setLoading(true);
    try {
      const response = await chatApi.getConfounders(
        sessionId,
        outcome,
        outcomeType,
        timeCol,
        eventCol
      );
      if (response.success) {
        const vars: VariableInfo[] = response.confounders.map((c: any) => ({
          name: c.variable,
          type: c.type,
          missing_pct: c.missing_pct,
          p_value: c.p_value,
          effect_change: c.effect_change,
        }));
        setVariables(vars);
      }
    } catch (error) {
      console.error('Failed to load variables:', error);
    } finally {
      setLoading(false);
    }
  };

  const computeSelectedVariables = () => {
    let selected: string[] = [];

    switch (strategy) {
      case 'none':
        selected = [];
        break;

      case 'clinical':
        selected = variables
          .filter(v => CLINICAL_PATTERNS.some(p =>
            v.name.toLowerCase().includes(p)
          ))
          .filter(v => v.missing_pct < 30)
          .map(v => v.name);
        break;

      case 'significant':
        selected = variables
          .filter(v => v.p_value !== undefined && v.p_value < 0.1)
          .filter(v => v.missing_pct < 30)
          .map(v => v.name);
        break;

      case 'change_in_estimate':
        selected = variables
          .filter(v => v.effect_change !== undefined && Math.abs(v.effect_change) > 10)
          .filter(v => v.missing_pct < 30)
          .map(v => v.name);
        // Fallback to clinical if no change-in-estimate data
        if (selected.length === 0) {
          selected = variables
            .filter(v => CLINICAL_PATTERNS.some(p =>
              v.name.toLowerCase().includes(p)
            ))
            .filter(v => v.missing_pct < 30)
            .map(v => v.name);
        }
        break;

      case 'minimal':
        // AI-recommended: significant clinical variables with low missing
        selected = variables
          .filter(v =>
            CLINICAL_PATTERNS.some(p => v.name.toLowerCase().includes(p)) &&
            (v.p_value === undefined || v.p_value < 0.2) &&
            v.missing_pct < 20
          )
          .slice(0, 5)
          .map(v => v.name);
        break;

      case 'custom':
        selected = Array.from(selectedVars);
        break;
    }

    setSelectedVars(new Set(selected));
    onStrategyChange(strategy, selected);
  };

  const handleStrategySelect = (newStrategy: AdjustmentStrategy) => {
    setStrategy(newStrategy);
    if (newStrategy === 'custom') {
      setShowCustom(true);
    } else {
      setShowCustom(false);
    }
    setExpanded(false);
  };

  const toggleCustomVar = (varName: string) => {
    const newSelected = new Set(selectedVars);
    if (newSelected.has(varName)) {
      newSelected.delete(varName);
    } else {
      newSelected.add(varName);
    }
    setSelectedVars(newSelected);
    onStrategyChange('custom', Array.from(newSelected));
  };

  const config = STRATEGY_CONFIG[strategy];
  const IconComponent = config.icon;

  const getColorClasses = (color: string, _isSelected: boolean) => {
    const colors: Record<string, { bg: string; border: string; text: string }> = {
      slate: { bg: 'bg-slate-500/10', border: 'border-slate-500/30', text: 'text-slate-400' },
      blue: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400' },
      emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400' },
      purple: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400' },
      amber: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400' },
      indigo: { bg: 'bg-indigo-500/10', border: 'border-indigo-500/30', text: 'text-indigo-400' },
    };
    return colors[color] || colors.slate;
  };

  if (compact) {
    // Compact inline version
    return (
      <div className="relative">
        <button
          onClick={() => setExpanded(!expanded)}
          className={clsx(
            'flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm transition-all',
            getColorClasses(config.color, true).bg,
            getColorClasses(config.color, true).border,
            getColorClasses(config.color, true).text
          )}
        >
          <IconComponent className="w-3.5 h-3.5" />
          <span>{config.label}</span>
          {selectedVars.size > 0 && strategy !== 'none' && (
            <span className="px-1.5 py-0.5 bg-white/10 rounded text-xs">
              {selectedVars.size}
            </span>
          )}
          <ChevronDown className={clsx(
            'w-3.5 h-3.5 transition-transform',
            expanded && 'rotate-180'
          )} />
        </button>

        {expanded && (
          <div className="absolute top-full left-0 mt-1 w-64 bg-slate-800 border border-slate-700 rounded-xl shadow-xl z-50 overflow-hidden">
            {Object.entries(STRATEGY_CONFIG).map(([key, cfg]) => {
              const Icon = cfg.icon;
              const colors = getColorClasses(cfg.color, key === strategy);
              return (
                <button
                  key={key}
                  onClick={() => handleStrategySelect(key as AdjustmentStrategy)}
                  className={clsx(
                    'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors',
                    key === strategy ? colors.bg : 'hover:bg-slate-700/50'
                  )}
                >
                  <Icon className={clsx('w-4 h-4', colors.text)} />
                  <div className="flex-1 min-w-0">
                    <div className={clsx(
                      'text-sm font-medium',
                      key === strategy ? colors.text : 'text-slate-200'
                    )}>
                      {cfg.label}
                    </div>
                    <div className="text-xs text-slate-500 truncate">
                      {cfg.description}
                    </div>
                  </div>
                  {key === strategy && (
                    <Check className={clsx('w-4 h-4', colors.text)} />
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  // Full version with variable list
  return (
    <div className="border border-slate-700/50 rounded-xl overflow-hidden bg-slate-800/30">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-800/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className={clsx(
            'w-8 h-8 rounded-lg flex items-center justify-center',
            getColorClasses(config.color, true).bg
          )}>
            <IconComponent className={clsx('w-4 h-4', getColorClasses(config.color, true).text)} />
          </div>
          <div className="text-left">
            <div className="text-sm font-medium text-slate-200">
              Adjustment: {config.label}
            </div>
            <div className="text-xs text-slate-500">
              {strategy === 'none'
                ? 'No confounders included'
                : `${selectedVars.size} variables selected`
              }
            </div>
          </div>
        </div>
        <ChevronDown className={clsx(
          'w-5 h-5 text-slate-400 transition-transform',
          expanded && 'rotate-180'
        )} />
      </button>

      {expanded && (
        <div className="border-t border-slate-700/50">
          {/* Strategy Options */}
          <div className="p-3 grid grid-cols-2 gap-2">
            {Object.entries(STRATEGY_CONFIG).map(([key, cfg]) => {
              const Icon = cfg.icon;
              const colors = getColorClasses(cfg.color, key === strategy);
              const isSelected = key === strategy;

              return (
                <button
                  key={key}
                  onClick={() => handleStrategySelect(key as AdjustmentStrategy)}
                  className={clsx(
                    'flex items-center gap-2 px-3 py-2 rounded-lg border text-left transition-all',
                    isSelected
                      ? `${colors.bg} ${colors.border}`
                      : 'border-slate-700/50 hover:border-slate-600 hover:bg-slate-700/30'
                  )}
                >
                  <Icon className={clsx(
                    'w-4 h-4 flex-shrink-0',
                    isSelected ? colors.text : 'text-slate-500'
                  )} />
                  <span className={clsx(
                    'text-sm',
                    isSelected ? colors.text : 'text-slate-300'
                  )}>
                    {cfg.label}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Selected Variables Preview */}
          {strategy !== 'none' && (
            <div className="px-4 pb-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-500 uppercase tracking-wider">
                  Selected Variables
                </span>
                {loading && (
                  <Loader2 className="w-3 h-3 animate-spin text-slate-500" />
                )}
              </div>

              {selectedVars.size > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {Array.from(selectedVars).slice(0, 8).map(varName => (
                    <span
                      key={varName}
                      className={clsx(
                        'inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs',
                        strategy === 'custom' ? 'pr-1' : '',
                        getColorClasses(config.color, true).bg,
                        getColorClasses(config.color, true).text
                      )}
                    >
                      {varName}
                      {strategy === 'custom' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleCustomVar(varName);
                          }}
                          className="p-0.5 hover:bg-white/10 rounded"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      )}
                    </span>
                  ))}
                  {selectedVars.size > 8 && (
                    <span className="px-2 py-1 text-xs text-slate-500">
                      +{selectedVars.size - 8} more
                    </span>
                  )}
                </div>
              ) : (
                <div className="text-sm text-slate-500 italic">
                  No variables match this strategy
                </div>
              )}
            </div>
          )}

          {/* Custom Selection Panel */}
          {strategy === 'custom' && showCustom && (
            <div className="border-t border-slate-700/50 p-3 max-h-48 overflow-y-auto">
              <div className="space-y-1">
                {variables.map(v => (
                  <label
                    key={v.name}
                    className={clsx(
                      'flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors',
                      selectedVars.has(v.name)
                        ? 'bg-indigo-500/20'
                        : 'hover:bg-slate-700/50'
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={selectedVars.has(v.name)}
                      onChange={() => toggleCustomVar(v.name)}
                      className="rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500"
                    />
                    <span className={clsx(
                      'flex-1 text-sm',
                      selectedVars.has(v.name) ? 'text-slate-100' : 'text-slate-300'
                    )}>
                      {v.name}
                    </span>
                    <div className="flex items-center gap-2 text-xs">
                      <span className={clsx(
                        'px-1.5 py-0.5 rounded',
                        v.type === 'continuous'
                          ? 'bg-blue-500/20 text-blue-400'
                          : 'bg-purple-500/20 text-purple-400'
                      )}>
                        {v.type}
                      </span>
                      {v.p_value !== undefined && (
                        <span className={clsx(
                          'font-mono',
                          v.p_value < 0.05 ? 'text-emerald-400' : 'text-slate-500'
                        )}>
                          p={v.p_value < 0.001 ? '<.001' : v.p_value.toFixed(3)}
                        </span>
                      )}
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Info Footer */}
          {strategy !== 'none' && strategy !== 'custom' && (
            <div className="px-4 py-2 bg-slate-800/50 border-t border-slate-700/50">
              <div className="flex items-start gap-2 text-xs text-slate-500">
                <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                <span>{config.description}</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
