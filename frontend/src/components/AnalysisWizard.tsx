import { useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Check,
  Beaker,
  Users,
  Target,
  BarChart3,
  Activity,
  GitBranch,
  FileText,
  Sparkles,
  X,
} from 'lucide-react';
import clsx from 'clsx';

interface WizardStep {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

const STEPS: WizardStep[] = [
  {
    id: 'study_type',
    title: 'Study Type',
    description: 'What type of study is this?',
    icon: Beaker,
  },
  {
    id: 'outcome',
    title: 'Outcome',
    description: 'What is your primary outcome?',
    icon: Target,
  },
  {
    id: 'exposure',
    title: 'Exposure/Group',
    description: 'What is your main exposure or grouping variable?',
    icon: Users,
  },
  {
    id: 'analyses',
    title: 'Analyses',
    description: 'Select the analyses to perform',
    icon: BarChart3,
  },
  {
    id: 'review',
    title: 'Review',
    description: 'Review and run your analysis plan',
    icon: FileText,
  },
];

const STUDY_TYPES = [
  { id: 'rct', label: 'Randomized Controlled Trial', icon: GitBranch, description: 'Experimental study with random assignment' },
  { id: 'cohort', label: 'Cohort Study', icon: Users, description: 'Follow groups over time' },
  { id: 'case_control', label: 'Case-Control Study', icon: Target, description: 'Compare cases with controls' },
  { id: 'cross_sectional', label: 'Cross-Sectional Study', icon: BarChart3, description: 'Single point in time' },
  { id: 'diagnostic', label: 'Diagnostic Accuracy', icon: Activity, description: 'Test sensitivity/specificity' },
];

interface AnalysisWizardProps {
  variables: Array<{ name: string; type: string }>;
  onComplete: (config: WizardConfig) => void;
  onClose: () => void;
}

export interface WizardConfig {
  studyType: string;
  outcome: string;
  outcomeType: 'binary' | 'continuous' | 'time_to_event';
  exposure: string;
  timeVariable?: string;
  eventVariable?: string;
  confounders: string[];
  analyses: string[];
}

export default function AnalysisWizard({ variables, onComplete, onClose }: AnalysisWizardProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [config, setConfig] = useState<Partial<WizardConfig>>({
    analyses: [],
    confounders: [],
  });

  const numericVars = variables.filter(v => v.type === 'continuous' || v.type === 'numeric');
  const categoricalVars = variables.filter(v => v.type === 'categorical' || v.type === 'binary');
  const binaryVars = variables.filter(v => v.type === 'binary');

  const canProceed = () => {
    switch (STEPS[currentStep].id) {
      case 'study_type':
        return !!config.studyType;
      case 'outcome':
        return !!config.outcome && !!config.outcomeType;
      case 'exposure':
        return !!config.exposure;
      case 'analyses':
        return (config.analyses?.length || 0) > 0;
      case 'review':
        return true;
      default:
        return false;
    }
  };

  const handleNext = () => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete(config as WizardConfig);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const getSuggestedAnalyses = () => {
    const suggestions: Array<{ id: string; name: string; description: string; recommended: boolean }> = [];

    // Always suggest Table 1
    suggestions.push({
      id: 'table1',
      name: 'Table 1: Baseline Characteristics',
      description: 'Compare demographics between groups',
      recommended: true,
    });

    if (config.outcomeType === 'binary') {
      suggestions.push({
        id: 'chi_square',
        name: 'Chi-Square Test',
        description: 'Test association between categorical variables',
        recommended: true,
      });
      suggestions.push({
        id: 'logistic_regression',
        name: 'Logistic Regression',
        description: 'Adjusted odds ratios for binary outcomes',
        recommended: true,
      });
    }

    if (config.outcomeType === 'continuous') {
      suggestions.push({
        id: 'ttest',
        name: 't-Test / Mann-Whitney',
        description: 'Compare means between groups',
        recommended: true,
      });
      suggestions.push({
        id: 'linear_regression',
        name: 'Linear Regression',
        description: 'Adjusted analysis for continuous outcomes',
        recommended: true,
      });
    }

    if (config.outcomeType === 'time_to_event') {
      suggestions.push({
        id: 'kaplan_meier',
        name: 'Kaplan-Meier Curves',
        description: 'Survival curves with log-rank test',
        recommended: true,
      });
      suggestions.push({
        id: 'cox_regression',
        name: 'Cox Regression',
        description: 'Adjusted hazard ratios',
        recommended: true,
      });
    }

    // Secondary analyses
    suggestions.push({
      id: 'subgroup',
      name: 'Subgroup Analysis',
      description: 'Analyze effects in subgroups',
      recommended: false,
    });

    return suggestions;
  };

  const renderStepContent = () => {
    switch (STEPS[currentStep].id) {
      case 'study_type':
        return (
          <div className="space-y-3">
            {STUDY_TYPES.map((type) => (
              <button
                key={type.id}
                onClick={() => setConfig({ ...config, studyType: type.id })}
                className={clsx(
                  'w-full flex items-center space-x-4 p-4 rounded-xl border transition-all text-left',
                  config.studyType === type.id
                    ? 'bg-indigo-500/20 border-indigo-500/50'
                    : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800/50'
                )}
              >
                <div
                  className={clsx(
                    'w-12 h-12 rounded-xl flex items-center justify-center',
                    config.studyType === type.id ? 'bg-indigo-500/30' : 'bg-slate-700/50'
                  )}
                >
                  <type.icon
                    className={clsx(
                      'w-6 h-6',
                      config.studyType === type.id ? 'text-indigo-400' : 'text-slate-400'
                    )}
                  />
                </div>
                <div className="flex-1">
                  <h4
                    className={clsx(
                      'font-medium',
                      config.studyType === type.id ? 'text-indigo-300' : 'text-slate-200'
                    )}
                  >
                    {type.label}
                  </h4>
                  <p className="text-sm text-slate-500">{type.description}</p>
                </div>
                {config.studyType === type.id && (
                  <Check className="w-5 h-5 text-indigo-400" />
                )}
              </button>
            ))}
          </div>
        );

      case 'outcome':
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Outcome Variable
              </label>
              <select
                value={config.outcome || ''}
                onChange={(e) => setConfig({ ...config, outcome: e.target.value })}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl
                           text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="">Select outcome variable</option>
                {variables.map((v) => (
                  <option key={v.name} value={v.name}>
                    {v.name} ({v.type})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Outcome Type
              </label>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { id: 'binary', label: 'Binary', desc: 'Yes/No, 0/1' },
                  { id: 'continuous', label: 'Continuous', desc: 'Numeric values' },
                  { id: 'time_to_event', label: 'Time-to-Event', desc: 'Survival data' },
                ].map((type) => (
                  <button
                    key={type.id}
                    onClick={() => setConfig({ ...config, outcomeType: type.id as any })}
                    className={clsx(
                      'p-4 rounded-xl border text-center transition-all',
                      config.outcomeType === type.id
                        ? 'bg-indigo-500/20 border-indigo-500/50'
                        : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800/50'
                    )}
                  >
                    <div
                      className={clsx(
                        'font-medium',
                        config.outcomeType === type.id ? 'text-indigo-300' : 'text-slate-200'
                      )}
                    >
                      {type.label}
                    </div>
                    <div className="text-xs text-slate-500 mt-1">{type.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {config.outcomeType === 'time_to_event' && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Time Variable
                  </label>
                  <select
                    value={config.timeVariable || ''}
                    onChange={(e) => setConfig({ ...config, timeVariable: e.target.value })}
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-xl
                               text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="">Select time variable</option>
                    {numericVars.map((v) => (
                      <option key={v.name} value={v.name}>{v.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Event Variable
                  </label>
                  <select
                    value={config.eventVariable || ''}
                    onChange={(e) => setConfig({ ...config, eventVariable: e.target.value })}
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-xl
                               text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="">Select event variable</option>
                    {binaryVars.map((v) => (
                      <option key={v.name} value={v.name}>{v.name}</option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </div>
        );

      case 'exposure':
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Main Exposure / Group Variable
              </label>
              <select
                value={config.exposure || ''}
                onChange={(e) => setConfig({ ...config, exposure: e.target.value })}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl
                           text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="">Select exposure variable</option>
                {categoricalVars.map((v) => (
                  <option key={v.name} value={v.name}>
                    {v.name} ({v.type})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Confounders (optional)
              </label>
              <div className="max-h-48 overflow-y-auto space-y-2 p-3 bg-slate-800/30 rounded-xl border border-slate-700/50">
                {variables
                  .filter((v) => v.name !== config.outcome && v.name !== config.exposure)
                  .map((v) => (
                    <label key={v.name} className="flex items-center space-x-3 p-2 hover:bg-slate-700/30 rounded-lg cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.confounders?.includes(v.name)}
                        onChange={(e) => {
                          const confounders = config.confounders || [];
                          if (e.target.checked) {
                            setConfig({ ...config, confounders: [...confounders, v.name] });
                          } else {
                            setConfig({ ...config, confounders: confounders.filter((c) => c !== v.name) });
                          }
                        }}
                        className="rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500"
                      />
                      <span className="text-sm text-slate-300">{v.name}</span>
                      <span className="text-xs text-slate-500">({v.type})</span>
                    </label>
                  ))}
              </div>
            </div>
          </div>
        );

      case 'analyses':
        const suggestions = getSuggestedAnalyses();
        return (
          <div className="space-y-3">
            {suggestions.map((analysis) => (
              <label
                key={analysis.id}
                className={clsx(
                  'flex items-center space-x-4 p-4 rounded-xl border transition-all cursor-pointer',
                  config.analyses?.includes(analysis.id)
                    ? 'bg-indigo-500/20 border-indigo-500/50'
                    : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800/50'
                )}
              >
                <input
                  type="checkbox"
                  checked={config.analyses?.includes(analysis.id)}
                  onChange={(e) => {
                    const analyses = config.analyses || [];
                    if (e.target.checked) {
                      setConfig({ ...config, analyses: [...analyses, analysis.id] });
                    } else {
                      setConfig({ ...config, analyses: analyses.filter((a) => a !== analysis.id) });
                    }
                  }}
                  className="rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500"
                />
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-slate-200">{analysis.name}</span>
                    {analysis.recommended && (
                      <span className="px-2 py-0.5 text-xs bg-emerald-500/20 text-emerald-400 rounded-full">
                        Recommended
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-slate-500">{analysis.description}</p>
                </div>
              </label>
            ))}
          </div>
        );

      case 'review':
        return (
          <div className="space-y-4">
            <div className="glass-card p-4 border-indigo-500/30 bg-indigo-500/5">
              <h4 className="font-medium text-indigo-300 mb-3">Analysis Configuration</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Study Type:</span>
                  <span className="text-slate-200">{STUDY_TYPES.find((s) => s.id === config.studyType)?.label}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Outcome:</span>
                  <span className="text-slate-200">{config.outcome} ({config.outcomeType})</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Exposure:</span>
                  <span className="text-slate-200">{config.exposure}</span>
                </div>
                {config.confounders && config.confounders.length > 0 && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">Confounders:</span>
                    <span className="text-slate-200">{config.confounders.length} selected</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-slate-400">Analyses:</span>
                  <span className="text-slate-200">{config.analyses?.length} selected</span>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-2 p-3 bg-emerald-500/10 rounded-xl border border-emerald-500/30">
              <Sparkles className="w-5 h-5 text-emerald-400" />
              <span className="text-sm text-emerald-300">
                Ready to run your analysis! Click "Run Analysis" to proceed.
              </span>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-2xl mx-4 glass-card overflow-hidden animate-fade-in">
        {/* Header */}
        <div className="px-6 py-4 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border-b border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="font-semibold text-slate-100">Analysis Wizard</h2>
                <p className="text-xs text-slate-400">Step-by-step analysis setup</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="px-6 py-4 border-b border-slate-700/50">
          <div className="flex items-center justify-between">
            {STEPS.map((step, index) => (
              <div key={step.id} className="flex items-center">
                <div
                  className={clsx(
                    'flex items-center justify-center w-8 h-8 rounded-full transition-all',
                    index < currentStep
                      ? 'bg-emerald-500 text-white'
                      : index === currentStep
                      ? 'bg-indigo-500 text-white'
                      : 'bg-slate-700 text-slate-500'
                  )}
                >
                  {index < currentStep ? (
                    <Check className="w-4 h-4" />
                  ) : (
                    <step.icon className="w-4 h-4" />
                  )}
                </div>
                {index < STEPS.length - 1 && (
                  <div
                    className={clsx(
                      'w-12 h-0.5 mx-2',
                      index < currentStep ? 'bg-emerald-500' : 'bg-slate-700'
                    )}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="p-6">
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-slate-100">{STEPS[currentStep].title}</h3>
            <p className="text-sm text-slate-400">{STEPS[currentStep].description}</p>
          </div>

          <div className="min-h-[300px] max-h-[400px] overflow-y-auto">
            {renderStepContent()}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-700/50 flex items-center justify-between">
          <button
            onClick={handleBack}
            disabled={currentStep === 0}
            className={clsx(
              'flex items-center space-x-2 px-4 py-2 rounded-xl transition-colors',
              currentStep === 0
                ? 'text-slate-600 cursor-not-allowed'
                : 'text-slate-300 hover:bg-slate-700'
            )}
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Back</span>
          </button>

          <button
            onClick={handleNext}
            disabled={!canProceed()}
            className={clsx(
              'flex items-center space-x-2 px-6 py-2 rounded-xl font-medium transition-colors',
              canProceed()
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:shadow-lg hover:shadow-indigo-500/30'
                : 'bg-slate-700 text-slate-500 cursor-not-allowed'
            )}
          >
            <span>{currentStep === STEPS.length - 1 ? 'Run Analysis' : 'Next'}</span>
            {currentStep < STEPS.length - 1 && <ChevronRight className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}
