import { useState, useRef, useEffect } from 'react';
import { useStore } from '../hooks/useStore';
import { chatApi, PlanModifications } from '../services/api';
import {
  Send,
  Bot,
  User,
  Loader2,
  Sparkles,
  CheckCircle,
  AlertTriangle,
  Edit2,
  X,
  Settings,
  Zap,
  ArrowRight,
  Brain,
  FlaskConical,
  Wand2,
} from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import ConfounderSelector from './ConfounderSelector';
import AnalysisProgress, { type AnalysisStep } from './AnalysisProgress';
import QuickActionBar from './QuickActionBar';

export default function ChatPanel() {
  const {
    sessionId,
    chatMessages,
    addChatMessage,
    dataProfile,
    analysisPlan,
    setAnalysisPlan,
    setActiveTab,
    addAnalysisResult,
    clearAnalysisResults,
  } = useStore();

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [editingMappings, setEditingMappings] = useState<Record<string, string>>({});
  const [excludedAnalyses, setExcludedAnalyses] = useState<Set<string>>(new Set());
  const [showConfounderSelector, setShowConfounderSelector] = useState(false);
  const [confounderConfig, setConfounderConfig] = useState<{
    outcome: string;
    outcomeType: 'binary' | 'continuous' | 'survival';
    mainPredictor?: string;
    timeCol?: string;
    eventCol?: string;
  } | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisStep[]>([]);
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(false);
  const [showQuickActions, setShowQuickActions] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const getAdjustedAnalysisFromPlan = () => {
    if (!analysisPlan) return null;

    const allAnalyses = [
      ...(analysisPlan.primary_analyses || []),
      ...(analysisPlan.secondary_analyses || [])
    ];

    for (const analysis of allAnalyses) {
      const testName = analysis.test_name?.toLowerCase() || '';
      const apiCall = analysis.api_call || {};
      const params = apiCall.parameters || {};

      if (testName.includes('logistic') || apiCall.analysis_type === 'logistic_regression') {
        return {
          type: 'logistic_regression' as const,
          outcome: params.outcome || params.dependent || '',
          outcomeType: 'binary' as const,
          mainPredictor: params.predictor || params.group || '',
        };
      }

      if (testName.includes('cox') || apiCall.analysis_type === 'cox_regression') {
        return {
          type: 'cox_regression' as const,
          outcome: '',
          outcomeType: 'survival' as const,
          mainPredictor: params.predictor || params.group || '',
          timeCol: params.time || params.time_col || '',
          eventCol: params.event || params.event_col || '',
        };
      }
    }

    return null;
  };

  const openConfounderSelector = () => {
    const adjusted = getAdjustedAnalysisFromPlan();
    if (adjusted) {
      setConfounderConfig({
        outcome: adjusted.outcome,
        outcomeType: adjusted.outcomeType,
        mainPredictor: adjusted.mainPredictor,
        timeCol: adjusted.type === 'cox_regression' ? adjusted.timeCol : undefined,
        eventCol: adjusted.type === 'cox_regression' ? adjusted.eventCol : undefined,
      });
      setShowConfounderSelector(true);
    } else {
      toast.error('No adjusted analysis found in plan');
    }
  };

  const handleConfounderAnalysisComplete = (result: any) => {
    setShowConfounderSelector(false);
    if (result.success) {
      addAnalysisResult(result as any);
      setActiveTab('results');
      addChatMessage({
        role: 'assistant',
        content: `Adjusted analysis completed with your selected confounders. View the results in the Results tab.`,
        timestamp: new Date().toISOString(),
      });
    }
  };

  const runAnalysesWithProgress = async (analyses: Array<{ test_name: string; rationale?: string }>) => {
    if (!sessionId) return;

    // Initialize progress steps
    const steps: AnalysisStep[] = analyses.map((a, i) => ({
      id: String(i),
      name: a.test_name,
      status: 'pending' as const,
    }));
    setAnalysisProgress(steps);
    setIsAnalysisRunning(true);

    const startTime = Date.now();

    for (let i = 0; i < analyses.length; i++) {
      // Update current step to running
      setAnalysisProgress(prev =>
        prev.map((s, idx) => (idx === i ? { ...s, status: 'running' } : s))
      );

      const stepStart = Date.now();

      try {
        const result = await chatApi.executePlan(sessionId, [analyses[i].test_name]);

        const duration = (Date.now() - stepStart) / 1000;

        if (result.success && result.results?.[0]?.result?.success) {
          addAnalysisResult(result.results[0].result);
          setAnalysisProgress(prev =>
            prev.map((s, idx) => (idx === i ? { ...s, status: 'completed', duration } : s))
          );
        } else {
          setAnalysisProgress(prev =>
            prev.map((s, idx) =>
              idx === i
                ? { ...s, status: 'failed', error: result.errors?.[0]?.error || 'Analysis failed', duration }
                : s
            )
          );
        }
      } catch (error: any) {
        const duration = (Date.now() - stepStart) / 1000;
        setAnalysisProgress(prev =>
          prev.map((s, idx) =>
            idx === i ? { ...s, status: 'failed', error: error.message || 'Error', duration } : s
          )
        );
      }
    }

    setIsAnalysisRunning(false);

    const completedCount = analysisProgress.filter(s => s.status === 'completed').length;
    toast.success(`Completed ${completedCount} of ${analyses.length} analyses`);
    setActiveTab('results');
  };

  const handleQuickAction = (actionId: string, analysisType?: string) => {
    if (!sessionId || !dataProfile) return;

    // Map quick actions to chat prompts
    const prompts: Record<string, string> = {
      table1: 'Create Table 1 with baseline characteristics',
      compare_groups: 'Compare the main outcome between groups',
      survival: 'Perform survival analysis with Kaplan-Meier curves and Cox regression',
      regression: 'Run a regression analysis to identify predictors of the outcome',
      visualize: 'Generate recommended visualizations',
      export_report: 'Generate a results report',
    };

    const prompt = prompts[actionId];
    if (prompt) {
      setInput(prompt);
      // Optionally auto-send
      // handleSend();
    }

    setShowQuickActions(false);
  };

  const handleSend = async () => {
    if (!input.trim() || !sessionId || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    addChatMessage({
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString(),
    });

    try {
      const questionLower = userMessage.toLowerCase();
      const hasQuestionMark = userMessage.includes('?');
      const hasResearchKeywords =
        questionLower.includes('relationship') ||
        questionLower.includes('difference') ||
        questionLower.includes('effect') ||
        questionLower.includes('compare') ||
        questionLower.includes('predict') ||
        questionLower.includes('associated') ||
        questionLower.includes('significant') ||
        questionLower.includes('impact') ||
        questionLower.includes('outcome') ||
        questionLower.includes('affect') ||
        questionLower.includes('influence') ||
        questionLower.includes('association') ||
        questionLower.includes('correlation') ||
        questionLower.includes('risk') ||
        questionLower.includes('survival') ||
        questionLower.includes('mortality') ||
        questionLower.includes('analyze') ||
        questionLower.includes('analysis');

      const isResearchQuestion = dataProfile && (hasQuestionMark || hasResearchKeywords);

      if (isResearchQuestion) {
        const response = await chatApi.analyzeQuestion(sessionId, userMessage);

        if (response.success && response.plan) {
          setAnalysisPlan(response.plan);

          const requiresConfirmation = response.require_confirmation || response.plan.require_confirmation;

          let planSummary = `I've analyzed your research question and created an analysis plan:

**Research Type:** ${response.plan.research_type}

**Primary Analyses:**
${response.plan.primary_analyses.map((a, i) => `${i + 1}. **${a.test_name}** - ${a.rationale}`).join('\n')}

${response.plan.secondary_analyses?.length > 0 ? `**Secondary Analyses:**
${response.plan.secondary_analyses.map((a, i) => `${i + 1}. ${a.test_name}`).join('\n')}` : ''}`;

          if (response.plan.variable_warnings && response.plan.variable_warnings.length > 0) {
            planSummary += `

**Variable Mappings:**
${response.plan.variable_warnings.map((w: string) => `- ${w}`).join('\n')}`;
          }

          if (response.plan.validated === false) {
            planSummary += `

**Warning:** Some variables could not be matched to your dataset. Please review the plan before executing.`;
          }

          planSummary += `

**Assumption Checks:** ${response.plan.assumption_checks.join(', ')}

**Recommended Visualizations:** ${response.plan.visualizations.join(', ')}`;

          if (requiresConfirmation) {
            planSummary += `

**Please review and confirm this plan before execution.** Click "Review & Confirm Plan" below to proceed.`;
            setShowConfirmation(true);
            setEditingMappings(response.plan.variable_mappings || {});
            setExcludedAnalyses(new Set());
          } else {
            planSummary += `

Would you like me to run these analyses? Just say "run the analyses" or select specific ones.`;
          }

          addChatMessage({
            role: 'assistant',
            content: planSummary,
            timestamp: new Date().toISOString(),
          });
        }
      } else {
        const response = await chatApi.sendMessage(sessionId, userMessage);

        addChatMessage({
          role: 'assistant',
          content: response.message,
          timestamp: new Date().toISOString(),
        });

        if (response.action_executed && response.action_result) {
          toast.success('Analysis completed!');
          setActiveTab('results');
        }
      }
    } catch (error) {
      toast.error('Failed to send message');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const suggestedQuestions = [
    { text: 'Is there a significant difference in outcome between groups?', icon: Zap },
    { text: 'What factors predict the primary outcome?', icon: Brain },
    { text: 'Compare survival rates between treatment and control', icon: FlaskConical },
  ];

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-slate-900 to-slate-950">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-sm">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 rounded-full border-2 border-slate-900" />
          </div>
          <div>
            <h2 className="font-semibold text-slate-100">Research Assistant</h2>
            <p className="text-xs text-slate-400">Powered by AI</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {chatMessages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <div className="relative mb-8">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-3xl blur-2xl opacity-20 animate-pulse" />
              <div className="relative w-24 h-24 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-3xl flex items-center justify-center shadow-2xl shadow-indigo-500/20">
                <Bot className="w-12 h-12 text-white" />
              </div>
            </div>

            <h3 className="text-2xl font-bold text-slate-100 mb-3">
              How can I help with your analysis?
            </h3>
            <p className="text-slate-400 mb-8 max-w-md">
              {dataProfile
                ? 'Ask a research question, describe your hypothesis, or request specific statistical analyses.'
                : 'Upload your data first, then I can help you analyze it with advanced statistical methods.'}
            </p>

            {dataProfile && (
              <div className="w-full max-w-lg space-y-3">
                <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">
                  Suggested questions
                </p>
                {suggestedQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(q.text)}
                    className="w-full flex items-center gap-4 p-4 glass-card hover:bg-slate-800/50 transition-all duration-200 group"
                  >
                    <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center group-hover:bg-indigo-500/20 transition-colors">
                      <q.icon className="w-5 h-5 text-slate-400 group-hover:text-indigo-400 transition-colors" />
                    </div>
                    <span className="flex-1 text-left text-sm text-slate-300 group-hover:text-slate-100 transition-colors">
                      {q.text}
                    </span>
                    <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-indigo-400 transition-colors" />
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          chatMessages.map((message, i) => (
            <div
              key={i}
              className={clsx(
                'flex gap-4 animate-fade-in',
                message.role === 'user' ? 'flex-row-reverse' : ''
              )}
            >
              <div
                className={clsx(
                  'w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0',
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-indigo-500 to-purple-600'
                    : 'bg-slate-800 border border-slate-700'
                )}
              >
                {message.role === 'user' ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Bot className="w-5 h-5 text-indigo-400" />
                )}
              </div>
              <div
                className={clsx(
                  'max-w-[80%] rounded-2xl px-5 py-4 chat-message',
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-indigo-600 to-purple-600 text-white'
                    : 'glass-card'
                )}
              >
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                  {message.content.split('**').map((part, j) =>
                    j % 2 === 1 ? (
                      <strong key={j} className={message.role === 'user' ? 'text-white font-semibold' : ''}>
                        {part}
                      </strong>
                    ) : (
                      <span key={j}>{part}</span>
                    )
                  )}
                </div>
              </div>
            </div>
          ))
        )}

        {isLoading && (
          <div className="flex gap-4 animate-fade-in">
            <div className="w-9 h-9 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center">
              <Bot className="w-5 h-5 text-indigo-400" />
            </div>
            <div className="glass-card rounded-2xl px-5 py-4">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                <span className="text-sm text-slate-400">Analyzing...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Plan Confirmation Dialog */}
      {showConfirmation && analysisPlan && (
        <div className="border-t border-slate-700/50 bg-gradient-to-b from-amber-500/10 to-amber-500/5 px-6 py-4 max-h-80 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              </div>
              <span className="font-semibold text-slate-100">Review Analysis Plan</span>
            </div>
            <button
              onClick={() => setShowConfirmation(false)}
              className="p-1 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>

          {analysisPlan.variable_warnings && analysisPlan.variable_warnings.length > 0 && (
            <div className="mb-4 p-3 rounded-xl bg-slate-800/50 border border-slate-700/50">
              <h4 className="text-sm font-medium text-slate-300 mb-2">Variable Mappings</h4>
              <div className="space-y-2">
                {analysisPlan.variable_warnings.map((warning, i) => (
                  <div key={i} className="flex items-center text-sm">
                    {warning.includes('not found') ? (
                      <AlertTriangle className="w-4 h-4 text-red-400 mr-2 flex-shrink-0" />
                    ) : (
                      <CheckCircle className="w-4 h-4 text-emerald-400 mr-2 flex-shrink-0" />
                    )}
                    <span className={warning.includes('not found') ? 'text-red-300' : 'text-slate-300'}>
                      {warning}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mb-4">
            <h4 className="text-sm font-medium text-slate-300 mb-3">Analyses to Run</h4>
            <div className="space-y-2">
              {analysisPlan.primary_analyses.map((analysis, i) => (
                <label key={i} className="flex items-start gap-3 p-3 rounded-xl bg-slate-800/30 hover:bg-slate-800/50 cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={!excludedAnalyses.has(analysis.test_name)}
                    onChange={(e) => {
                      const newExcluded = new Set(excludedAnalyses);
                      if (e.target.checked) {
                        newExcluded.delete(analysis.test_name);
                      } else {
                        newExcluded.add(analysis.test_name);
                      }
                      setExcludedAnalyses(newExcluded);
                    }}
                    className="mt-1 rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500 focus:ring-offset-slate-900"
                  />
                  <div>
                    <span className="font-medium text-slate-200">{analysis.test_name}</span>
                    <p className="text-sm text-slate-400 mt-0.5">{analysis.rationale}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-end space-x-3">
            <button
              onClick={() => setShowConfirmation(false)}
              className="btn-secondary text-sm"
            >
              Cancel
            </button>
            <button
              onClick={async () => {
                if (!sessionId) return;
                setIsLoading(true);
                try {
                  const modifications: PlanModifications = {};
                  if (excludedAnalyses.size > 0) {
                    modifications.exclude_analyses = Array.from(excludedAnalyses);
                  }
                  if (Object.keys(editingMappings).length > 0) {
                    modifications.variable_mappings = editingMappings;
                  }

                  // First confirm the plan
                  await chatApi.confirmPlan(sessionId, Object.keys(modifications).length > 0 ? modifications : undefined);

                  // Check if plan has adjusted analyses (logistic/cox regression)
                  const adjustedConfig = getAdjustedAnalysisFromPlan();

                  if (adjustedConfig) {
                    // Show confounder selector instead of running immediately
                    setConfounderConfig({
                      outcome: adjustedConfig.outcome,
                      outcomeType: adjustedConfig.outcomeType,
                      mainPredictor: adjustedConfig.mainPredictor,
                      timeCol: adjustedConfig.type === 'cox_regression' ? adjustedConfig.timeCol : undefined,
                      eventCol: adjustedConfig.type === 'cox_regression' ? adjustedConfig.eventCol : undefined,
                    });
                    setShowConfirmation(false);
                    setShowConfounderSelector(true);

                    addChatMessage({
                      role: 'assistant',
                      content: `Plan confirmed! Before running adjusted analyses, please select which variables to include as confounders. Review the univariate p-values to help guide your selection.`,
                      timestamp: new Date().toISOString(),
                    });

                    // Run non-adjusted analyses first (Table 1, chi-square, etc.)
                    const nonAdjustedAnalyses = analysisPlan?.primary_analyses.filter(a => {
                      const type = a.api_call?.analysis_type?.toLowerCase() || '';
                      return !type.includes('logistic') && !type.includes('cox') && !type.includes('regression');
                    }).map(a => a.test_name) || [];

                    if (nonAdjustedAnalyses.length > 0) {
                      const result = await chatApi.executePlan(sessionId, nonAdjustedAnalyses);
                      if (result.results) {
                        result.results.forEach((r: any) => {
                          if (r.result?.success) {
                            addAnalysisResult(r.result);
                          }
                        });
                      }
                    }
                  } else {
                    // No adjusted analyses - run everything
                    const result = await chatApi.executePlan(sessionId);

                    if (result.success || result.n_executed > 0) {
                      toast.success(`${result.n_executed} analyses completed!`);
                      addChatMessage({
                        role: 'assistant',
                        content: result.message,
                        timestamp: new Date().toISOString(),
                      });

                      if (result.results) {
                        clearAnalysisResults();
                        result.results.forEach((r: any) => {
                          if (r.result?.success) {
                            addAnalysisResult(r.result);
                          }
                        });
                      }

                      setActiveTab('results');
                    } else {
                      toast.error(result.message || 'Some analyses failed');
                    }

                    setShowConfirmation(false);
                  }
                } catch (error: any) {
                  toast.error(error.response?.data?.detail || 'Failed to execute analyses');
                }
                setIsLoading(false);
              }}
              disabled={isLoading}
              className="btn-primary text-sm flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <CheckCircle className="w-4 h-4" />
              )}
              <span>{getAdjustedAnalysisFromPlan() ? 'Confirm & Select Confounders' : 'Confirm & Execute'}</span>
            </button>
          </div>
        </div>
      )}

      {/* Confounder Selector */}
      {showConfounderSelector && confounderConfig && sessionId && (
        <div className="border-t border-slate-700/50 bg-slate-900 px-6 py-4 max-h-[60vh] overflow-y-auto">
          <ConfounderSelector
            outcome={confounderConfig.outcome}
            outcomeType={confounderConfig.outcomeType}
            mainPredictor={confounderConfig.mainPredictor}
            timeCol={confounderConfig.timeCol}
            eventCol={confounderConfig.eventCol}
            onAnalysisComplete={handleConfounderAnalysisComplete}
            onClose={() => setShowConfounderSelector(false)}
          />
        </div>
      )}

      {/* Analysis Plan Quick Actions */}
      {analysisPlan && !showConfirmation && !showConfounderSelector && (
        <div className="border-t border-slate-700/50 bg-slate-900/50 backdrop-blur-sm px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
              <span className="text-sm text-slate-300">
                Plan ready • {analysisPlan.primary_analyses.length} analyses
              </span>
              {analysisPlan.confirmed && (
                <span className="badge badge-green flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" />
                  Confirmed
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {analysisPlan.require_confirmation && !analysisPlan.confirmed && (
                <button
                  onClick={() => {
                    setShowConfirmation(true);
                    setEditingMappings(analysisPlan.variable_mappings || {});
                    setExcludedAnalyses(new Set());
                  }}
                  className="btn-secondary text-sm flex items-center gap-2"
                >
                  <Edit2 className="w-4 h-4" />
                  <span>Review Plan</span>
                </button>
              )}
              {getAdjustedAnalysisFromPlan() && (
                <button
                  onClick={openConfounderSelector}
                  className="px-4 py-2 bg-purple-600/20 text-purple-300 border border-purple-500/30 rounded-xl text-sm font-medium hover:bg-purple-600/30 transition-colors flex items-center gap-2"
                >
                  <Settings className="w-4 h-4" />
                  <span>Confounders</span>
                </button>
              )}
              <button
                onClick={async () => {
                  if (!sessionId) return;

                  if (analysisPlan.require_confirmation && !analysisPlan.confirmed) {
                    setShowConfirmation(true);
                    return;
                  }

                  setIsLoading(true);
                  try {
                    const result = await chatApi.executePlan(sessionId);
                    if (result.success || result.n_executed > 0) {
                      toast.success(`${result.n_executed} analyses completed!`);

                      if (result.results) {
                        clearAnalysisResults();
                        result.results.forEach((r: any) => {
                          if (r.result?.success) {
                            addAnalysisResult(r.result);
                          }
                        });
                      }
                    } else {
                      toast.error(result.message || 'Some analyses failed');
                    }
                    setActiveTab('results');
                  } catch (error: any) {
                    toast.error(error.response?.data?.detail || 'Failed to execute analyses');
                  }
                  setIsLoading(false);
                }}
                disabled={isLoading}
                className="btn-primary text-sm"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  'Run All'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Analysis Progress */}
      {(isAnalysisRunning || analysisProgress.length > 0) && (
        <div className="border-t border-slate-700/50 bg-slate-900/50 backdrop-blur-sm px-6 py-4">
          <AnalysisProgress
            steps={analysisProgress}
            isRunning={isAnalysisRunning}
            onCancel={() => {
              setIsAnalysisRunning(false);
              toast('Analysis cancelled');
            }}
          />
        </div>
      )}

      {/* Quick Actions */}
      {showQuickActions && dataProfile && (
        <div className="border-t border-slate-700/50 bg-slate-900/50 backdrop-blur-sm px-6 py-4">
          <QuickActionBar
            onAction={handleQuickAction}
            disabled={isLoading || isAnalysisRunning}
            compact={true}
          />
        </div>
      )}

      {/* Input */}
      <div className="border-t border-slate-800/50 bg-slate-900/80 backdrop-blur-sm p-4">
        <div className="flex items-center gap-3">
          {/* Quick Actions Toggle */}
          {dataProfile && (
            <button
              onClick={() => setShowQuickActions(!showQuickActions)}
              className={clsx(
                'p-3 rounded-xl transition-all duration-200',
                showQuickActions
                  ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              )}
              title="Quick Actions"
            >
              <Wand2 className="w-5 h-5" />
            </button>
          )}

          <div className="flex-1 relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder={
                dataProfile
                  ? 'Ask a research question or describe your analysis...'
                  : 'Upload data first to start analyzing...'
              }
              disabled={!dataProfile || isLoading}
              className="input-modern pr-12"
            />
            {input && (
              <button
                onClick={() => setInput('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-slate-700 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-500" />
              </button>
            )}
          </div>
          <button
            onClick={handleSend}
            disabled={!input.trim() || !dataProfile || isLoading}
            className={clsx(
              'p-3 rounded-xl transition-all duration-200',
              input.trim() && dataProfile && !isLoading
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-500/25 hover:shadow-xl hover:shadow-indigo-500/30'
                : 'bg-slate-800 text-slate-500 cursor-not-allowed'
            )}
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
