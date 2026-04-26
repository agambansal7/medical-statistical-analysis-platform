import { useState, useRef, useEffect } from 'react';
import { useStore } from '../hooks/useStore';
import { chatApi } from '../services/api';
import {
  Send,
  Bot,
  User,
  Loader2,
  Sparkles,
  X,
  Zap,
  ArrowRight,
  Brain,
  FlaskConical,
  Wand2,
  FileText,
} from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import AnalysisPlanView from './AnalysisPlanView';
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
  } = useStore();

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPlanView, setShowPlanView] = useState(false);
  const [analysisProgress] = useState<AnalysisStep[]>([]);
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(false);
  const [showQuickActions, setShowQuickActions] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const handlePlanExecutionComplete = (results: any[]) => {
    const successCount = results.filter(r => !r.error).length;
    const failCount = results.filter(r => r.error).length;

    setShowPlanView(false);
    setActiveTab('results');

    addChatMessage({
      role: 'assistant',
      content: `Analysis complete! ${successCount} analyses completed successfully${failCount > 0 ? `, ${failCount} failed` : ''}. View the results in the Results tab.`,
      timestamp: new Date().toISOString(),
    });
  };

  const handlePlanCancel = () => {
    setShowPlanView(false);
  };

  const handlePlanConfirm = () => {
    // User wants to modify the plan - keep the view open
    toast('You can modify analyses by expanding them and adjusting settings');
  };

  const handleQuickAction = (actionId: string, _analysisType?: string) => {
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

          // Always show the plan view for review
          planSummary += `

**Review the analysis plan below.** You can adjust confounder settings for each regression analysis and run them individually or all at once.`;
          setShowPlanView(true);

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

      {/* Integrated Analysis Plan View */}
      {showPlanView && analysisPlan && sessionId && (
        <div className="border-t border-slate-700/50 bg-slate-900/50 backdrop-blur-sm max-h-[70vh] overflow-y-auto">
          <AnalysisPlanView
            plan={{
              research_question: analysisPlan.research_question || '',
              research_type: analysisPlan.research_type || 'COMPARISON',
              primary_analyses: analysisPlan.primary_analyses.map((a: any, i: number) => ({
                id: a.id || `primary_${i}`,
                test_name: a.test_name,
                analysis_type: a.api_call?.analysis_type || 'unknown',
                rationale: a.rationale || '',
                outcome: a.api_call?.parameters?.outcome || a.api_call?.parameters?.dependent,
                outcome_type: a.api_call?.parameters?.outcome_type || 'binary',
                predictor: a.api_call?.parameters?.predictor || a.api_call?.parameters?.group,
                time_col: a.api_call?.parameters?.time || a.api_call?.parameters?.time_col,
                event_col: a.api_call?.parameters?.event || a.api_call?.parameters?.event_col,
                parameters: a.api_call?.parameters,
                requires_adjustment: ['logistic_regression', 'cox_regression', 'linear_regression'].includes(
                  a.api_call?.analysis_type || ''
                ),
              })),
              secondary_analyses: (analysisPlan.secondary_analyses || []).map((a: any, i: number) => ({
                id: a.id || `secondary_${i}`,
                test_name: a.test_name,
                analysis_type: a.api_call?.analysis_type || 'unknown',
                rationale: a.rationale || '',
                outcome: a.api_call?.parameters?.outcome,
                outcome_type: a.api_call?.parameters?.outcome_type || 'binary',
                predictor: a.api_call?.parameters?.predictor,
                parameters: a.api_call?.parameters,
                requires_adjustment: ['logistic_regression', 'cox_regression', 'linear_regression'].includes(
                  a.api_call?.analysis_type || ''
                ),
              })),
            }}
            onConfirm={handlePlanConfirm}
            onCancel={handlePlanCancel}
            onExecutionComplete={handlePlanExecutionComplete}
          />
        </div>
      )}

      {/* Collapsed Plan Status Bar */}
      {analysisPlan && !showPlanView && (
        <div className="border-t border-slate-700/50 bg-slate-900/50 backdrop-blur-sm px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
              <span className="text-sm text-slate-300">
                Analysis plan ready • {analysisPlan.primary_analyses.length} analyses
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowPlanView(true)}
                className="btn-secondary text-sm flex items-center gap-2"
              >
                <FileText className="w-4 h-4" />
                <span>View Plan</span>
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
