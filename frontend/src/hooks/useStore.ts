import { create } from 'zustand';
import type { DataProfile, AnalysisPlan, AnalysisResult, ChatMessage } from '../types';
import type { AnalysisStep } from '../components/AnalysisProgress';
import type { VariableMap } from '../components/VariableMapping';

interface AppState {
  // Session
  sessionId: string | null;
  setSessionId: (id: string | null) => void;

  // Data
  dataProfile: DataProfile | null;
  setDataProfile: (profile: DataProfile | null) => void;

  // Analysis
  analysisPlan: AnalysisPlan | null;
  setAnalysisPlan: (plan: AnalysisPlan | null) => void;
  updatePlanConfirmation: (confirmed: boolean) => void;
  analysisResults: AnalysisResult[];
  addAnalysisResult: (result: AnalysisResult) => void;
  clearAnalysisResults: () => void;

  // Analysis Progress
  analysisProgress: AnalysisStep[];
  setAnalysisProgress: (steps: AnalysisStep[]) => void;
  updateAnalysisStep: (id: string, updates: Partial<AnalysisStep>) => void;
  isAnalysisRunning: boolean;
  setIsAnalysisRunning: (running: boolean) => void;

  // Variable Mapping
  variableMappings: VariableMap[];
  setVariableMappings: (mappings: VariableMap[]) => void;
  updateVariableMapping: (role: string, variable: string) => void;
  showVariableMapping: boolean;
  setShowVariableMapping: (show: boolean) => void;

  // Generated Code (for transparency)
  generatedCode: {
    python?: string;
    r?: string;
    packages?: { python?: string[]; r?: string[] };
  } | null;
  setGeneratedCode: (code: typeof initialState.generatedCode) => void;

  // Report
  resultsReport: string | null;
  setResultsReport: (report: string | null) => void;

  // Chat
  chatMessages: ChatMessage[];
  addChatMessage: (message: ChatMessage) => void;
  setChatMessages: (messages: ChatMessage[]) => void;
  clearChat: () => void;

  // UI State
  activeTab: 'data' | 'results';
  setActiveTab: (tab: 'data' | 'results') => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  showWizard: boolean;
  setShowWizard: (show: boolean) => void;
  showExportPanel: boolean;
  setShowExportPanel: (show: boolean) => void;

  // Reset
  reset: () => void;
}

const initialState = {
  sessionId: null,
  dataProfile: null,
  analysisPlan: null,
  analysisResults: [],
  analysisProgress: [] as AnalysisStep[],
  isAnalysisRunning: false,
  variableMappings: [] as VariableMap[],
  showVariableMapping: false,
  generatedCode: null as AppState['generatedCode'],
  resultsReport: null,
  chatMessages: [],
  activeTab: 'data' as const,
  isLoading: false,
  showWizard: false,
  showExportPanel: false,
};

export const useStore = create<AppState>((set) => ({
  ...initialState,

  setSessionId: (id) => set({ sessionId: id }),

  setDataProfile: (profile) => set({ dataProfile: profile }),

  setAnalysisPlan: (plan) => set({ analysisPlan: plan }),

  updatePlanConfirmation: (confirmed) =>
    set((state) => ({
      analysisPlan: state.analysisPlan
        ? { ...state.analysisPlan, confirmed, confirmed_at: confirmed ? new Date().toISOString() : undefined }
        : null,
    })),

  addAnalysisResult: (result) =>
    set((state) => ({ analysisResults: [...state.analysisResults, result] })),

  clearAnalysisResults: () => set({ analysisResults: [] }),

  // Analysis Progress
  setAnalysisProgress: (steps) => set({ analysisProgress: steps }),

  updateAnalysisStep: (id, updates) =>
    set((state) => ({
      analysisProgress: state.analysisProgress.map((step) =>
        step.id === id ? { ...step, ...updates } : step
      ),
    })),

  setIsAnalysisRunning: (running) => set({ isAnalysisRunning: running }),

  // Variable Mapping
  setVariableMappings: (mappings) => set({ variableMappings: mappings }),

  updateVariableMapping: (role, variable) =>
    set((state) => ({
      variableMappings: state.variableMappings.map((m) =>
        m.role === role ? { ...m, suggestedVariable: variable, matched: true } : m
      ),
    })),

  setShowVariableMapping: (show) => set({ showVariableMapping: show }),

  // Generated Code
  setGeneratedCode: (code) => set({ generatedCode: code }),

  setResultsReport: (report) => set({ resultsReport: report }),

  addChatMessage: (message) =>
    set((state) => ({ chatMessages: [...state.chatMessages, message] })),

  setChatMessages: (messages) => set({ chatMessages: messages }),

  clearChat: () => set({ chatMessages: [] }),

  setActiveTab: (tab) => set({ activeTab: tab }),

  setIsLoading: (loading) => set({ isLoading: loading }),

  setShowWizard: (show) => set({ showWizard: show }),

  setShowExportPanel: (show) => set({ showExportPanel: show }),

  reset: () => set(initialState),
}));
