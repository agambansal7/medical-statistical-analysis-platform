import axios from 'axios';
import type { DataProfile, AnalysisPlan, AnalysisResult, ChatMessage, VisualizationSuggestion } from '../types';

const API_BASE = '/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Session endpoints
export const sessionApi = {
  create: async () => {
    const { data } = await api.post('/session/create');
    return data.session_id as string;
  },

  get: async (sessionId: string) => {
    const { data } = await api.get(`/session/${sessionId}`);
    return data;
  },

  getSummary: async (sessionId: string) => {
    const { data } = await api.get(`/session/${sessionId}/summary`);
    return data.summary;
  },

  list: async () => {
    const { data } = await api.get('/session/');
    return data.sessions;
  },

  delete: async (sessionId: string) => {
    await api.delete(`/session/${sessionId}`);
  },
};

// Data endpoints
export const dataApi = {
  upload: async (file: File, sessionId?: string) => {
    const formData = new FormData();
    formData.append('file', file);

    // Send session_id as query parameter, not form data
    const url = sessionId ? `/data/upload?session_id=${sessionId}` : '/data/upload';

    const { data } = await api.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  getProfile: async (sessionId: string): Promise<DataProfile> => {
    const { data } = await api.get(`/data/profile/${sessionId}`);
    return data.profile;
  },

  getPreview: async (sessionId: string, nRows = 20) => {
    const { data } = await api.get(`/data/preview/${sessionId}?n_rows=${nRows}`);
    return data;
  },

  getColumns: async (sessionId: string) => {
    const { data } = await api.get(`/data/columns/${sessionId}`);
    return data.columns;
  },
};

// Analysis endpoints
export const analysisApi = {
  run: async (sessionId: string, analysisType: string, parameters: Record<string, unknown>): Promise<AnalysisResult> => {
    const { data } = await api.post('/analysis/run', {
      session_id: sessionId,
      analysis_type: analysisType,
      parameters,
    });
    return data;
  },

  getTypes: async () => {
    const { data } = await api.get('/analysis/types');
    return data.analysis_types;
  },

  getHistory: async (sessionId: string) => {
    const { data } = await api.get(`/analysis/history/${sessionId}`);
    return data.analyses;
  },

  interpret: async (sessionId: string, analysisId: string) => {
    const { data } = await api.post(`/analysis/interpret?session_id=${sessionId}&analysis_id=${analysisId}`);
    return data.interpretation;
  },
};

// Plan modification types
export interface PlanModifications {
  variable_mappings?: Record<string, string>;
  exclude_analyses?: string[];
  additional_covariates?: string[];
}

// Chat endpoints
export const chatApi = {
  sendMessage: async (sessionId: string, message: string, context?: Record<string, unknown>) => {
    const { data } = await api.post('/chat/message', {
      session_id: sessionId,
      message,
      context,
    });
    return data;
  },

  analyzeQuestion: async (sessionId: string, question: string): Promise<{
    success: boolean;
    plan?: AnalysisPlan;
    require_confirmation?: boolean;
  }> => {
    const { data } = await api.post('/chat/research-question', {
      session_id: sessionId,
      question,
    });
    return data;
  },

  getPlan: async (sessionId: string): Promise<{
    success: boolean;
    plan?: AnalysisPlan;
    confirmed?: boolean;
  }> => {
    const { data } = await api.get(`/chat/plan/${sessionId}`);
    return data;
  },

  confirmPlan: async (sessionId: string, modifications?: PlanModifications): Promise<{
    success: boolean;
    message: string;
    plan?: AnalysisPlan;
  }> => {
    const { data } = await api.post(`/chat/confirm-plan/${sessionId}`, modifications || null);
    return data;
  },

  getHistory: async (sessionId: string): Promise<ChatMessage[]> => {
    const { data } = await api.get(`/chat/history/${sessionId}`);
    return data.history;
  },

  executePlan: async (sessionId: string, analyses?: string[]): Promise<{
    success: boolean;
    message: string;
    n_executed: number;
    n_total: number;
    results: Array<{
      test_name: string;
      rationale?: string;
      result: AnalysisResult;
    }>;
    errors?: Array<{ test_name: string; error: string }>;
  }> => {
    // Send analyses as request body
    const { data } = await api.post(`/chat/execute-plan/${sessionId}`, { analyses: analyses || null });
    return data;
  },

  generateReport: async (sessionId: string): Promise<{
    success: boolean;
    report?: string;
    research_question?: string;
    n_analyses?: number;
    message?: string;
  }> => {
    const { data } = await api.post(`/chat/generate-report/${sessionId}`);
    return data;
  },

  getReport: async (sessionId: string): Promise<{
    success: boolean;
    report?: string;
    message?: string;
  }> => {
    const { data } = await api.get(`/chat/report/${sessionId}`);
    return data;
  },

  getConfounders: async (
    sessionId: string,
    outcome?: string,
    outcomeType?: string,
    timeCol?: string,
    eventCol?: string
  ): Promise<{
    success: boolean;
    outcome?: string;
    outcome_type?: string;
    confounders: Array<{
      variable: string;
      n_valid: number;
      missing_pct: number;
      type: string;
      test?: string;
      statistic?: number;
      p_value?: number;
      significant?: boolean;
      test_error?: string;
    }>;
    all_columns: string[];
  }> => {
    const params = new URLSearchParams();
    if (outcome) params.append('outcome', outcome);
    if (outcomeType) params.append('outcome_type', outcomeType);
    if (timeCol) params.append('time_col', timeCol);
    if (eventCol) params.append('event_col', eventCol);
    const { data } = await api.get(`/chat/confounders/${sessionId}?${params.toString()}`);
    return data;
  },

  runAdjustedAnalysis: async (sessionId: string, request: {
    analysis_type: 'logistic_regression' | 'cox_regression';
    outcome: string;
    predictors: string[];
    confounders: string[];
    time?: string;
    event?: string;
  }): Promise<{
    success: boolean;
    results?: Record<string, unknown>;
    error?: string;
  }> => {
    const { data } = await api.post(`/chat/run-adjusted-analysis/${sessionId}`, request);
    return data;
  },
};

// Visualization endpoints
export const vizApi = {
  create: async (sessionId: string, plotType: string, variables: Record<string, string>, options?: Record<string, unknown>) => {
    const { data } = await api.post('/viz/create', {
      session_id: sessionId,
      plot_type: plotType,
      variables,
      options,
    });
    return data;
  },

  getTypes: async () => {
    const { data } = await api.get('/viz/types');
    return data.visualization_types;
  },

  getSuggestions: async (sessionId: string): Promise<VisualizationSuggestion[]> => {
    const { data } = await api.get(`/viz/suggestions/${sessionId}`);
    return data.suggestions;
  },
};

// Code Generation endpoints
export const codeApi = {
  generate: async (
    sessionId: string,
    analysisType: string,
    parameters: Record<string, unknown>,
    languages: string[] = ['python', 'r']
  ): Promise<{
    success: boolean;
    python_code?: string;
    r_code?: string;
    python_packages?: string[];
    r_packages?: string[];
    methodology?: string;
  }> => {
    const { data } = await api.post('/code/generate', {
      session_id: sessionId,
      analysis_type: analysisType,
      parameters,
      languages,
    });
    return data;
  },

  getForAnalysis: async (sessionId: string, analysisId: string): Promise<{
    success: boolean;
    python_code?: string;
    r_code?: string;
    python_packages?: string[];
    r_packages?: string[];
  }> => {
    const { data } = await api.get(`/code/analysis/${sessionId}/${analysisId}`);
    return data;
  },

  downloadNotebook: async (sessionId: string, analysisId: string): Promise<Blob> => {
    const response = await api.get(`/code/notebook/${sessionId}/${analysisId}`, {
      responseType: 'blob',
    });
    return response.data;
  },
};

// Export endpoints
export interface ExportOptions {
  includeMethods: boolean;
  includeResults: boolean;
  includeFigures: boolean;
  includeTables: boolean;
  includeCode: boolean;
  includeInterpretations: boolean;
  journalStyle: string;
}

export const exportApi = {
  generateReport: async (
    sessionId: string,
    format: string,
    options: ExportOptions
  ): Promise<Blob> => {
    const response = await api.post(
      `/export/report/${sessionId}`,
      { format, options },
      { responseType: 'blob' }
    );
    return response.data;
  },

  exportTables: async (sessionId: string, format: 'xlsx' | 'csv'): Promise<Blob> => {
    const response = await api.get(`/export/tables/${sessionId}?format=${format}`, {
      responseType: 'blob',
    });
    return response.data;
  },

  exportFigures: async (sessionId: string, format: 'png' | 'pdf' | 'svg'): Promise<Blob> => {
    const response = await api.get(`/export/figures/${sessionId}?format=${format}`, {
      responseType: 'blob',
    });
    return response.data;
  },

  getPreview: async (sessionId: string, format: string): Promise<{
    success: boolean;
    preview?: string;
    n_pages?: number;
  }> => {
    const { data } = await api.get(`/export/preview/${sessionId}?format=${format}`);
    return data;
  },
};

// Quick Analysis endpoints (for wizard and quick actions)
export const quickAnalysisApi = {
  runTable1: async (sessionId: string, groupVar?: string): Promise<{
    success: boolean;
    result?: Record<string, unknown>;
    error?: string;
  }> => {
    const { data } = await api.post(`/analysis/quick/table1/${sessionId}`, { group_var: groupVar });
    return data;
  },

  runComparison: async (
    sessionId: string,
    outcome: string,
    group: string,
    outcomeType: 'binary' | 'continuous'
  ): Promise<{
    success: boolean;
    result?: Record<string, unknown>;
    error?: string;
  }> => {
    const { data } = await api.post(`/analysis/quick/comparison/${sessionId}`, {
      outcome,
      group,
      outcome_type: outcomeType,
    });
    return data;
  },

  runSurvival: async (
    sessionId: string,
    timeVar: string,
    eventVar: string,
    groupVar?: string
  ): Promise<{
    success: boolean;
    km_result?: Record<string, unknown>;
    cox_result?: Record<string, unknown>;
    error?: string;
  }> => {
    const { data } = await api.post(`/analysis/quick/survival/${sessionId}`, {
      time_var: timeVar,
      event_var: eventVar,
      group_var: groupVar,
    });
    return data;
  },

  detectAnalyses: async (sessionId: string): Promise<{
    success: boolean;
    suggested_analyses: Array<{
      type: string;
      name: string;
      description: string;
      parameters: Record<string, unknown>;
    }>;
    study_type?: string;
  }> => {
    const { data } = await api.get(`/analysis/quick/detect/${sessionId}`);
    return data;
  },
};

export default api;
