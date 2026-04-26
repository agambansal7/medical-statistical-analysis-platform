// Session types
export interface Session {
  session_id: string;
  created_at: string;
  data_loaded: boolean;
  filename?: string;
  n_analyses_performed: number;
}

// Data types
export interface VariableInfo {
  name: string;
  dtype: string;
  statistical_type: 'continuous' | 'categorical' | 'binary' | 'ordinal' | 'datetime' | 'text';
  n_total: number;
  n_missing: number;
  missing_pct: number;
  n_unique: number;
  mean?: number;
  std?: number;
  median?: number;
  categories?: string[];
}

export interface DataProfile {
  session_id: string;
  filename: string;
  n_rows: number;
  n_columns: number;
  n_continuous: number;
  n_categorical: number;
  n_binary: number;
  variables: VariableInfo[];
  potential_outcomes: string[];
  potential_groups: string[];
  warnings: string[];
}

// Analysis types
export interface AnalysisPlanItem {
  test_name: string;
  category: string;
  priority: number;
  rationale: string;
  variables: Record<string, string>;
  assumptions: string[];
  api_call?: {
    analysis_type: string;
    parameters: Record<string, unknown>;
  };
}

export interface AnalysisPlan {
  research_question: string;
  research_type: string;
  primary_analyses: AnalysisPlanItem[];
  secondary_analyses: AnalysisPlanItem[];
  assumption_checks: string[];
  visualizations: string[];
  notes: string[];
  // Validation fields
  variable_warnings?: string[];
  variable_mappings?: Record<string, string>;
  validated?: boolean;
  available_columns?: string[];
  // Confirmation fields
  require_confirmation?: boolean;
  confirmed?: boolean;
  confirmed_at?: string;
}

export interface AnalysisResult {
  success: boolean;
  analysis_id: string;
  analysis_type: string;
  results?: Record<string, unknown>;
  error?: string;
}

// Chat types
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

// Visualization types
export interface VisualizationSuggestion {
  type: string;
  description: string;
  params: Record<string, unknown>;
}

// Analysis Progress types
export interface AnalysisStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration?: number;
  error?: string;
}

// Variable Mapping types
export interface VariableMap {
  role: string;
  suggestedVariable: string | null;
  alternatives: string[];
  required: boolean;
  matched: boolean;
  description?: string;
}

// Generated Code types
export interface GeneratedCode {
  python?: string;
  r?: string;
  packages?: {
    python?: string[];
    r?: string[];
  };
  methodology?: string;
}

// Export types
export interface ExportOptions {
  includeMethods: boolean;
  includeResults: boolean;
  includeFigures: boolean;
  includeTables: boolean;
  includeCode: boolean;
  includeInterpretations: boolean;
  journalStyle: string;
}

// Wizard Config types
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
