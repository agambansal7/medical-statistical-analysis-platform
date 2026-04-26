import { useState } from 'react';
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  ChevronDown,
  Search,
  ArrowRight,
  Zap,
} from 'lucide-react';
import clsx from 'clsx';

export interface VariableMap {
  role: string;
  suggestedVariable: string | null;
  alternatives: string[];
  required: boolean;
  matched: boolean;
  description?: string;
}

interface VariableMappingProps {
  mappings: VariableMap[];
  availableVariables: string[];
  onUpdate: (role: string, variable: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
  isCompact?: boolean;
}

export default function VariableMapping({
  mappings,
  availableVariables,
  onUpdate,
  onConfirm,
  onCancel,
  isCompact = false,
}: VariableMappingProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [editingRole, setEditingRole] = useState<string | null>(null);

  const allRequired = mappings.filter(m => m.required).every(m => m.matched);
  const matchedCount = mappings.filter(m => m.matched).length;

  const getStatusIcon = (mapping: VariableMap) => {
    if (mapping.matched) {
      return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    }
    if (mapping.required) {
      return <XCircle className="w-4 h-4 text-red-400" />;
    }
    return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
  };

  const filteredVariables = availableVariables.filter(v =>
    v.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (isCompact) {
    return (
      <div className="space-y-2">
        {mappings.map((mapping) => (
          <div
            key={mapping.role}
            className={clsx(
              'flex items-center justify-between p-2 rounded-lg text-sm',
              mapping.matched
                ? 'bg-emerald-500/10 border border-emerald-500/30'
                : mapping.required
                ? 'bg-red-500/10 border border-red-500/30'
                : 'bg-slate-800/50 border border-slate-700/50'
            )}
          >
            <div className="flex items-center space-x-2">
              {getStatusIcon(mapping)}
              <span className="font-medium text-slate-300">{mapping.role}</span>
            </div>
            <div className="flex items-center space-x-2">
              <ArrowRight className="w-3 h-3 text-slate-500" />
              {mapping.matched ? (
                <span className="text-emerald-300">{mapping.suggestedVariable}</span>
              ) : (
                <span className="text-red-300 italic">Not found</span>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="glass-card overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-amber-600/20 to-orange-600/20 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
              <Zap className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-100">Variable Mapping</h3>
              <p className="text-xs text-slate-400">
                {matchedCount}/{mappings.length} variables matched
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={onCancel}
              className="px-3 py-1.5 text-sm text-slate-400 hover:text-slate-300 hover:bg-slate-700/50 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              disabled={!allRequired}
              className={clsx(
                'px-4 py-1.5 text-sm font-medium rounded-lg transition-colors flex items-center space-x-2',
                allRequired
                  ? 'bg-emerald-600 text-white hover:bg-emerald-500'
                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
              )}
            >
              <CheckCircle className="w-4 h-4" />
              <span>Confirm Mapping</span>
            </button>
          </div>
        </div>
      </div>

      {/* Mapping List */}
      <div className="p-4 space-y-3">
        {mappings.map((mapping) => (
          <div
            key={mapping.role}
            className={clsx(
              'p-3 rounded-xl border transition-all',
              mapping.matched
                ? 'bg-emerald-500/5 border-emerald-500/30'
                : mapping.required
                ? 'bg-red-500/5 border-red-500/30'
                : 'bg-slate-800/30 border-slate-700/50'
            )}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                <div className="mt-0.5">{getStatusIcon(mapping)}</div>
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-slate-200">{mapping.role}</span>
                    {mapping.required && (
                      <span className="text-xs text-red-400">(required)</span>
                    )}
                  </div>
                  {mapping.description && (
                    <p className="text-xs text-slate-500 mt-0.5">{mapping.description}</p>
                  )}
                </div>
              </div>

              {/* Variable Selection */}
              <div className="flex items-center space-x-2">
                <ArrowRight className="w-4 h-4 text-slate-500" />
                {editingRole === mapping.role ? (
                  <div className="relative">
                    <div className="relative">
                      <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-slate-500" />
                      <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Search variables..."
                        className="w-48 pl-7 pr-3 py-1.5 text-sm bg-slate-800 border border-slate-600 rounded-lg
                                   text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                        autoFocus
                      />
                    </div>
                    <div className="absolute z-10 top-full mt-1 w-48 max-h-48 overflow-y-auto
                                    bg-slate-800 border border-slate-600 rounded-lg shadow-xl">
                      {filteredVariables.map((variable) => (
                        <button
                          key={variable}
                          onClick={() => {
                            onUpdate(mapping.role, variable);
                            setEditingRole(null);
                            setSearchTerm('');
                          }}
                          className={clsx(
                            'w-full px-3 py-2 text-left text-sm hover:bg-slate-700 transition-colors',
                            variable === mapping.suggestedVariable
                              ? 'text-emerald-400 bg-emerald-500/10'
                              : 'text-slate-300'
                          )}
                        >
                          {variable}
                        </button>
                      ))}
                      {filteredVariables.length === 0 && (
                        <div className="px-3 py-2 text-sm text-slate-500">No matches</div>
                      )}
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={() => setEditingRole(mapping.role)}
                    className={clsx(
                      'flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
                      mapping.matched
                        ? 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30'
                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                    )}
                  >
                    <span>{mapping.suggestedVariable || 'Select variable'}</span>
                    <ChevronDown className="w-3 h-3" />
                  </button>
                )}
              </div>
            </div>

            {/* Alternatives */}
            {mapping.alternatives.length > 1 && !editingRole && (
              <div className="mt-2 ml-7 flex flex-wrap gap-1">
                <span className="text-xs text-slate-500">Alternatives:</span>
                {mapping.alternatives
                  .filter(a => a !== mapping.suggestedVariable)
                  .slice(0, 3)
                  .map((alt) => (
                    <button
                      key={alt}
                      onClick={() => onUpdate(mapping.role, alt)}
                      className="px-2 py-0.5 text-xs bg-slate-700/50 text-slate-400 rounded hover:bg-slate-600 hover:text-slate-300 transition-colors"
                    >
                      {alt}
                    </button>
                  ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Warning if not all required mapped */}
      {!allRequired && (
        <div className="px-4 py-3 bg-red-500/10 border-t border-red-500/30">
          <div className="flex items-center space-x-2 text-sm text-red-300">
            <AlertTriangle className="w-4 h-4" />
            <span>Some required variables are not mapped. Please select variables for all required fields.</span>
          </div>
        </div>
      )}
    </div>
  );
}

// Quick inline mapping status
export function MappingStatusBadge({ mappings }: { mappings: VariableMap[] }) {
  const matched = mappings.filter(m => m.matched).length;
  const total = mappings.length;
  const allMatched = matched === total;

  return (
    <div
      className={clsx(
        'flex items-center space-x-1.5 px-2 py-1 rounded-full text-xs font-medium',
        allMatched
          ? 'bg-emerald-500/20 text-emerald-400'
          : 'bg-amber-500/20 text-amber-400'
      )}
    >
      {allMatched ? (
        <CheckCircle className="w-3 h-3" />
      ) : (
        <AlertTriangle className="w-3 h-3" />
      )}
      <span>{matched}/{total} mapped</span>
    </div>
  );
}
