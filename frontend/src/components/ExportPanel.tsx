import { useState } from 'react';
import {
  FileText,
  FileCode,
  Download,
  Loader2,
  Check,
  ChevronDown,
  Image,
  Table2,
  FileSpreadsheet,
  Presentation,
  X,
  Settings,
  BookOpen,
} from 'lucide-react';
import clsx from 'clsx';
import toast from 'react-hot-toast';

interface ExportFormat {
  id: string;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  extension: string;
  color: string;
}

const EXPORT_FORMATS: ExportFormat[] = [
  {
    id: 'docx',
    label: 'Microsoft Word',
    description: 'Editable document with tables and figures',
    icon: FileText,
    extension: '.docx',
    color: 'blue',
  },
  {
    id: 'pdf',
    label: 'PDF',
    description: 'Publication-ready PDF document',
    icon: BookOpen,
    extension: '.pdf',
    color: 'red',
  },
  {
    id: 'latex',
    label: 'LaTeX',
    description: 'For academic publication',
    icon: FileCode,
    extension: '.tex',
    color: 'emerald',
  },
  {
    id: 'html',
    label: 'HTML Report',
    description: 'Interactive web-based report',
    icon: FileCode,
    extension: '.html',
    color: 'orange',
  },
  {
    id: 'pptx',
    label: 'PowerPoint',
    description: 'Slides with key findings',
    icon: Presentation,
    extension: '.pptx',
    color: 'purple',
  },
  {
    id: 'xlsx',
    label: 'Excel Tables',
    description: 'All tables in spreadsheet format',
    icon: FileSpreadsheet,
    extension: '.xlsx',
    color: 'green',
  },
];

interface ExportOptions {
  includeMethods: boolean;
  includeResults: boolean;
  includeFigures: boolean;
  includeTables: boolean;
  includeCode: boolean;
  includeInterpretations: boolean;
  journalStyle: string;
}

interface ExportPanelProps {
  onExport: (format: string, options: ExportOptions) => Promise<Blob | null>;
  isOpen: boolean;
  onClose: () => void;
  hasResults: boolean;
  hasFigures: boolean;
}

export default function ExportPanel({
  onExport,
  isOpen,
  onClose,
  hasResults,
  hasFigures,
}: ExportPanelProps) {
  const [selectedFormat, setSelectedFormat] = useState<string>('docx');
  const [isExporting, setIsExporting] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [options, setOptions] = useState<ExportOptions>({
    includeMethods: true,
    includeResults: true,
    includeFigures: true,
    includeTables: true,
    includeCode: false,
    includeInterpretations: true,
    journalStyle: 'nejm',
  });

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const blob = await onExport(selectedFormat, options);
      if (blob) {
        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const format = EXPORT_FORMATS.find((f) => f.id === selectedFormat);
        a.download = `analysis_report${format?.extension || '.docx'}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast.success('Report exported successfully!');
        onClose();
      }
    } catch (error) {
      toast.error('Failed to export report');
      console.error(error);
    } finally {
      setIsExporting(false);
    }
  };

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; text: string; border: string }> = {
      blue: { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/30' },
      red: { bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/30' },
      emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30' },
      orange: { bg: 'bg-orange-500/10', text: 'text-orange-400', border: 'border-orange-500/30' },
      purple: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30' },
      green: { bg: 'bg-green-500/10', text: 'text-green-400', border: 'border-green-500/30' },
    };
    return colors[color] || colors.blue;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg mx-4 glass-card overflow-hidden animate-fade-in">
        {/* Header */}
        <div className="px-6 py-4 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border-b border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Download className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="font-semibold text-slate-100">Export Report</h2>
                <p className="text-xs text-slate-400">Choose format and options</p>
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

        {/* Format Selection */}
        <div className="p-6 space-y-6">
          <div>
            <h3 className="text-sm font-medium text-slate-300 mb-3">Export Format</h3>
            <div className="grid grid-cols-2 gap-3">
              {EXPORT_FORMATS.map((format) => {
                const colors = getColorClasses(format.color);
                const isSelected = selectedFormat === format.id;
                return (
                  <button
                    key={format.id}
                    onClick={() => setSelectedFormat(format.id)}
                    className={clsx(
                      'flex items-center space-x-3 p-3 rounded-xl border transition-all text-left',
                      isSelected
                        ? `${colors.bg} ${colors.border}`
                        : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800/50'
                    )}
                  >
                    <div
                      className={clsx(
                        'w-10 h-10 rounded-lg flex items-center justify-center',
                        isSelected ? colors.bg : 'bg-slate-700/50'
                      )}
                    >
                      <format.icon
                        className={clsx('w-5 h-5', isSelected ? colors.text : 'text-slate-400')}
                      />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div
                        className={clsx(
                          'font-medium text-sm',
                          isSelected ? colors.text : 'text-slate-200'
                        )}
                      >
                        {format.label}
                      </div>
                      <div className="text-xs text-slate-500 truncate">{format.description}</div>
                    </div>
                    {isSelected && <Check className={clsx('w-4 h-4', colors.text)} />}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Options */}
          <div>
            <button
              onClick={() => setShowOptions(!showOptions)}
              className="flex items-center space-x-2 text-sm text-slate-300 hover:text-slate-100 transition-colors"
            >
              <Settings className="w-4 h-4" />
              <span>Export Options</span>
              <ChevronDown
                className={clsx('w-4 h-4 transition-transform', showOptions && 'rotate-180')}
              />
            </button>

            {showOptions && (
              <div className="mt-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50 space-y-3">
                {/* Include sections */}
                <div className="space-y-2">
                  <p className="text-xs text-slate-500 uppercase tracking-wider">Include Sections</p>
                  {[
                    { key: 'includeMethods', label: 'Methods section', icon: FileText },
                    { key: 'includeResults', label: 'Results section', icon: FileText },
                    { key: 'includeTables', label: 'Tables', icon: Table2 },
                    { key: 'includeFigures', label: 'Figures', icon: Image, disabled: !hasFigures },
                    { key: 'includeInterpretations', label: 'Interpretations', icon: BookOpen },
                    { key: 'includeCode', label: 'Reproducible code', icon: FileCode },
                  ].map((item) => (
                    <label
                      key={item.key}
                      className={clsx(
                        'flex items-center space-x-3 p-2 rounded-lg cursor-pointer hover:bg-slate-700/30',
                        item.disabled && 'opacity-50 cursor-not-allowed'
                      )}
                    >
                      <input
                        type="checkbox"
                        checked={options[item.key as keyof ExportOptions] as boolean}
                        onChange={(e) =>
                          setOptions({ ...options, [item.key]: e.target.checked })
                        }
                        disabled={item.disabled}
                        className="rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-500"
                      />
                      <item.icon className="w-4 h-4 text-slate-400" />
                      <span className="text-sm text-slate-300">{item.label}</span>
                    </label>
                  ))}
                </div>

                {/* Journal Style */}
                <div>
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">
                    Journal Formatting
                  </p>
                  <select
                    value={options.journalStyle}
                    onChange={(e) => setOptions({ ...options, journalStyle: e.target.value })}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg
                               text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="nejm">NEJM Style</option>
                    <option value="jama">JAMA Style</option>
                    <option value="lancet">Lancet Style</option>
                    <option value="bmj">BMJ Style</option>
                    <option value="annals">Annals of Internal Medicine</option>
                    <option value="apa">APA Style</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Warning if no results */}
          {!hasResults && (
            <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl">
              <div className="flex items-center space-x-2 text-sm text-amber-300">
                <Settings className="w-4 h-4" />
                <span>No analysis results to export. Run analyses first.</span>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-700/50 flex items-center justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-slate-400 hover:text-slate-300 hover:bg-slate-700/50 rounded-xl transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={isExporting || !hasResults}
            className={clsx(
              'flex items-center space-x-2 px-6 py-2 rounded-xl font-medium transition-colors',
              hasResults
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:shadow-lg hover:shadow-indigo-500/30'
                : 'bg-slate-700 text-slate-500 cursor-not-allowed'
            )}
          >
            {isExporting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Exporting...</span>
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                <span>Export</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// Quick export button component
export function QuickExportButton({
  onClick,
  format = 'docx',
  disabled = false,
}: {
  onClick: () => void;
  format?: string;
  disabled?: boolean;
}) {
  const formatLabels: Record<string, string> = {
    docx: 'Word',
    pdf: 'PDF',
    latex: 'LaTeX',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
        disabled
          ? 'bg-slate-700/50 text-slate-500 cursor-not-allowed'
          : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/30'
      )}
    >
      <Download className="w-4 h-4" />
      <span>Export {formatLabels[format] || format}</span>
    </button>
  );
}
