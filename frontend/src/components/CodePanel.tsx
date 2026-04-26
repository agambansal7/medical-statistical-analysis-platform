import { useState } from 'react';
import {
  Code,
  Copy,
  Check,
  Download,
  ChevronDown,
  ChevronRight,
  FileCode,
  Terminal,
} from 'lucide-react';
import clsx from 'clsx';
import toast from 'react-hot-toast';

interface CodePanelProps {
  pythonCode?: string;
  rCode?: string;
  analysisType: string;
  packages?: {
    python?: string[];
    r?: string[];
  };
  isCollapsible?: boolean;
  defaultExpanded?: boolean;
}

export default function CodePanel({
  pythonCode,
  rCode,
  analysisType,
  packages,
  isCollapsible = true,
  defaultExpanded = false,
}: CodePanelProps) {
  const [activeTab, setActiveTab] = useState<'python' | 'r'>('python');
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(defaultExpanded);

  const currentCode = activeTab === 'python' ? pythonCode : rCode;
  const currentPackages = activeTab === 'python' ? packages?.python : packages?.r;

  const copyToClipboard = async () => {
    if (!currentCode) return;
    try {
      await navigator.clipboard.writeText(currentCode);
      setCopied(true);
      toast.success('Code copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error('Failed to copy code');
    }
  };

  const downloadCode = () => {
    if (!currentCode) return;
    const extension = activeTab === 'python' ? 'py' : 'R';
    const filename = `${analysisType.replace(/\s+/g, '_').toLowerCase()}.${extension}`;
    const blob = new Blob([currentCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success(`Downloaded ${filename}`);
  };

  const downloadNotebook = () => {
    if (!pythonCode) return;

    // Create Jupyter notebook structure
    const notebook = {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {
        kernelspec: {
          display_name: 'Python 3',
          language: 'python',
          name: 'python3'
        }
      },
      cells: [
        {
          cell_type: 'markdown',
          metadata: {},
          source: [`# ${analysisType}\n\nGenerated code for statistical analysis.`]
        },
        {
          cell_type: 'code',
          execution_count: null,
          metadata: {},
          outputs: [],
          source: [pythonCode]
        }
      ]
    };

    const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${analysisType.replace(/\s+/g, '_').toLowerCase()}.ipynb`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Downloaded Jupyter notebook');
  };

  if (!pythonCode && !rCode) return null;

  const header = (
    <div
      className={clsx(
        'flex items-center justify-between px-4 py-3',
        isCollapsible && 'cursor-pointer hover:bg-slate-800/30 transition-colors'
      )}
      onClick={() => isCollapsible && setExpanded(!expanded)}
    >
      <div className="flex items-center space-x-3">
        {isCollapsible && (
          expanded ? (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-slate-400" />
          )
        )}
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
          <Code className="w-4 h-4 text-white" />
        </div>
        <div>
          <h4 className="font-medium text-slate-200">Reproducible Code</h4>
          <p className="text-xs text-slate-500">Python & R implementations</p>
        </div>
      </div>
      {!isCollapsible || expanded ? (
        <div className="flex items-center space-x-2" onClick={e => e.stopPropagation()}>
          <button
            onClick={copyToClipboard}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            title="Copy code"
          >
            {copied ? (
              <Check className="w-4 h-4 text-emerald-400" />
            ) : (
              <Copy className="w-4 h-4 text-slate-400" />
            )}
          </button>
          <button
            onClick={downloadCode}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            title="Download script"
          >
            <Download className="w-4 h-4 text-slate-400" />
          </button>
          {activeTab === 'python' && (
            <button
              onClick={downloadNotebook}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              title="Download Jupyter notebook"
            >
              <FileCode className="w-4 h-4 text-slate-400" />
            </button>
          )}
        </div>
      ) : null}
    </div>
  );

  return (
    <div className="glass-card overflow-hidden border-cyan-500/20">
      {header}

      {(!isCollapsible || expanded) && (
        <div className="border-t border-slate-700/50">
          {/* Language Tabs */}
          <div className="flex border-b border-slate-700/50">
            <button
              onClick={() => setActiveTab('python')}
              className={clsx(
                'flex items-center space-x-2 px-4 py-2.5 text-sm font-medium transition-colors',
                activeTab === 'python'
                  ? 'text-cyan-400 border-b-2 border-cyan-400 bg-cyan-500/5'
                  : 'text-slate-400 hover:text-slate-300'
              )}
            >
              <Terminal className="w-4 h-4" />
              <span>Python</span>
            </button>
            <button
              onClick={() => setActiveTab('r')}
              className={clsx(
                'flex items-center space-x-2 px-4 py-2.5 text-sm font-medium transition-colors',
                activeTab === 'r'
                  ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-500/5'
                  : 'text-slate-400 hover:text-slate-300'
              )}
            >
              <Code className="w-4 h-4" />
              <span>R</span>
            </button>
          </div>

          {/* Package Requirements */}
          {currentPackages && currentPackages.length > 0 && (
            <div className="px-4 py-2 bg-slate-800/30 border-b border-slate-700/50">
              <div className="flex items-center space-x-2 text-xs text-slate-400">
                <span>Required packages:</span>
                <div className="flex flex-wrap gap-1">
                  {currentPackages.map((pkg, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 bg-slate-700/50 rounded text-slate-300"
                    >
                      {pkg}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Code Block */}
          <div className="relative">
            <pre className="p-4 overflow-x-auto text-sm font-mono leading-relaxed max-h-96 overflow-y-auto">
              <code className="text-slate-300">
                {currentCode || `No ${activeTab === 'python' ? 'Python' : 'R'} code available`}
              </code>
            </pre>

            {/* Line Numbers */}
            <div className="absolute left-0 top-0 p-4 select-none pointer-events-none">
              {currentCode?.split('\n').map((_, i) => (
                <div key={i} className="text-xs text-slate-600 leading-relaxed h-[1.5rem]">
                  {i + 1}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Inline code viewer for results
export function InlineCodeButton({
  onClick,
  hasCode
}: {
  onClick: () => void;
  hasCode: boolean;
}) {
  if (!hasCode) return null;

  return (
    <button
      onClick={onClick}
      className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-cyan-400
                 bg-cyan-500/10 border border-cyan-500/30 rounded-lg
                 hover:bg-cyan-500/20 transition-colors"
    >
      <Code className="w-3.5 h-3.5" />
      <span>View Code</span>
    </button>
  );
}
