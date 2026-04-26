import { ReactNode, useState } from 'react';
import { useStore } from '../hooks/useStore';
import {
  Database,
  FileText,
  Settings,
  HelpCircle,
  ChevronRight,
  Activity,
  FlaskConical,
  Download,
  Save,
  Wand2,
} from 'lucide-react';
import clsx from 'clsx';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { activeTab, setActiveTab, dataProfile, sessionId, setShowWizard, setShowExportPanel, analysisResults } = useStore();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const tabs = [
    { id: 'data' as const, label: 'Data', icon: Database, description: 'Upload & explore data' },
    { id: 'results' as const, label: 'Results', icon: FileText, description: 'Analysis results & plots' },
  ];

  return (
    <div className="min-h-screen max-w-full overflow-x-hidden flex">
      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed left-0 top-0 h-full z-40 transition-all duration-300 ease-in-out',
          sidebarOpen ? 'w-64' : 'w-20'
        )}
      >
        <div className="h-full glass-card rounded-none border-l-0 border-t-0 border-b-0 flex flex-col">
          {/* Logo */}
          <div className="p-6 border-b border-slate-700/50">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
                  <FlaskConical className="w-5 h-5 text-white" />
                </div>
                <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full border-2 border-slate-900 flex items-center justify-center">
                  <Activity className="w-2 h-2 text-white" />
                </div>
              </div>
              {sidebarOpen && (
                <div className="animate-fade-in">
                  <h1 className="font-bold text-lg gradient-text">StatSage</h1>
                  <p className="text-xs text-slate-500">AI-Powered Analytics</p>
                </div>
              )}
            </div>
          </div>

          {/* Toggle Button */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="absolute -right-3 top-20 w-6 h-6 bg-slate-800 border border-slate-700 rounded-full flex items-center justify-center hover:bg-slate-700 transition-colors z-50"
          >
            <ChevronRight
              className={clsx(
                'w-4 h-4 text-slate-400 transition-transform duration-300',
                !sidebarOpen && 'rotate-180'
              )}
            />
          </button>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-2">
            <div className={clsx('text-xs font-medium text-slate-500 uppercase tracking-wider mb-3', !sidebarOpen && 'sr-only')}>
              Navigation
            </div>
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={clsx(
                  'w-full flex items-center rounded-xl transition-all duration-200 group',
                  sidebarOpen ? 'px-4 py-3 space-x-3' : 'p-3 justify-center',
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border border-indigo-500/30 text-white'
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                )}
              >
                <tab.icon
                  className={clsx(
                    'w-5 h-5 flex-shrink-0 transition-colors',
                    activeTab === tab.id ? 'text-indigo-400' : 'text-slate-500 group-hover:text-slate-300'
                  )}
                />
                {sidebarOpen && (
                  <div className="text-left animate-fade-in">
                    <div className="font-medium text-sm">{tab.label}</div>
                    <div className="text-xs text-slate-500">{tab.description}</div>
                  </div>
                )}
              </button>
            ))}
          </nav>

          {/* Data Status */}
          {dataProfile && (
            <div className={clsx('px-4 py-3 border-t border-slate-700/50', !sidebarOpen && 'px-2')}>
              {sidebarOpen ? (
                <div className="animate-fade-in">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                    <span className="text-xs font-medium text-emerald-400">Data Loaded</span>
                  </div>
                  <p className="text-sm font-medium text-slate-300 truncate">{dataProfile.filename}</p>
                  <p className="text-xs text-slate-500">{dataProfile.n_rows.toLocaleString()} rows • {dataProfile.n_columns} cols</p>
                </div>
              ) : (
                <div className="flex justify-center">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                </div>
              )}
            </div>
          )}

          {/* Bottom Actions */}
          <div className="p-4 border-t border-slate-700/50 space-y-2">
            {/* Analysis Wizard Button */}
            {dataProfile && (
              <button
                onClick={() => setShowWizard(true)}
                className={clsx(
                  'w-full flex items-center rounded-xl transition-all',
                  'bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border border-indigo-500/30',
                  'text-indigo-300 hover:from-indigo-600/30 hover:to-purple-600/30',
                  sidebarOpen ? 'px-4 py-3 space-x-3' : 'p-3 justify-center'
                )}
              >
                <Wand2 className="w-5 h-5" />
                {sidebarOpen && <span className="text-sm font-medium">Analysis Wizard</span>}
              </button>
            )}

            <button className={clsx(
              'w-full flex items-center rounded-xl text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 transition-all',
              sidebarOpen ? 'px-4 py-3 space-x-3' : 'p-3 justify-center'
            )}>
              <Save className="w-5 h-5" />
              {sidebarOpen && <span className="text-sm">Save Session</span>}
            </button>

            <button
              onClick={() => setShowExportPanel(true)}
              disabled={analysisResults.length === 0}
              className={clsx(
                'w-full flex items-center rounded-xl transition-all',
                analysisResults.length > 0
                  ? 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                  : 'text-slate-600 cursor-not-allowed',
                sidebarOpen ? 'px-4 py-3 space-x-3' : 'p-3 justify-center'
              )}
            >
              <Download className="w-5 h-5" />
              {sidebarOpen && <span className="text-sm">Export Results</span>}
            </button>
          </div>

          {/* Session ID */}
          {sidebarOpen && sessionId && (
            <div className="px-4 py-3 border-t border-slate-700/50">
              <p className="text-xs text-slate-600">Session: {sessionId.slice(0, 8)}...</p>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <div
        className={clsx(
          'flex-1 min-w-0 overflow-x-hidden transition-all duration-300',
          sidebarOpen ? 'ml-64' : 'ml-20'
        )}
      >
        {/* Top Header */}
        <header className="sticky top-0 z-30 px-6 py-4 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/50">
          <div className="flex items-center justify-between">
            {/* Page Title */}
            <div>
              <h2 className="text-xl font-bold text-slate-100">
                {tabs.find(t => t.id === activeTab)?.label || 'Dashboard'}
              </h2>
              <p className="text-sm text-slate-500">
                {tabs.find(t => t.id === activeTab)?.description}
              </p>
            </div>

            {/* Right Actions */}
            <div className="flex items-center space-x-3">
              {dataProfile && (
                <div className="hidden md:flex items-center space-x-4 px-4 py-2 bg-slate-800/50 rounded-xl border border-slate-700/50">
                  <div className="flex items-center space-x-2">
                    <Database className="w-4 h-4 text-indigo-400" />
                    <span className="text-sm text-slate-300">{dataProfile.filename}</span>
                  </div>
                  <div className="h-4 w-px bg-slate-700" />
                  <span className="text-sm text-slate-400">
                    {dataProfile.n_rows.toLocaleString()} rows
                  </span>
                </div>
              )}

              <button className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors">
                <HelpCircle className="w-5 h-5" />
              </button>
              <button className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors">
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="h-[calc(100vh-73px)]">{children}</main>
      </div>
    </div>
  );
}
