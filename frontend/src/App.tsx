import { useEffect } from 'react';
import { useStore } from './hooks/useStore';
import { sessionApi, exportApi } from './services/api';
import Layout from './components/Layout';
import DataPanel from './components/DataPanel';
import ChatPanel from './components/ChatPanel';
import StreamlinedResultsPanel from './components/StreamlinedResultsPanel';
import AnalysisWizard from './components/AnalysisWizard';
import ExportPanel from './components/ExportPanel';
import toast, { Toaster } from 'react-hot-toast';

function App() {
  const {
    sessionId,
    setSessionId,
    activeTab,
    dataProfile,
    analysisResults,
    showWizard,
    setShowWizard,
    showExportPanel,
    setShowExportPanel,
  } = useStore();

  useEffect(() => {
    const initSession = async () => {
      if (!sessionId) {
        try {
          const newSessionId = await sessionApi.create();
          setSessionId(newSessionId);
        } catch (error) {
          toast.error('Failed to create session');
          console.error(error);
        }
      }
    };
    initSession();
  }, [sessionId, setSessionId]);

  const handleWizardComplete = (config: any) => {
    setShowWizard(false);
    // Convert wizard config to a research question and send to chat
    // Future: Send this to chat - `Analyze ${config.outcome} (${config.outcomeType}) comparing ${config.exposure} groups. Include ${config.analyses.join(', ')}.`
    console.log('Wizard config:', config);
    toast.success('Analysis plan created!');
  };

  const handleExport = async (format: string, options: any): Promise<Blob | null> => {
    if (!sessionId) return null;
    try {
      const blob = await exportApi.generateReport(sessionId, format, options);
      return blob;
    } catch (error) {
      console.error('Export failed:', error);
      return null;
    }
  };

  // Get variables for wizard
  const wizardVariables = dataProfile?.variables.map(v => ({
    name: v.name,
    type: v.statistical_type,
  })) || [];

  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1e293b',
            color: '#f1f5f9',
            border: '1px solid #334155',
            borderRadius: '12px',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#f1f5f9',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#f1f5f9',
            },
          },
        }}
      />

      <Layout>
        <div className="flex h-full w-full">
          {/* Left Panel - Data & Analysis */}
          <div className="w-1/2 min-w-0 border-r border-slate-800/50 overflow-auto bg-slate-950/50">
            {activeTab === 'data' && <DataPanel />}
            {activeTab === 'results' && <StreamlinedResultsPanel />}
          </div>

          {/* Right Panel - Chat */}
          <div className="w-1/2 min-w-0 overflow-hidden">
            <ChatPanel />
          </div>
        </div>
      </Layout>

      {/* Analysis Wizard Modal */}
      {showWizard && dataProfile && (
        <AnalysisWizard
          variables={wizardVariables}
          onComplete={handleWizardComplete}
          onClose={() => setShowWizard(false)}
        />
      )}

      {/* Export Panel Modal */}
      <ExportPanel
        isOpen={showExportPanel}
        onClose={() => setShowExportPanel(false)}
        onExport={handleExport}
        hasResults={analysisResults.length > 0}
        hasFigures={false} // Would need to track generated figures
      />
    </>
  );
}

export default App;
