import { useState, useEffect, useRef } from 'react';
import { Loader2, Download, Maximize2, X } from 'lucide-react';
import clsx from 'clsx';

interface InteractivePlotProps {
  plotlyJson?: string;
  base64Image?: string;
  title?: string;
  description?: string;
  onDownload?: (format: 'png' | 'svg') => void;
  compact?: boolean;
  className?: string;
}

// Dynamically load Plotly only when needed
let PlotlyPromise: Promise<typeof import('plotly.js-dist-min')> | null = null;

const getPlotly = () => {
  if (!PlotlyPromise) {
    PlotlyPromise = import('plotly.js-dist-min');
  }
  return PlotlyPromise;
};

export default function InteractivePlot({
  plotlyJson,
  base64Image,
  title,
  description,
  onDownload,
  compact = false,
  className,
}: InteractivePlotProps) {
  const plotRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (!plotlyJson || !plotRef.current) return;

    const renderPlot = async () => {
      setLoading(true);
      setError(null);

      try {
        const Plotly = await getPlotly();
        const plotData = JSON.parse(plotlyJson);

        // Configure for dark theme
        const layout = {
          ...plotData.layout,
          paper_bgcolor: 'rgba(15, 23, 42, 0)',
          plot_bgcolor: 'rgba(30, 41, 59, 0.5)',
          font: { color: '#e2e8f0', family: 'Inter, system-ui, sans-serif' },
          margin: compact ? { t: 40, r: 20, b: 40, l: 50 } : plotData.layout?.margin,
          ...(compact && { height: 300 }),
        };

        // Update axis colors for dark theme
        if (layout.xaxis) {
          layout.xaxis = {
            ...layout.xaxis,
            gridcolor: 'rgba(100, 116, 139, 0.2)',
            linecolor: '#475569',
            tickfont: { color: '#94a3b8' },
          };
        }
        if (layout.yaxis) {
          layout.yaxis = {
            ...layout.yaxis,
            gridcolor: 'rgba(100, 116, 139, 0.2)',
            linecolor: '#475569',
            tickfont: { color: '#94a3b8' },
          };
        }

        const config = {
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          displaylogo: false,
          toImageButtonOptions: {
            format: 'png',
            filename: title || 'plot',
            height: 800,
            width: 1200,
            scale: 2,
          },
        };

        await Plotly.default.newPlot(plotRef.current!, plotData.data, layout, config);
      } catch (err) {
        console.error('Failed to render plot:', err);
        setError('Failed to render interactive plot');
      } finally {
        setLoading(false);
      }
    };

    renderPlot();

    return () => {
      if (plotRef.current) {
        getPlotly().then(Plotly => {
          Plotly.default.purge(plotRef.current!);
        }).catch(() => {});
      }
    };
  }, [plotlyJson, compact, title]);

  const handleDownload = async (format: 'png' | 'svg') => {
    if (onDownload) {
      onDownload(format);
      return;
    }

    if (base64Image && format === 'png') {
      const link = document.createElement('a');
      link.href = `data:image/png;base64,${base64Image}`;
      link.download = `${title || 'plot'}.png`;
      link.click();
      return;
    }

    if (plotRef.current && plotlyJson) {
      try {
        const Plotly = await getPlotly();
        await Plotly.default.downloadImage(plotRef.current, {
          format,
          filename: title || 'plot',
          height: 800,
          width: 1200,
          scale: 2,
        });
      } catch (err) {
        console.error('Download failed:', err);
      }
    }
  };

  // Fullscreen modal
  if (isFullscreen) {
    return (
      <div className="fixed inset-0 z-50 bg-slate-900/95 backdrop-blur-sm flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <div>
            {title && <h3 className="text-lg font-semibold text-slate-100">{title}</h3>}
            {description && <p className="text-sm text-slate-400">{description}</p>}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleDownload('png')}
              className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-slate-200"
              title="Download PNG"
            >
              <Download className="w-5 h-5" />
            </button>
            <button
              onClick={() => setIsFullscreen(false)}
              className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-slate-200"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
        <div className="flex-1 p-6 overflow-auto">
          {plotlyJson ? (
            <div ref={plotRef} className="w-full h-full min-h-[600px]" />
          ) : base64Image ? (
            <img
              src={`data:image/png;base64,${base64Image}`}
              alt={title || 'Plot'}
              className="max-w-full max-h-full mx-auto"
            />
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('relative', className)}>
      {/* Header */}
      {(title || description) && !compact && (
        <div className="mb-3">
          {title && <h4 className="font-medium text-slate-200">{title}</h4>}
          {description && <p className="text-sm text-slate-400 mt-1">{description}</p>}
        </div>
      )}

      {/* Plot container */}
      <div className="relative rounded-xl overflow-hidden bg-slate-800/30 border border-slate-700/50">
        {/* Loading state */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50 z-10">
            <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="p-8 text-center text-red-400">
            <p>{error}</p>
          </div>
        )}

        {/* Interactive Plotly chart */}
        {plotlyJson && !error && (
          <div ref={plotRef} className={clsx('w-full', compact ? 'h-[300px]' : 'min-h-[400px]')} />
        )}

        {/* Static image fallback */}
        {!plotlyJson && base64Image && (
          <div className="p-4 bg-white rounded-lg m-2">
            <img
              src={`data:image/png;base64,${base64Image}`}
              alt={title || 'Plot'}
              className="w-full"
            />
          </div>
        )}

        {/* Action buttons */}
        <div className="absolute top-2 right-2 flex items-center gap-1 opacity-0 hover:opacity-100 transition-opacity">
          <button
            onClick={() => setIsFullscreen(true)}
            className="p-1.5 bg-slate-800/90 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-slate-200 transition-colors"
            title="Fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => handleDownload('png')}
            className="p-1.5 bg-slate-800/90 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-slate-200 transition-colors"
            title="Download"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
