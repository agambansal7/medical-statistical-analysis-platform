import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useStore } from '../hooks/useStore';
import { dataApi } from '../services/api';
import {
  Upload,
  FileSpreadsheet,
  AlertTriangle,
  CheckCircle,
  Info,
  Database,
  TrendingUp,
  Hash,
  ToggleLeft,
  RefreshCw,
} from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';

export default function DataPanel() {
  const { sessionId, setSessionId, dataProfile, setDataProfile, setIsLoading, isLoading } = useStore();
  const [preview, setPreview] = useState<Record<string, unknown>[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!sessionId || acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setIsLoading(true);
    setUploadProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => Math.min(prev + 10, 90));
    }, 100);

    try {
      const response = await dataApi.upload(file, sessionId);

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.success) {
        if (response.session_id) {
          setSessionId(response.session_id);
        }
        setDataProfile(response.profile);
        setPreview(response.preview || []);
        toast.success(`Successfully loaded ${response.profile.n_rows.toLocaleString()} rows`);
      } else {
        toast.error(response.message || 'Upload failed');
      }
    } catch (error) {
      clearInterval(progressInterval);
      toast.error('Failed to upload file');
      console.error(error);
    } finally {
      setIsLoading(false);
      setTimeout(() => setUploadProgress(0), 500);
    }
  }, [sessionId, setSessionId, setDataProfile, setIsLoading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    maxFiles: 1,
  });

  if (!dataProfile) {
    return (
      <div className="h-full flex items-center justify-center p-8">
        <div className="w-full max-w-xl">
          <div
            {...getRootProps()}
            className={clsx(
              'upload-zone text-center',
              isDragActive && 'active'
            )}
          >
            <input {...getInputProps()} />

            <div className="relative mx-auto w-20 h-20 mb-6">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl opacity-20 blur-xl animate-pulse" />
              <div className="relative w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-xl shadow-indigo-500/25">
                <Upload className={clsx('w-8 h-8 text-white', isDragActive && 'animate-bounce')} />
              </div>
            </div>

            <h3 className="text-xl font-bold text-slate-100 mb-2">
              {isDragActive ? 'Drop your file here' : 'Upload your data'}
            </h3>
            <p className="text-slate-400 mb-6">
              Drag and drop your file, or click to browse
            </p>

            <div className="flex items-center justify-center gap-4 text-sm text-slate-500">
              <span className="flex items-center gap-1">
                <FileSpreadsheet className="w-4 h-4" />
                CSV
              </span>
              <span className="w-1 h-1 bg-slate-600 rounded-full" />
              <span className="flex items-center gap-1">
                <FileSpreadsheet className="w-4 h-4" />
                Excel
              </span>
            </div>

            {isLoading && uploadProgress > 0 && (
              <div className="mt-6">
                <div className="progress-bar">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="text-sm text-slate-400 mt-2">Uploading... {uploadProgress}%</p>
              </div>
            )}
          </div>

          {/* Features */}
          <div className="mt-8 grid grid-cols-3 gap-4">
            {[
              { icon: TrendingUp, label: 'Auto-detect types' },
              { icon: Database, label: 'Smart profiling' },
              { icon: CheckCircle, label: 'Data validation' },
            ].map((feature, i) => (
              <div key={i} className="text-center p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
                <feature.icon className="w-5 h-5 text-indigo-400 mx-auto mb-2" />
                <p className="text-xs text-slate-400">{feature.label}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* Data Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="stat-card blue">
          <div className="flex items-center justify-between mb-2">
            <Database className="w-4 h-4 text-blue-400" />
            <span className="badge badge-blue text-xs">Total</span>
          </div>
          <div className="text-2xl font-bold text-slate-100">
            {dataProfile.n_rows.toLocaleString()}
          </div>
          <div className="text-xs text-slate-400">Rows</div>
        </div>

        <div className="stat-card purple">
          <div className="flex items-center justify-between mb-2">
            <Hash className="w-4 h-4 text-purple-400" />
            <span className="badge badge-purple text-xs">Columns</span>
          </div>
          <div className="text-2xl font-bold text-slate-100">
            {dataProfile.n_columns}
          </div>
          <div className="text-xs text-slate-400">Variables</div>
        </div>

        <div className="stat-card green">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <span className="badge badge-green text-xs">Numeric</span>
          </div>
          <div className="text-2xl font-bold text-slate-100">
            {dataProfile.n_continuous}
          </div>
          <div className="text-xs text-slate-400">Continuous</div>
        </div>

        <div className="stat-card orange">
          <div className="flex items-center justify-between mb-2">
            <ToggleLeft className="w-4 h-4 text-orange-400" />
            <span className="badge badge-yellow text-xs">Category</span>
          </div>
          <div className="text-2xl font-bold text-slate-100">
            {dataProfile.n_categorical + dataProfile.n_binary}
          </div>
          <div className="text-xs text-slate-400">Categorical</div>
        </div>
      </div>

      {/* File Info */}
      <div className="glass-card p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
              <FileSpreadsheet className="w-5 h-5 text-white" />
            </div>
            <div className="min-w-0">
              <h3 className="font-semibold text-slate-100 truncate">{dataProfile.filename}</h3>
              <p className="text-xs text-slate-400">
                {dataProfile.n_rows.toLocaleString()} rows • {dataProfile.n_columns} columns
              </p>
            </div>
          </div>
          <button
            onClick={() => {
              setDataProfile(null);
              setPreview([]);
            }}
            className="btn-ghost flex items-center gap-1 text-sm flex-shrink-0"
          >
            <RefreshCw className="w-3 h-3" />
            Change
          </button>
        </div>
      </div>

      {/* Warnings */}
      {dataProfile.warnings.length > 0 && (
        <div className="glass-card p-4 border-yellow-500/30 bg-yellow-500/5">
          <div className="flex items-center space-x-3 mb-3">
            <div className="w-8 h-8 rounded-lg bg-yellow-500/20 flex items-center justify-center">
              <AlertTriangle className="w-4 h-4 text-yellow-400" />
            </div>
            <h3 className="font-semibold text-yellow-400">Data Quality Warnings</h3>
          </div>
          <ul className="space-y-2">
            {dataProfile.warnings.map((warning, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-yellow-200/80">
                <span className="text-yellow-500 mt-1">•</span>
                {warning}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Detected Roles */}
      {(dataProfile.potential_outcomes.length > 0 || dataProfile.potential_groups.length > 0) && (
        <div className="glass-card p-4 border-cyan-500/30 bg-cyan-500/5">
          <div className="flex items-center space-x-3 mb-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
              <Info className="w-4 h-4 text-cyan-400" />
            </div>
            <h3 className="font-semibold text-cyan-400">Detected Variable Roles</h3>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {dataProfile.potential_outcomes.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Potential Outcomes</p>
                <div className="flex flex-wrap gap-2">
                  {dataProfile.potential_outcomes.map((v, i) => (
                    <span key={i} className="badge badge-green">{v}</span>
                  ))}
                </div>
              </div>
            )}
            {dataProfile.potential_groups.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Potential Groups</p>
                <div className="flex flex-wrap gap-2">
                  {dataProfile.potential_groups.map((v, i) => (
                    <span key={i} className="badge badge-purple">{v}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Variables Table */}
      <div className="glass-card overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
          <h3 className="font-semibold text-slate-100">Variables</h3>
          <span className="text-sm text-slate-400">{dataProfile.variables.length} total</span>
        </div>
        <div className="overflow-x-auto max-h-80">
          <table className="data-table">
            <thead>
              <tr>
                <th className="min-w-[150px]">Name</th>
                <th className="min-w-[100px]">Type</th>
                <th className="min-w-[80px]">Missing</th>
                <th className="min-w-[80px]">Unique</th>
              </tr>
            </thead>
            <tbody>
              {dataProfile.variables.slice(0, 20).map((variable) => (
                <tr key={variable.name}>
                  <td className="font-medium text-slate-200">{variable.name}</td>
                  <td>
                    <span
                      className={clsx(
                        'badge text-xs',
                        variable.statistical_type === 'continuous'
                          ? 'badge-blue'
                          : variable.statistical_type === 'binary'
                          ? 'badge-green'
                          : 'badge-purple'
                      )}
                    >
                      {variable.statistical_type}
                    </span>
                  </td>
                  <td>
                    {variable.missing_pct > 0 ? (
                      <span className={clsx(
                        'text-sm',
                        variable.missing_pct > 10 ? 'text-red-400' : 'text-slate-400'
                      )}>
                        {variable.missing_pct.toFixed(1)}%
                      </span>
                    ) : (
                      <CheckCircle className="w-4 h-4 text-emerald-400" />
                    )}
                  </td>
                  <td className="text-slate-400">{variable.n_unique}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {dataProfile.variables.length > 20 && (
          <div className="px-4 py-2 border-t border-slate-700/50 text-center">
            <p className="text-xs text-slate-500">
              Showing 20 of {dataProfile.variables.length} variables
            </p>
          </div>
        )}
      </div>

      {/* Data Preview */}
      {preview.length > 0 && (
        <div className="glass-card overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50">
            <h3 className="font-semibold text-slate-100">Data Preview</h3>
            <p className="text-xs text-slate-400">First 10 rows</p>
          </div>
          <div className="overflow-x-auto max-h-64">
            <table className="data-table">
              <thead>
                <tr>
                  {Object.keys(preview[0]).map((col) => (
                    <th key={col} className="whitespace-nowrap min-w-[100px]">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((val, j) => (
                      <td key={j} className="whitespace-nowrap max-w-[200px] truncate">
                        {val === null ? (
                          <span className="text-slate-600">—</span>
                        ) : (
                          String(val)
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
