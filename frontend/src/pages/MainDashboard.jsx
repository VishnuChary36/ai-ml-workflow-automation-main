import React from 'react';
import { Database, Workflow, TrendingUp, Layers, Sparkles, ChevronDown, Target } from 'lucide-react';
import Uploader from '../components/Uploader/Uploader';
import PipelineEditor from '../components/PipelineEditor/PipelineEditor';
import PageHeader from '../components/UI/PageHeader';
import { useSessionStorage } from '../hooks/useSessionStorage';

export default function MainDashboard() {
  const [dataset, setDataset] = useSessionStorage('ml_dataset', null);
  const [targetColumn, setTargetColumn] = useSessionStorage('ml_targetColumn', null);

  const handleUploadComplete = (result) => {
    setDataset(result);
    setTargetColumn(result.suggested_target);
  };

  const handleClearDataset = () => {
    setDataset(null);
    setTargetColumn(null);
    sessionStorage.removeItem('ml_pipelineState');
    sessionStorage.removeItem('ml_trainingState');
  };

  return (
    <div className="page-container">
      <main className="content-wrapper">
        {!dataset ? (
          <div className="max-w-2xl mx-auto animate-in">
            <div className="text-center mb-10">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-primary-400 to-primary-500 rounded-2xl shadow-glow-blue mb-6">
                <Database className="text-white" size={32} />
              </div>
              <h2 className="text-3xl font-bold text-on-surface mb-3">Data Analytics Platform</h2>
              <p className="text-on-surface-variant text-lg max-w-lg mx-auto">
                Upload your dataset to begin automated data processing, model training, and performance analysis
              </p>
            </div>
            <Uploader onUploadComplete={handleUploadComplete} />
            
            <div className="mt-14 grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="feature-card text-center">
                <div className="w-14 h-14 rounded-xl flex items-center justify-center mx-auto mb-4" style={{ background: 'rgba(77, 142, 255, 0.1)' }}>
                  <Workflow className="text-primary-500" size={26} />
                </div>
                <h3 className="font-semibold text-on-surface mb-2">Data Processing</h3>
                <p className="text-sm text-on-surface-variant">Automated data cleaning, preprocessing, and feature engineering pipeline</p>
              </div>
              <div className="feature-card text-center">
                <div className="w-14 h-14 rounded-xl flex items-center justify-center mx-auto mb-4" style={{ background: 'rgba(16, 185, 129, 0.1)' }}>
                  <TrendingUp className="text-success-500" size={26} />
                </div>
                <h3 className="font-semibold text-on-surface mb-2">Model Training</h3>
                <p className="text-sm text-on-surface-variant">Train and evaluate multiple machine learning models with detailed metrics</p>
              </div>
              <div className="feature-card text-center">
                <div className="w-14 h-14 rounded-xl flex items-center justify-center mx-auto mb-4" style={{ background: 'rgba(129, 140, 248, 0.1)' }}>
                  <Layers className="text-accent-indigo" size={26} />
                </div>
                <h3 className="font-semibold text-on-surface mb-2">Model Deployment</h3>
                <p className="text-sm text-on-surface-variant">Deploy trained models to local, Docker, or cloud platforms seamlessly</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6 animate-in">
            <div className="card p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="w-14 h-14 rounded-xl flex items-center justify-center" style={{ background: 'rgba(77, 142, 255, 0.12)' }}>
                    <Database className="text-primary-500" size={28} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg text-on-surface">{dataset.filename}</h3>
                    <div className="flex items-center space-x-4 mt-1">
                      <span className="badge badge-primary">{dataset.rows.toLocaleString()} rows</span>
                      <span className="badge badge-slate">{dataset.columns} columns</span>
                    </div>
                  </div>
                </div>
                <button onClick={handleClearDataset} className="btn-secondary text-sm">Upload New Dataset</button>
              </div>

              {dataset.column_names && dataset.column_names.length > 0 && (
                <div className="mt-6 p-4 rounded-xl" style={{ background: 'rgba(77, 142, 255, 0.06)', border: '1px solid rgba(77, 142, 255, 0.15)' }}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <Target size={16} className="text-primary-500" />
                      <label htmlFor="target-column" className="text-sm font-semibold text-on-surface">Target Column</label>
                    </div>
                    {dataset.target_suggestion && (
                      <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg" style={{ background: 'rgba(77, 142, 255, 0.12)', border: '1px solid rgba(77, 142, 255, 0.2)' }}>
                        <Sparkles size={13} className="text-primary-500" />
                        <span className="text-xs font-semibold text-primary-800">AI Suggested</span>
                        {dataset.target_suggestion.confidence && (
                          <span className="text-xs text-on-surface-variant opacity-75">
                            ({Math.round(dataset.target_suggestion.confidence * 100)}% confidence)
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="relative">
                    <select
                      id="target-column"
                      value={targetColumn || ''}
                      onChange={(e) => setTargetColumn(e.target.value)}
                      className="input w-full appearance-none pr-10 font-medium"
                    >
                      {dataset.column_names.map((col) => (
                        <option key={col} value={col}>
                          {col}{col === dataset.suggested_target ? ' (AI Recommended)' : ''}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none" />
                  </div>
                  {dataset.target_suggestion && dataset.target_suggestion.reason && (
                    <div className="mt-3 flex items-start space-x-2 p-2.5 rounded-lg" style={{ background: 'rgba(77, 142, 255, 0.05)', border: '1px solid rgba(77, 142, 255, 0.1)' }}>
                      <Sparkles size={14} className="text-primary-500 mt-0.5 flex-shrink-0" />
                      <p className="text-xs text-on-surface-variant">
                        <span className="font-semibold text-primary-800">Why this column: </span>
                        {dataset.target_suggestion.reason}
                      </p>
                    </div>
                  )}
                  <p className="text-xs text-on-surface-variant mt-2 opacity-70">
                    The column your model will predict. Change it using the dropdown if needed.
                  </p>
                </div>
              )}
            </div>

            <PipelineEditor datasetId={dataset.dataset_id} targetColumn={targetColumn} setDataset={setDataset} />
          </div>
        )}
      </main>
    </div>
  );
}
