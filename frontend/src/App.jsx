import React, { useState } from 'react';
import { Routes, Route, useNavigate, useParams } from 'react-router-dom';
import Uploader from './components/Uploader/Uploader';
import PipelineEditor from './components/PipelineEditor/PipelineEditor';
import Visualization from './components/Visualization/Visualization';
import Explainability from './components/Explainability/Explainability';
import Deployment from './components/Deployment/Deployment';
import DataDashboard from './components/Dashboard/DataDashboard';
import { Database, Workflow, Brain, BarChart3 } from 'lucide-react';

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainApp />} />
      <Route path="/dashboard" element={<DashboardPage />} />
      <Route path="/dashboard/:taskId" element={<DashboardPage />} />
      <Route path="/visualizations/:modelId" element={<VisualizationPage />} />
      <Route path="/explainability/:modelId" element={<ExplainabilityPage />} />
      <Route path="/deploy/:modelId" element={<DeploymentPage />} />
    </Routes>
  );
}

function MainApp() {
  const [dataset, setDataset] = useState(null);
  const [targetColumn, setTargetColumn] = useState(null);

  const handleUploadComplete = (result) => {
    setDataset(result);
    setTargetColumn(result.suggested_target);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center space-x-3">
            <Brain className="text-blue-600" size={32} />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                AI-ML Workflow Automation
              </h1>
              <p className="text-sm text-gray-600">
                Production-ready ML lifecycle automation with live console streaming
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {!dataset ? (
          <div className="max-w-2xl mx-auto">
            <Uploader onUploadComplete={handleUploadComplete} />
          </div>
        ) : (
          <div className="space-y-8">
            {/* Dataset Info */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <Database className="text-blue-600" size={24} />
                  <div>
                    <h3 className="font-semibold text-lg">{dataset.filename}</h3>
                    <p className="text-sm text-gray-600">
                      {dataset.rows.toLocaleString()} rows × {dataset.columns} columns
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setDataset(null)}
                  className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition"
                >
                  Upload New Dataset
                </button>
              </div>

              {targetColumn && (
                <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
                  <p className="text-sm">
                    <span className="font-semibold text-blue-800">
                      Suggested Target Column:
                    </span>{' '}
                    <span className="text-blue-700">{targetColumn}</span>
                  </p>
                  <input
                    type="text"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="mt-2 w-full px-3 py-2 border border-blue-300 rounded text-sm"
                    placeholder="Enter target column name"
                  />
                </div>
              )}
            </div>

            {/* Pipeline Editor */}
            <PipelineEditor
              datasetId={dataset.dataset_id}
              targetColumn={targetColumn}
              setDataset={setDataset}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-600">
            AI-ML Workflow Automation Platform v1.0.0
          </p>
        </div>
      </footer>
    </div>
  );
}

function VisualizationPage() {
  const { modelId } = useParams();
  const navigate = useNavigate();
  
  return (
    <Visualization 
      modelId={modelId} 
      onBack={() => navigate('/')} 
    />
  );
}

function ExplainabilityPage() {
  const { modelId } = useParams();
  const navigate = useNavigate();
  
  return (
    <Explainability 
      modelId={modelId} 
      onBack={() => navigate(`/visualizations/${modelId}`)} 
    />
  );
}

function DeploymentPage() {
  const { modelId } = useParams();
  const navigate = useNavigate();
  
  return (
    <Deployment 
      modelId={modelId} 
      onBack={() => navigate(`/visualizations/${modelId}`)} 
    />
  );
}

function DashboardPage() {
  const { taskId } = useParams();
  const navigate = useNavigate();
  
  return (
    <DataDashboard 
      taskId={taskId} 
      onBack={() => navigate('/')} 
    />
  );
}

export default App;
