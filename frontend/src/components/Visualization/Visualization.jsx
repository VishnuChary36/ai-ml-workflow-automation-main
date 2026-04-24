import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, BarChart3, Info, Target, Layers, Database, Clock, CheckCircle, Settings, Rocket, TrendingUp, Eye, LayoutDashboard } from 'lucide-react';
import { getVisualizations } from '../../api/client';

// Visualization descriptions for data analyst understanding
const VIZ_DESCRIPTIONS = {
  confusion_matrix: "Shows how well the model classifies each category. Diagonal values represent correct predictions.",
  feature_importance: "Highlights which features have the most impact on predictions. Higher scores mean more influence.",
  prediction_distribution: "Compares actual vs predicted class distributions to identify any bias in predictions.",
  actual_vs_predicted: "Scatter plot comparing true values against predictions. Points closer to the red line are more accurate.",
  residual_plot: "Analyzes prediction errors. A good model shows residuals randomly scattered around zero."
};

const Visualization = ({ modelId, onBack }) => {
  const navigate = useNavigate();
  const [visualizations, setVisualizations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchVisualizations();
  }, [modelId]);

  const fetchVisualizations = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getVisualizations(modelId);
      
      if (data.status === 'generation_failed') {
        console.warn('Visualization generation failed:', data.error);
      }
      
      setVisualizations(data);
    } catch (err) {
      if (err.response?.status === 404) {
        if (err.response?.data?.detail === 'Model not found') {
          setError('Model not found. It may have been deleted.');
        } else {
          setError('Visualizations are being generated. Please wait a moment and try again.');
        }
      } else if (err.response?.status === 500) {
        setError('Error generating visualizations. Please try again later.');
      } else {
        setError('Failed to load visualizations. Please check your connection and try again.');
      }
      console.error('Error fetching visualizations:', err);
    } finally {
      setLoading(false);
    }
  };

  const downloadVisualization = (imgData, title) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${imgData}`;
    link.download = `${title.replace(/\s+/g, '_').toLowerCase()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="content-wrapper">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-on-surface">Model Visualizations</h1>
          </div>
          <div className="flex flex-col justify-center items-center h-64 gap-3">
            <div className="w-12 h-12 border-4 border-primary-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-sm text-on-surface-variant">Loading visualizations...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container flex items-center justify-center min-h-[60vh] bg-surface-container-low">
        <div className="max-w-md w-full p-8 bg-surface-container rounded-2xl shadow-xl border border-outline-variant text-center animate-in">
          <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6" style={{ background: 'rgba(239, 68, 68, 0.12)' }}>
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-on-surface mb-2">Visualization Error</h2>
          <p className="text-on-surface-variant mb-8">{error}</p>
          <div className="flex flex-col space-y-3">
            <button onClick={fetchVisualizations} className="w-full btn-primary justify-center shadow-md">
              Try Again
            </button>
            <button onClick={onBack} className="w-full btn-secondary justify-center">
              Back to Training
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!visualizations || !visualizations.visualizations) {
    return (
      <div className="page-container">
        <div className="content-wrapper">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-on-surface">Model Visualizations</h1>
            <button
              onClick={onBack}
              className="btn-secondary"
            >
              <ArrowLeft size={18} />
              Back to Training
            </button>
          </div>
          <div className="rounded-xl p-6 text-center" style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
            <p className="text-warning-400">No visualizations available for this model.</p>
          </div>
        </div>
      </div>
    );
  }

  const vizData = visualizations.visualizations;
  const metrics = visualizations.metrics;
  const modelType = visualizations.model_type;
  const modelName = visualizations.model_name || modelType;
  const targetColumn = visualizations.target_column;
  const datasetInfo = visualizations.dataset_info || {};

  // Determine which visualizations to show based on model type
  const isClassification = modelType === 'classification';
  
  // Create visualization cards for rendering
  const vizCards = [];
  
  if (vizData.confusion_matrix) {
    vizCards.push({ key: 'confusion_matrix', title: 'Confusion Matrix', data: vizData.confusion_matrix });
  }
  if (vizData.feature_importance) {
    vizCards.push({ key: 'feature_importance', title: 'Feature Importance', data: vizData.feature_importance });
  }
  if (vizData.prediction_distribution) {
    vizCards.push({ key: 'prediction_distribution', title: 'Class Distribution', data: vizData.prediction_distribution });
  }
  if (vizData.actual_vs_predicted) {
    vizCards.push({ key: 'actual_vs_predicted', title: 'Actual vs Predicted', data: vizData.actual_vs_predicted });
  }
  if (vizData.residual_plot) {
    vizCards.push({ key: 'residual_plot', title: 'Residual Analysis', data: vizData.residual_plot });
  }

  return (
    <div className="page-container">
      <div className="content-wrapper">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-on-surface flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center shadow-lg shadow-primary-500/20">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              <div>
                <span>Model Performance Analysis</span>
                <p className="text-sm font-normal text-on-surface-variant mt-0.5">Comprehensive visualization of model metrics and predictions</p>
              </div>
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="btn-secondary"
            >
              <ArrowLeft size={18} />
              Back to Training
            </button>
          </div>
        </div>

        {/* Model Information Card */}
        <div className="section-card-elevated bg-gradient-to-r from-primary-500 via-primary-600 to-primary-700 p-6 mb-6 text-white overflow-hidden relative">
          {/* Background Pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-64 h-64 bg-surface-container rounded-full -translate-y-1/2 translate-x-1/2"></div>
            <div className="absolute bottom-0 left-0 w-48 h-48 bg-surface-container rounded-full translate-y-1/2 -translate-x-1/2"></div>
          </div>
          
          <div className="relative z-10">
            <div className="flex items-center mb-5">
              <div className="w-10 h-10 bg-surface-container/20 rounded-lg flex items-center justify-center mr-3">
                <Layers size={22} />
              </div>
              <h2 className="text-xl font-bold">Model Information</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Model Type */}
              <div className="bg-surface-container/15 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center mb-2">
                  <BarChart3 size={16} className="mr-2 opacity-80" />
                  <h3 className="text-sm font-medium opacity-90">Model Type</h3>
                </div>
                <p className="text-xl font-bold">{modelName}</p>
                <p className="text-sm opacity-75 mt-1">
                  {isClassification ? 'Classification Model' : 'Regression Model'}
                </p>
              </div>
              
              {/* Target Column */}
              <div className="bg-surface-container/15 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center mb-2">
                  <Target size={16} className="mr-2 opacity-80" />
                  <h3 className="text-sm font-medium opacity-90">Target Column</h3>
                </div>
                <p className="text-xl font-bold">{targetColumn}</p>
                <p className="text-sm opacity-75 mt-1">
                  {isClassification && datasetInfo.n_classes ? `${datasetInfo.n_classes} classes` : 'Continuous variable'}
                </p>
              </div>
              
              {/* Dataset Info */}
              <div className="bg-surface-container/15 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center mb-2">
                  <Database size={16} className="mr-2 opacity-80" />
                  <h3 className="text-sm font-medium opacity-90">Training Data</h3>
                </div>
                <p className="text-xl font-bold">
                  {datasetInfo.n_samples ? datasetInfo.n_samples.toLocaleString() : 'N/A'} samples
                </p>
                <p className="text-sm opacity-75 mt-1">
                  {datasetInfo.n_features ? `${datasetInfo.n_features} features` : ''}
                </p>
              </div>
              
              {/* Generated At */}
              <div className="bg-surface-container/15 backdrop-blur-sm rounded-xl p-4 border border-white/20">
                <div className="flex items-center mb-2">
                  <Clock size={16} className="mr-2 opacity-80" />
                  <h3 className="text-sm font-medium opacity-90">Generated</h3>
                </div>
                <p className="text-lg font-bold">
                  {visualizations.generated_at ? new Date(visualizations.generated_at).toLocaleDateString() : 'Recently'}
                </p>
                <p className="text-sm opacity-75 mt-1">
                  {visualizations.generated_at ? new Date(visualizations.generated_at).toLocaleTimeString() : ''}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Model Performance Metrics */}
        {metrics && Object.keys(metrics).length > 0 && (
          <div className="section-card p-6 mb-6">
            <div className="flex items-center mb-5">
              <div className="w-10 h-10 bg-gradient-to-br from-success-100 to-success-50 rounded-lg flex items-center justify-center mr-3">
                <CheckCircle size={20} className="text-success-500" />
              </div>
              <h2 className="text-xl font-bold text-on-surface">Performance Metrics</h2>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(metrics).map(([key, value]) => {
                // Format metric name nicely
                const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                // Determine if it's a percentage metric
                const isPercentage = key.includes('accuracy') || key.includes('f1') || key.includes('precision') || key.includes('recall');
                const displayValue = typeof value === 'number' 
                  ? (isPercentage && value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(4))
                  : value;
                
                return (
                  <div key={key} className="border border-outline-variant rounded-xl p-4 text-center bg-surface-container">
                    <h3 className="text-sm font-medium text-on-surface-variant mb-1">{formattedKey}</h3>
                    <p className="text-2xl font-bold text-primary-500">{displayValue}</p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Visualizations */}
        <div className="space-y-6">
          <div className="flex items-center mb-2">
            <div className="w-8 h-8 bg-gradient-to-br from-accent-indigo/20 to-accent-indigo/10 rounded-lg flex items-center justify-center mr-3">
              <Info size={16} className="text-accent-indigo" />
            </div>
            <h2 className="text-lg font-semibold text-on-surface-variant">
              Key Visualizations <span className="text-primary-500">({vizCards.length} charts)</span>
            </h2>
          </div>
          
          {vizCards.map(({ key, title, data }) => (
            <div key={key} className="viz-panel">
              <div className="viz-panel-header flex justify-between items-center">
                <div>
                  <h2 className="text-lg font-semibold text-on-surface">{title}</h2>
                  {VIZ_DESCRIPTIONS[key] && (
                    <p className="text-sm text-on-surface-variant mt-1">{VIZ_DESCRIPTIONS[key]}</p>
                  )}
                </div>
                <button
                  onClick={() => downloadVisualization(data, title)}
                  className="btn-primary text-sm"
                >
                  <Download size={16} />
                  Download PNG
                </button>
              </div>
              <div className="p-6 flex justify-center bg-surface-container-low rounded-b-xl">
                <img 
                  src={`data:image/png;base64,${data}`} 
                  alt={title} 
                  className="max-w-full h-auto rounded-xl shadow-lg"
                />
              </div>
            </div>
          ))}
        </div>
        
        {vizCards.length === 0 && (
          <div className="rounded-xl p-6 text-center" style={{ background: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
            <p className="text-warning-400">No visualizations were generated for this model.</p>
          </div>
        )}

        {/* View Grafana Dashboard Button */}
        <div className="mt-8 section-card-elevated bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 p-6 overflow-hidden relative">
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-48 h-48 bg-surface-container rounded-full -translate-y-1/2 translate-x-1/2"></div>
          </div>
          <div className="relative z-10 flex items-center justify-between">
            <div className="text-white">
              <div className="flex items-center mb-2">
                <div className="w-10 h-10 bg-surface-container/20 rounded-lg flex items-center justify-center mr-3">
                  <LayoutDashboard size={20} />
                </div>
                <h2 className="text-xl font-bold">Grafana Dashboard</h2>
              </div>
              <p className="opacity-90 ml-13">
                Interactive analytics dashboard with AI-generated insights and live Grafana panels
              </p>
            </div>
            <button
              onClick={() => navigate(`/grafana/${modelId}`)}
              className="flex items-center gap-2 px-6 py-3 bg-surface-container text-warning-500 font-semibold rounded-xl hover:bg-surface-container-high transition shadow-lg"
            >
              <LayoutDashboard size={18} />
              View Dashboard
            </button>
          </div>
        </div>

        {/* Go to Explainability Button */}
        <div className="mt-4 section-card-elevated bg-gradient-to-r from-slate-700 via-slate-800 to-slate-900 p-6 overflow-hidden relative">
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-48 h-48 bg-surface-container rounded-full -translate-y-1/2 translate-x-1/2"></div>
          </div>
          <div className="relative z-10 flex items-center justify-between">
            <div className="text-white">
              <div className="flex items-center mb-2">
                <div className="w-10 h-10 bg-surface-container/20 rounded-lg flex items-center justify-center mr-3">
                  <Eye size={20} />
                </div>
                <h2 className="text-xl font-bold">Model Explainability</h2>
              </div>
              <p className="opacity-90 ml-13">
                Dive deeper with SHAP, LIME, Partial Dependence Plots, and more advanced analysis techniques
              </p>
            </div>
            <button
              onClick={() => navigate(`/explainability/${modelId}`)}
              className="flex items-center gap-2 px-6 py-3 bg-surface-container text-on-surface-variant font-semibold rounded-xl hover:bg-surface-container-low transition shadow-lg"
            >
              <Eye size={18} />
              View Explainability
            </button>
          </div>
        </div>

        {/* Deploy Model Button */}
        <div className="mt-4 section-card-elevated bg-gradient-to-r from-success-500 via-success-600 to-success-700 p-6 overflow-hidden relative">
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-48 h-48 bg-surface-container rounded-full -translate-y-1/2 translate-x-1/2"></div>
          </div>
          <div className="relative z-10 flex items-center justify-between">
            <div className="text-white">
              <div className="flex items-center mb-2">
                <div className="w-10 h-10 bg-surface-container/20 rounded-lg flex items-center justify-center mr-3">
                  <Rocket size={20} />
                </div>
                <h2 className="text-xl font-bold">Deploy Model</h2>
              </div>
              <p className="opacity-90 ml-13">
                Package and deploy your model with inference API, Docker container, and monitoring
              </p>
            </div>
            <button
              onClick={() => navigate(`/deploy/${modelId}`)}
              className="flex items-center gap-2 px-6 py-3 bg-surface-container text-success-500 font-semibold rounded-xl hover:bg-surface-container-high transition shadow-md"
            >
              <Rocket size={18} />
              Deploy Model
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Visualization;