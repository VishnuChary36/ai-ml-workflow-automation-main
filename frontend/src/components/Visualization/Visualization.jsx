import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, BarChart3, Info, Target, Layers, Database, Clock, CheckCircle, Sparkles, Rocket } from 'lucide-react';
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
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-gray-800">Model Visualizations</h1>
          </div>
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-gray-800">Model Visualizations</h1>
            <button
              onClick={onBack}
              className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition"
            >
              <ArrowLeft size={18} className="mr-2" />
              Back to Training
            </button>
          </div>
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <p className="text-red-700 mb-4">{error}</p>
            <button
              onClick={fetchVisualizations}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!visualizations || !visualizations.visualizations) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-gray-800">Model Visualizations</h1>
            <button
              onClick={onBack}
              className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition"
            >
              <ArrowLeft size={18} className="mr-2" />
              Back to Training
            </button>
          </div>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
            <p className="text-yellow-700">No visualizations available for this model.</p>
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
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">Model Analysis</h1>
            <p className="text-gray-600">Key insights and visualizations for your trained model</p>
          </div>
          <button
            onClick={onBack}
            className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition"
          >
            <ArrowLeft size={18} className="mr-2" />
            Back to Training
          </button>
        </div>

        {/* Model Information Card - Enhanced */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl shadow-lg p-6 mb-6 text-white">
          <div className="flex items-center mb-4">
            <Layers size={24} className="mr-2" />
            <h2 className="text-xl font-bold">Model Information</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Model Type */}
            <div className="bg-white/20 backdrop-blur rounded-lg p-4">
              <div className="flex items-center mb-2">
                <BarChart3 size={18} className="mr-2 opacity-80" />
                <h3 className="font-medium opacity-90">Model Type</h3>
              </div>
              <p className="text-xl font-bold">{modelName}</p>
              <p className="text-sm opacity-75 mt-1">
                {isClassification ? 'Classification Model' : 'Regression Model'}
              </p>
            </div>
            
            {/* Target Column */}
            <div className="bg-white/20 backdrop-blur rounded-lg p-4">
              <div className="flex items-center mb-2">
                <Target size={18} className="mr-2 opacity-80" />
                <h3 className="font-medium opacity-90">Target Column</h3>
              </div>
              <p className="text-xl font-bold">{targetColumn}</p>
              <p className="text-sm opacity-75 mt-1">
                {isClassification && datasetInfo.n_classes ? `${datasetInfo.n_classes} classes` : 'Continuous variable'}
              </p>
            </div>
            
            {/* Dataset Info */}
            <div className="bg-white/20 backdrop-blur rounded-lg p-4">
              <div className="flex items-center mb-2">
                <Database size={18} className="mr-2 opacity-80" />
                <h3 className="font-medium opacity-90">Training Data</h3>
              </div>
              <p className="text-xl font-bold">
                {datasetInfo.n_samples ? datasetInfo.n_samples.toLocaleString() : 'N/A'} samples
              </p>
              <p className="text-sm opacity-75 mt-1">
                {datasetInfo.n_features ? `${datasetInfo.n_features} features` : ''}
              </p>
            </div>
            
            {/* Generated At */}
            <div className="bg-white/20 backdrop-blur rounded-lg p-4">
              <div className="flex items-center mb-2">
                <Clock size={18} className="mr-2 opacity-80" />
                <h3 className="font-medium opacity-90">Generated</h3>
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

        {/* Model Performance Metrics */}
        {metrics && Object.keys(metrics).length > 0 && (
          <div className="bg-white rounded-xl shadow-md p-6 mb-6">
            <div className="flex items-center mb-4">
              <CheckCircle size={22} className="mr-2 text-green-600" />
              <h2 className="text-xl font-bold text-gray-800">Model Performance</h2>
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
                  <div key={key} className="border border-gray-200 rounded-lg p-4 text-center">
                    <h3 className="text-sm font-medium text-gray-500 mb-1">{formattedKey}</h3>
                    <p className="text-2xl font-bold text-blue-600">{displayValue}</p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Visualizations */}
        <div className="space-y-6">
          <div className="flex items-center">
            <Info size={20} className="mr-2 text-gray-500" />
            <h2 className="text-lg font-semibold text-gray-700">
              Key Visualizations ({vizCards.length} charts)
            </h2>
          </div>
          
          {vizCards.map(({ key, title, data }) => (
            <div key={key} className="bg-white rounded-xl shadow-md overflow-hidden">
              <div className="flex justify-between items-center p-4 border-b border-gray-100">
                <div>
                  <h2 className="text-xl font-semibold text-gray-800">{title}</h2>
                  {VIZ_DESCRIPTIONS[key] && (
                    <p className="text-sm text-gray-500 mt-1">{VIZ_DESCRIPTIONS[key]}</p>
                  )}
                </div>
                <button
                  onClick={() => downloadVisualization(data, title)}
                  className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition shadow-sm"
                >
                  <Download size={16} className="mr-2" />
                  Download PNG
                </button>
              </div>
              <div className="p-6 flex justify-center bg-gray-50">
                <img 
                  src={`data:image/png;base64,${data}`} 
                  alt={title} 
                  className="max-w-full h-auto rounded-lg shadow-sm"
                />
              </div>
            </div>
          ))}
        </div>
        
        {vizCards.length === 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
            <p className="text-yellow-700">No visualizations were generated for this model.</p>
          </div>
        )}

        {/* Go to Explainability Button */}
        <div className="mt-8 bg-gradient-to-r from-purple-600 to-indigo-700 rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div className="text-white">
              <div className="flex items-center mb-2">
                <Sparkles size={24} className="mr-2" />
                <h2 className="text-xl font-bold">Model Explainability</h2>
              </div>
              <p className="opacity-90">
                Dive deeper with SHAP, LIME, Partial Dependence Plots, and more advanced analysis techniques
              </p>
            </div>
            <button
              onClick={() => navigate(`/explainability/${modelId}`)}
              className="flex items-center px-6 py-3 bg-white text-purple-700 font-semibold rounded-lg hover:bg-purple-50 transition shadow-md"
            >
              <Sparkles size={18} className="mr-2" />
              View Explainability
            </button>
          </div>
        </div>

        {/* Deploy Model Button */}
        <div className="mt-4 bg-gradient-to-r from-green-600 to-emerald-700 rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div className="text-white">
              <div className="flex items-center mb-2">
                <Rocket size={24} className="mr-2" />
                <h2 className="text-xl font-bold">Deploy Model</h2>
              </div>
              <p className="opacity-90">
                Package and deploy your model with inference API, Docker container, and monitoring
              </p>
            </div>
            <button
              onClick={() => navigate(`/deploy/${modelId}`)}
              className="flex items-center px-6 py-3 bg-white text-green-700 font-semibold rounded-lg hover:bg-green-50 transition shadow-md"
            >
              <Rocket size={18} className="mr-2" />
              Deploy Model
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Visualization;