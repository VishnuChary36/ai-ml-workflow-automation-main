import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, Sparkles, Info, AlertTriangle, CheckCircle, HelpCircle, Zap, GitBranch, BarChart3, Target, Layers, Rocket } from 'lucide-react';
import { getExplainability } from '../../api/client';

// Detailed descriptions for each explainability technique
const EXPLAINABILITY_DESCRIPTIONS = {
  permutation_importance: {
    title: "Permutation Feature Importance",
    description: "Measures how much each feature contributes to model accuracy by randomly shuffling feature values. Higher values indicate more important features.",
    interpretation: "If shuffling a feature causes a big drop in accuracy, that feature is important for predictions."
  },
  shap_summary: {
    title: "SHAP Summary Plot",
    description: "Shows the impact of each feature on model output using SHAP (SHapley Additive exPlanations) values. Each dot represents a sample.",
    interpretation: "Red dots indicate high feature values, blue indicate low. Dots further from center have greater impact on predictions."
  },
  shap_importance: {
    title: "SHAP Feature Importance",
    description: "Aggregated importance scores from SHAP analysis, showing mean absolute SHAP values for each feature.",
    interpretation: "Higher bars indicate features that have more influence on the model's predictions overall."
  },
  shap_dependence: {
    title: "SHAP Dependence Plots",
    description: "Shows how the top feature's value affects the model's output, with color indicating interaction with the most correlated feature.",
    interpretation: "The vertical spread at each x-value shows the interaction effect with other features."
  },
  pdp: {
    title: "Partial Dependence Plots",
    description: "Shows the marginal effect of a feature on predicted outcome, averaging out all other features.",
    interpretation: "Rising/falling lines indicate positive/negative relationships between the feature and prediction."
  },
  lime: {
    title: "LIME Local Explanations",
    description: "Explains individual predictions by approximating the model locally with an interpretable linear model.",
    interpretation: "Green bars push toward one class, red bars push toward the other. Length indicates importance."
  },
  surrogate_tree: {
    title: "Surrogate Decision Tree",
    description: "A simplified interpretable tree model that approximates the complex model's decisions.",
    interpretation: "Follow the branches to understand approximate decision rules. Fidelity score shows how well it mimics the original model."
  },
  confusion_analysis: {
    title: "Detailed Confusion Analysis",
    description: "Comprehensive breakdown of model predictions with per-class metrics including precision, recall, and F1 scores.",
    interpretation: "Diagonal values should be high (correct predictions). Off-diagonal shows misclassification patterns."
  },
  calibration_plot: {
    title: "Probability Calibration Plot",
    description: "Shows how well the predicted probabilities match actual outcomes. A well-calibrated model follows the diagonal.",
    interpretation: "If the line is above diagonal, the model is under-confident. Below diagonal means over-confident."
  },
  feature_distributions: {
    title: "Feature Distributions",
    description: "Histograms showing the distribution of top features in your training data.",
    interpretation: "Helps identify skewed distributions, outliers, or unusual patterns in your data."
  },
  correlation_heatmap: {
    title: "Feature Correlation Heatmap",
    description: "Shows correlations between top features. High correlations may indicate redundant features.",
    interpretation: "Dark red/blue squares indicate strong positive/negative correlations. Consider removing highly correlated features."
  }
};

const Explainability = ({ modelId, onBack }) => {
  const navigate = useNavigate();
  const [explainability, setExplainability] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedSection, setExpandedSection] = useState(null);

  useEffect(() => {
    fetchExplainability();
  }, [modelId]);

  const fetchExplainability = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getExplainability(modelId);
      setExplainability(data);
    } catch (err) {
      if (err.response?.status === 404) {
        setError('Model not found. It may have been deleted.');
      } else if (err.response?.status === 500) {
        setError('Error generating explainability analysis. Please try again later.');
      } else {
        setError('Failed to load explainability analysis. Please check your connection and try again.');
      }
      console.error('Error fetching explainability:', err);
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

  const toggleSection = (key) => {
    setExpandedSection(expandedSection === key ? null : key);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Sparkles className="text-purple-600 mr-3" size={28} />
              <h1 className="text-2xl font-bold text-gray-800">Model Explainability</h1>
            </div>
          </div>
          <div className="flex flex-col justify-center items-center h-64 space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
            <p className="text-gray-600">Generating explainability analysis...</p>
            <p className="text-sm text-gray-500">This may take a minute for SHAP and LIME calculations</p>
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
            <div className="flex items-center">
              <Sparkles className="text-purple-600 mr-3" size={28} />
              <h1 className="text-2xl font-bold text-gray-800">Model Explainability</h1>
            </div>
            <button
              onClick={onBack}
              className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition"
            >
              <ArrowLeft size={18} className="mr-2" />
              Back to Visualizations
            </button>
          </div>
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <AlertTriangle className="mx-auto text-red-500 mb-4" size={48} />
            <p className="text-red-700 mb-4">{error}</p>
            <button
              onClick={fetchExplainability}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!explainability) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Sparkles className="text-purple-600 mr-3" size={28} />
              <h1 className="text-2xl font-bold text-gray-800">Model Explainability</h1>
            </div>
            <button
              onClick={onBack}
              className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition"
            >
              <ArrowLeft size={18} className="mr-2" />
              Back to Visualizations
            </button>
          </div>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
            <p className="text-yellow-700">No explainability data available for this model.</p>
          </div>
        </div>
      </div>
    );
  }

  const modelName = explainability.model_name || 'Unknown Model';
  const modelType = explainability.model_type || 'classification';
  const targetColumn = explainability.target_column || 'Unknown';
  const explanations = explainability.explanations || {};
  const isClassification = modelType === 'classification';

  // Build sections from available explanations
  const sections = [];

  // Group explanations by category
  const globalExplanations = [];
  const localExplanations = [];
  const dataAnalysis = [];
  const performanceAnalysis = [];

  // Permutation Importance
  if (explanations.permutation_importance) {
    globalExplanations.push({
      key: 'permutation_importance',
      data: explanations.permutation_importance
    });
  }

  // SHAP
  if (explanations.shap) {
    if (explanations.shap.summary_plot) {
      globalExplanations.push({
        key: 'shap_summary',
        data: { plot: explanations.shap.summary_plot }
      });
    }
    if (explanations.shap.importance_plot) {
      globalExplanations.push({
        key: 'shap_importance',
        data: { plot: explanations.shap.importance_plot }
      });
    }
    if (explanations.shap.dependence_plots && explanations.shap.dependence_plots.length > 0) {
      explanations.shap.dependence_plots.forEach((plot, idx) => {
        globalExplanations.push({
          key: `shap_dependence_${idx}`,
          customTitle: `SHAP Dependence: ${plot.feature}`,
          data: { plot: plot.plot },
          description: EXPLAINABILITY_DESCRIPTIONS.shap_dependence
        });
      });
    }
  }

  // PDP
  if (explanations.pdp && explanations.pdp.plots && explanations.pdp.plots.length > 0) {
    explanations.pdp.plots.forEach((plot, idx) => {
      globalExplanations.push({
        key: `pdp_${idx}`,
        customTitle: `Partial Dependence: ${plot.feature}`,
        data: { plot: plot.plot },
        description: EXPLAINABILITY_DESCRIPTIONS.pdp
      });
    });
  }

  // Surrogate Tree
  if (explanations.surrogate_tree) {
    globalExplanations.push({
      key: 'surrogate_tree',
      data: explanations.surrogate_tree
    });
  }

  // LIME
  if (explanations.lime && explanations.lime.explanations && explanations.lime.explanations.length > 0) {
    explanations.lime.explanations.forEach((exp, idx) => {
      localExplanations.push({
        key: `lime_${idx}`,
        customTitle: `LIME Explanation: Sample ${exp.sample_index}`,
        data: { 
          plot: exp.plot,
          prediction: exp.prediction,
          actual: exp.actual
        },
        description: EXPLAINABILITY_DESCRIPTIONS.lime
      });
    });
  }

  // Performance Analysis
  if (explanations.confusion_analysis) {
    performanceAnalysis.push({
      key: 'confusion_analysis',
      data: explanations.confusion_analysis
    });
  }

  if (explanations.calibration_plot) {
    performanceAnalysis.push({
      key: 'calibration_plot',
      data: explanations.calibration_plot
    });
  }

  // Data Analysis
  if (explanations.feature_distributions) {
    dataAnalysis.push({
      key: 'feature_distributions',
      data: explanations.feature_distributions
    });
  }

  if (explanations.correlation_heatmap) {
    dataAnalysis.push({
      key: 'correlation_heatmap',
      data: explanations.correlation_heatmap
    });
  }

  const renderExplanationCard = (item) => {
    const key = item.key;
    const baseKey = key.replace(/_\d+$/, ''); // Remove index suffix
    const desc = item.description || EXPLAINABILITY_DESCRIPTIONS[baseKey] || {};
    const title = item.customTitle || desc.title || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    const data = item.data;
    
    // Get the plot image
    const plotImage = data.plot || data.importance_plot || null;
    
    return (
      <div key={key} className="bg-white rounded-xl shadow-md overflow-hidden">
        <div 
          className="flex justify-between items-center p-4 border-b border-gray-100 cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection(key)}
        >
          <div className="flex-1">
            <div className="flex items-center">
              <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
              <HelpCircle 
                size={16} 
                className="ml-2 text-gray-400 hover:text-purple-600" 
                title={desc.interpretation || "Click for more info"}
              />
            </div>
            {desc.description && (
              <p className="text-sm text-gray-500 mt-1">{desc.description}</p>
            )}
          </div>
          {plotImage && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                downloadVisualization(plotImage, title);
              }}
              className="flex items-center px-3 py-2 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 transition shadow-sm ml-4"
            >
              <Download size={14} className="mr-1" />
              Download
            </button>
          )}
        </div>
        
        {/* Interpretation box - always visible */}
        {desc.interpretation && (
          <div className="px-4 py-3 bg-purple-50 border-b border-purple-100">
            <div className="flex items-start">
              <Info size={16} className="text-purple-600 mr-2 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-purple-800">
                <strong>How to interpret:</strong> {desc.interpretation}
              </p>
            </div>
          </div>
        )}
        
        {/* Additional info for LIME */}
        {data.prediction !== undefined && (
          <div className="px-4 py-2 bg-blue-50 border-b border-blue-100 flex space-x-6">
            <span className="text-sm">
              <strong className="text-blue-700">Predicted:</strong> {data.prediction}
            </span>
            {data.actual !== undefined && (
              <span className="text-sm">
                <strong className="text-blue-700">Actual:</strong> {data.actual}
              </span>
            )}
          </div>
        )}
        
        {/* Fidelity for surrogate tree */}
        {data.fidelity !== undefined && (
          <div className="px-4 py-2 bg-green-50 border-b border-green-100">
            <span className="text-sm">
              <strong className="text-green-700">Fidelity Score:</strong> {(data.fidelity * 100).toFixed(1)}% 
              <span className="text-gray-500 ml-2">(how well this tree mimics the original model)</span>
            </span>
          </div>
        )}
        
        {/* High correlations warning */}
        {data.high_correlations && data.high_correlations.length > 0 && (
          <div className="px-4 py-2 bg-yellow-50 border-b border-yellow-100">
            <div className="flex items-start">
              <AlertTriangle size={16} className="text-yellow-600 mr-2 mt-0.5" />
              <div className="text-sm text-yellow-800">
                <strong>High Correlations Detected:</strong>
                <ul className="list-disc ml-4 mt-1">
                  {data.high_correlations.slice(0, 3).map((corr, idx) => (
                    <li key={idx}>{corr.feature1} ↔ {corr.feature2}: {corr.correlation.toFixed(2)}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
        
        {/* Per-class metrics for confusion analysis */}
        {data.per_class_metrics && (
          <div className="px-4 py-3 bg-gray-50 border-b border-gray-100 overflow-x-auto">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-semibold text-gray-700">Per-Class Metrics (Top Classes):</h4>
              {data.total_classes && data.shown_classes && data.total_classes > data.shown_classes && (
                <span className="text-xs text-gray-500">
                  Showing {data.shown_classes} of {data.total_classes} classes
                </span>
              )}
            </div>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-gray-600 bg-gray-100">
                  <th className="px-3 py-2 rounded-tl">Class</th>
                  <th className="px-3 py-2">Precision</th>
                  <th className="px-3 py-2">Recall</th>
                  <th className="px-3 py-2">F1 Score</th>
                  <th className="px-3 py-2 rounded-tr">Support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.per_class_metrics).map(([cls, metrics], idx) => {
                  // Format metric value, handle NaN/undefined
                  const formatMetric = (val) => {
                    if (val === undefined || val === null || isNaN(val)) return 'N/A';
                    return `${(val * 100).toFixed(1)}%`;
                  };
                  
                  return (
                    <tr key={cls} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-3 py-2 font-medium text-gray-800">{cls}</td>
                      <td className="px-3 py-2">{formatMetric(metrics.precision)}</td>
                      <td className="px-3 py-2">{formatMetric(metrics.recall)}</td>
                      <td className="px-3 py-2">{formatMetric(metrics.f1)}</td>
                      <td className="px-3 py-2 text-gray-600">{metrics.support}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        
        {/* Plot image */}
        {plotImage && (
          <div className="p-6 flex justify-center bg-gray-50">
            <img 
              src={`data:image/png;base64,${plotImage}`} 
              alt={title} 
              className="max-w-full h-auto rounded-lg shadow-sm"
            />
          </div>
        )}
      </div>
    );
  };

  const renderSection = (title, icon, items, description) => {
    if (!items || items.length === 0) return null;
    
    return (
      <div className="mb-8">
        <div className="flex items-center mb-4">
          {icon}
          <h2 className="text-xl font-bold text-gray-800 ml-2">{title}</h2>
          <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
            {items.length} {items.length === 1 ? 'analysis' : 'analyses'}
          </span>
        </div>
        {description && (
          <p className="text-gray-600 mb-4 ml-8">{description}</p>
        )}
        <div className="space-y-4">
          {items.map(item => renderExplanationCard(item))}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center">
              <Sparkles className="text-purple-600 mr-3" size={28} />
              <h1 className="text-2xl font-bold text-gray-800">Model Explainability</h1>
            </div>
            <p className="text-gray-600 ml-10">Deep-dive analysis into how your model makes predictions</p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => navigate(`/deploy/${modelId}`)}
              className="flex items-center px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition shadow-md"
            >
              <Rocket size={18} className="mr-2" />
              Deploy Model
            </button>
            <button
              onClick={onBack}
              className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition"
            >
              <ArrowLeft size={18} className="mr-2" />
              Back to Visualizations
            </button>
          </div>
        </div>

        {/* Model Info Banner */}
        <div className="bg-gradient-to-r from-purple-600 to-indigo-700 rounded-xl shadow-lg p-6 mb-6 text-white">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center">
              <Layers size={20} className="mr-2 opacity-80" />
              <div>
                <p className="text-sm opacity-80">Model</p>
                <p className="font-bold text-lg">{modelName}</p>
              </div>
            </div>
            <div className="flex items-center">
              <Target size={20} className="mr-2 opacity-80" />
              <div>
                <p className="text-sm opacity-80">Target</p>
                <p className="font-bold text-lg">{targetColumn}</p>
              </div>
            </div>
            <div className="flex items-center">
              <BarChart3 size={20} className="mr-2 opacity-80" />
              <div>
                <p className="text-sm opacity-80">Type</p>
                <p className="font-bold text-lg">{isClassification ? 'Classification' : 'Regression'}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Guide Card */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-8 border-l-4 border-purple-500">
          <div className="flex items-start">
            <Info size={24} className="text-purple-600 mr-3 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-bold text-gray-800 mb-2">Understanding This Page</h3>
              <p className="text-gray-600 text-sm">
                This page provides multiple perspectives on how your model makes decisions. 
                <strong> Global explanations</strong> (SHAP, Permutation Importance) show overall feature importance.
                <strong> Local explanations</strong> (LIME) show why specific predictions were made.
                <strong> Data analysis</strong> helps you understand your training data.
                Each chart includes an interpretation guide to help you understand the insights.
              </p>
            </div>
          </div>
        </div>

        {/* Global Explanations */}
        {renderSection(
          "Global Feature Importance",
          <Zap className="text-yellow-500" size={24} />,
          globalExplanations,
          "These explanations show which features are most important across all predictions."
        )}

        {/* Local Explanations */}
        {renderSection(
          "Local Explanations (Individual Predictions)",
          <Target className="text-green-500" size={24} />,
          localExplanations,
          "These explain specific predictions to help you understand individual model decisions."
        )}

        {/* Performance Analysis */}
        {renderSection(
          "Performance Analysis",
          <CheckCircle className="text-blue-500" size={24} />,
          performanceAnalysis,
          "Detailed metrics and calibration analysis for model performance."
        )}

        {/* Data Analysis */}
        {renderSection(
          "Data Analysis",
          <BarChart3 className="text-orange-500" size={24} />,
          dataAnalysis,
          "Understand the characteristics of your training data."
        )}

        {/* No explanations available */}
        {globalExplanations.length === 0 && localExplanations.length === 0 && 
         performanceAnalysis.length === 0 && dataAnalysis.length === 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
            <AlertTriangle className="mx-auto text-yellow-500 mb-4" size={48} />
            <p className="text-yellow-700">No explainability analyses could be generated for this model.</p>
            <p className="text-yellow-600 text-sm mt-2">This may be due to missing data or unsupported model type.</p>
          </div>
        )}

        {/* Deploy Model CTA */}
        <div className="bg-gradient-to-r from-green-600 to-emerald-700 rounded-xl shadow-lg p-6 mt-8">
          <div className="flex items-center justify-between">
            <div className="text-white">
              <h2 className="text-xl font-bold mb-1">Ready to Deploy?</h2>
              <p className="opacity-90">
                Your model is analyzed and ready for production deployment
              </p>
            </div>
            <button
              onClick={() => navigate(`/deploy/${modelId}`)}
              className="flex items-center px-6 py-3 bg-white text-green-700 rounded-lg font-semibold hover:bg-green-50 transition shadow-md"
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

export default Explainability;
