import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, Eye, Info, AlertTriangle, CheckCircle, HelpCircle, Zap, GitBranch, BarChart3, Target, Layers, Rocket } from 'lucide-react';
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
      <div className="page-container">
        <div className="content-wrapper">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <Eye className="text-white" size={20} />
              </div>
              <h1 className="text-2xl font-bold text-on-surface">Model Explainability</h1>
            </div>
          </div>
          <div className="flex flex-col justify-center items-center h-64 space-y-4">
            <div className="w-12 h-12 border-4 border-primary-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-on-surface-variant font-medium">Generating explainability analysis...</p>
            <p className="text-sm text-on-surface-variant">This may take a minute for SHAP and LIME calculations</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="content-wrapper">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <Eye className="text-white" size={20} />
              </div>
              <h1 className="text-2xl font-bold text-on-surface">Model Explainability</h1>
            </div>
            <button
              onClick={onBack}
              className="btn-secondary"
            >
              <ArrowLeft size={18} />
              Back to Visualizations
            </button>
          </div>
          <div className="card p-6 text-center" style={{ background: 'rgba(239, 68, 68, 0.06)', borderColor: 'rgba(239, 68, 68, 0.2)' }}>
            <AlertTriangle className="mx-auto text-error-400 mb-4" size={48} />
            <p className="text-error-400 mb-4">{error}</p>
            <button
              onClick={fetchExplainability}
              className="btn-primary"
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
      <div className="page-container">
        <div className="content-wrapper">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <Eye className="text-white" size={20} />
              </div>
              <h1 className="text-2xl font-bold text-on-surface">Model Explainability</h1>
            </div>
            <button
              onClick={onBack}
              className="btn-secondary"
            >
              <ArrowLeft size={18} />
              Back to Visualizations
            </button>
          </div>
          <div className="card p-6 text-center" style={{ background: 'rgba(245, 158, 11, 0.06)', borderColor: 'rgba(245, 158, 11, 0.2)' }}>
            <p className="text-warning-400">No explainability data available for this model.</p>
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
      <div key={key} className="explain-card overflow-hidden">
        <div 
          className="flex justify-between items-center p-5 border-b border-outline-variant cursor-pointer hover:bg-surface-container-low/50 transition-colors"
          onClick={() => toggleSection(key)}
        >
          <div className="flex-1">
            <div className="flex items-center">
              <h2 className="text-lg font-semibold text-on-surface">{title}</h2>
              <HelpCircle 
                size={16} 
                className="ml-2 text-on-surface-variant hover:text-primary-500 transition-colors" 
                title={desc.interpretation || "Click for more info"}
              />
            </div>
            {desc.description && (
              <p className="text-sm text-on-surface-variant mt-1">{desc.description}</p>
            )}
          </div>
          {plotImage && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                downloadVisualization(plotImage, title);
              }}
              className="btn-primary flex items-center text-sm ml-4"
            >
              <Download size={14} className="mr-1" />
              Download
            </button>
          )}
        </div>
        
        {/* Interpretation box - always visible */}
        {desc.interpretation && (
          <div className="px-5 py-3 border-b rounded-t" style={{ background: 'rgba(77, 142, 255, 0.06)', borderColor: 'rgba(77, 142, 255, 0.1)' }}>
            <div className="flex items-start">
              <Info size={16} className="text-primary-500 mr-2 mt-0.5 flex-shrink-0" />
              <p className="text-sm text-on-surface-variant">
                <strong className="text-primary-500">How to interpret:</strong> {desc.interpretation}
              </p>
            </div>
          </div>
        )}
        
        {/* Additional info for LIME */}
        {data.prediction !== undefined && (
          <div className="px-5 py-2 bg-surface-container-low border-b border-outline-variant flex space-x-6">
            <span className="text-sm">
              <strong className="text-primary-500">Predicted:</strong> <span className="text-on-surface-variant">{data.prediction}</span>
            </span>
            {data.actual !== undefined && (
              <span className="text-sm">
                <strong className="text-primary-500">Actual:</strong> <span className="text-on-surface-variant">{data.actual}</span>
              </span>
            )}
          </div>
        )}
        
        {/* Fidelity for surrogate tree */}
        {data.fidelity !== undefined && (
          <div className="px-5 py-2 border-b" style={{ background: 'rgba(16, 185, 129, 0.06)', borderColor: 'rgba(16, 185, 129, 0.15)' }}>
            <span className="text-sm">
              <strong className="text-success-500">Fidelity Score:</strong> <span className="text-on-surface-variant">{(data.fidelity * 100).toFixed(1)}%</span>
              <span className="text-on-surface-variant ml-2">(how well this tree mimics the original model)</span>
            </span>
          </div>
        )}
        
        {/* High correlations warning */}
        {data.high_correlations && data.high_correlations.length > 0 && (
          <div className="px-5 py-2 border-b" style={{ background: 'rgba(245, 158, 11, 0.06)', borderColor: 'rgba(245, 158, 11, 0.15)' }}>
            <div className="flex items-start">
              <AlertTriangle size={16} className="text-warning-500 mr-2 mt-0.5" />
              <div className="text-sm">
                <strong className="text-warning-500">High Correlations Detected:</strong>
                <ul className="list-disc ml-4 mt-1 text-on-surface-variant">
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
          <div className="px-4 py-3 bg-surface-container-low border-b border-outline-variant overflow-x-auto">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-semibold text-on-surface-variant">Per-Class Metrics (Top Classes):</h4>
              {data.total_classes && data.shown_classes && data.total_classes > data.shown_classes && (
                <span className="text-xs text-on-surface-variant">
                  Showing {data.shown_classes} of {data.total_classes} classes
                </span>
              )}
            </div>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-on-surface-variant bg-surface-container-high uppercase text-xs">
                  <th className="px-3 py-2 rounded-tl-lg">Class</th>
                  <th className="px-3 py-2">Precision</th>
                  <th className="px-3 py-2">Recall</th>
                  <th className="px-3 py-2">F1 Score</th>
                  <th className="px-3 py-2 rounded-tr-lg">Support</th>
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
                    <tr key={cls} className={idx % 2 === 0 ? 'bg-surface-container' : 'bg-surface-container-low'}>
                      <td className="px-3 py-2 font-medium text-on-surface-variant">{cls}</td>
                      <td className="px-3 py-2 text-on-surface-variant">{formatMetric(metrics.precision)}</td>
                      <td className="px-3 py-2 text-on-surface-variant">{formatMetric(metrics.recall)}</td>
                      <td className="px-3 py-2 text-on-surface-variant">{formatMetric(metrics.f1)}</td>
                      <td className="px-3 py-2 text-on-surface-variant">{metrics.support}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        
        {/* Plot image */}
        {plotImage && (
          <div className="p-6 flex justify-center bg-surface-container-low">
            <img 
              src={`data:image/png;base64,${plotImage}`} 
              alt={title} 
              className="max-w-full h-auto rounded-xl shadow-soft"
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
          <h2 className="text-xl font-bold text-on-surface ml-2">{title}</h2>
          <span className="ml-2 badge badge-primary">
            {items.length} {items.length === 1 ? 'analysis' : 'analyses'}
          </span>
        </div>
        {description && (
          <p className="text-on-surface-variant mb-4 ml-8">{description}</p>
        )}
        <div className="space-y-4">
          {items.map(item => renderExplanationCard(item))}
        </div>
      </div>
    );
  };

  return (
    <div className="page-container">
      <div className="content-wrapper">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <Eye className="text-white" size={20} />
              </div>
              <h1 className="text-2xl font-bold text-on-surface">Model Explainability</h1>
            </div>
            <p className="text-on-surface-variant ml-13 mt-1">Deep-dive analysis into how your model makes predictions</p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => navigate(`/deploy/${modelId}`)}
              className="btn-success"
            >
              <Rocket size={18} />
              Deploy Model
            </button>
            <button
              onClick={onBack}
              className="btn-secondary"
            >
              <ArrowLeft size={18} />
              Back to Visualizations
            </button>
          </div>
        </div>

        {/* Model Info Banner */}
        <div className="card p-6 mb-6" style={{ background: 'rgba(77, 142, 255, 0.06)', borderColor: 'rgba(77, 142, 255, 0.15)' }}>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center">
              <Layers size={20} className="text-primary-500 mr-2" />
              <div>
                <p className="text-sm text-on-surface-variant">Model</p>
                <p className="font-bold text-lg text-on-surface">{modelName}</p>
              </div>
            </div>
            <div className="flex items-center">
              <Target size={20} className="text-success-500 mr-2" />
              <div>
                <p className="text-sm text-on-surface-variant">Target</p>
                <p className="font-bold text-lg text-on-surface">{targetColumn}</p>
              </div>
            </div>
            <div className="flex items-center">
              <BarChart3 size={20} className="text-warning-500 mr-2" />
              <div>
                <p className="text-sm text-on-surface-variant">Type</p>
                <p className="font-bold text-lg text-on-surface">{isClassification ? 'Classification' : 'Regression'}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Guide Card */}
        <div className="card p-6 mb-8 border-l-4 border-primary-500">
          <div className="flex items-start">
            <Info size={24} className="text-primary-500 mr-3 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-bold text-on-surface mb-2">Understanding This Page</h3>
              <p className="text-on-surface-variant text-sm">
                This page provides multiple perspectives on how your model makes decisions. 
                <strong className="text-primary-500"> Global explanations</strong> (SHAP, Permutation Importance) show overall feature importance.
                <strong className="text-success-500"> Local explanations</strong> (LIME) show why specific predictions were made.
                <strong className="text-warning-500"> Data analysis</strong> helps you understand your training data.
                Each chart includes an interpretation guide to help you understand the insights.
              </p>
            </div>
          </div>
        </div>

        {/* Global Explanations */}
        {renderSection(
          "Global Feature Importance",
          <Zap className="text-warning-500" size={24} />,
          globalExplanations,
          "These explanations show which features are most important across all predictions."
        )}

        {/* Local Explanations */}
        {renderSection(
          "Local Explanations (Individual Predictions)",
          <Target className="text-success-500" size={24} />,
          localExplanations,
          "These explain specific predictions to help you understand individual model decisions."
        )}

        {/* Performance Analysis */}
        {renderSection(
          "Performance Analysis",
          <CheckCircle className="text-primary-500" size={24} />,
          performanceAnalysis,
          "Detailed metrics and calibration analysis for model performance."
        )}

        {/* Data Analysis */}
        {renderSection(
          "Data Analysis",
          <BarChart3 className="text-on-surface-variant" size={24} />,
          dataAnalysis,
          "Understand the characteristics of your training data."
        )}

        {/* No explanations available */}
        {globalExplanations.length === 0 && localExplanations.length === 0 && 
         performanceAnalysis.length === 0 && dataAnalysis.length === 0 && (
          <div className="card p-6 text-center" style={{ background: 'rgba(245, 158, 11, 0.06)', borderColor: 'rgba(245, 158, 11, 0.15)' }}>
            <AlertTriangle className="mx-auto text-warning-400 mb-4" size={48} />
            <p className="text-warning-400">No explainability analyses could be generated for this model.</p>
            <p className="text-on-surface-variant text-sm mt-2">This may be due to missing data or unsupported model type.</p>
          </div>
        )}

        {/* Deploy Model CTA */}
        <div className="card bg-gradient-to-r from-success-500 to-success-600 p-6 mt-8">
          <div className="flex items-center justify-between">
            <div className="text-white">
              <h2 className="text-xl font-bold mb-1">Ready to Deploy?</h2>
              <p className="opacity-90">
                Your model is analyzed and ready for production deployment
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

export default Explainability;
