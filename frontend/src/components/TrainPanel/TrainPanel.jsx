import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Loader, Eye, Rocket, BarChart3, Star, Clock, Zap, CheckCircle, AlertCircle, TrendingUp, Award, Target, Sparkles, ChevronDown, ChevronUp } from 'lucide-react';
import Console from '../Console/Console';
import PageHeader from '../UI/PageHeader';
import EmptyState from '../UI/EmptyState';
import { trainModel, getTaskStatus } from '../../api/client';

const TrainPanel = ({ datasetId, modelSuggestions = [], targetColumn, setDataset }) => {
  // Restore persisted training state from sessionStorage
  const savedTrainingState = React.useMemo(() => {
    try {
      return JSON.parse(sessionStorage.getItem('ml_trainingState') || '{}');
    } catch { return {}; }
  }, []);

  const [trainingTaskId, setTrainingTaskId] = useState(savedTrainingState.trainingTaskId || null);
  const [trainingStatus, setTrainingStatus] = useState(savedTrainingState.trainingStatus || {});
  const [trainingProgress, setTrainingProgress] = useState({});
  const [completedModelId, setCompletedModelId] = useState(savedTrainingState.completedModelId || null);
  const [autoTraining, setAutoTraining] = useState(false);
  const [showAllModels, setShowAllModels] = useState(false);
  const navigate = useNavigate();

  // Persist training state to sessionStorage whenever it changes
  useEffect(() => {
    const state = { trainingTaskId, trainingStatus, completedModelId };
    sessionStorage.setItem('ml_trainingState', JSON.stringify(state));
  }, [trainingTaskId, trainingStatus, completedModelId]);
  
  // Find the recommended model
  const recommendedModel = modelSuggestions.find(s => s.is_recommended);
  
  const handleTrainModel = async (suggestion) => {
    try {
      setTrainingStatus(prev => ({
        ...prev,
        [suggestion.model]: { status: 'training', message: 'Initializing training...', step: 'init' }
      }));
      
      setTrainingProgress(prev => ({
        ...prev,
        [suggestion.model]: { percent: 0, currentStep: 'Preparing...', eta: null }
      }));
      
      const result = await trainModel(datasetId, suggestion, targetColumn);
      setTrainingTaskId(result.task_id);
      
      setTrainingStatus(prev => ({
        ...prev,
        [suggestion.model]: { status: 'running', message: 'Model training in progress...', step: 'training' }
      }));
      
      // Monitor task status with progress updates
      const taskStatusInterval = setInterval(async () => {
        try {
          const status = await getTaskStatus(result.task_id);
          
          // Update progress based on status
          if (status.status === 'running') {
            // Extract progress from logs if available
            setTrainingProgress(prev => ({
              ...prev,
              [suggestion.model]: {
                percent: prev[suggestion.model]?.percent || 0,
                currentStep: 'Training model...',
                eta: suggestion.estimated_time_seconds || null
              }
            }));
          }
          
          if (status.status === 'completed') {
            clearInterval(taskStatusInterval);
            
            setTrainingProgress(prev => ({
              ...prev,
              [suggestion.model]: { percent: 100, currentStep: 'Complete!', eta: 0 }
            }));
            
            setTrainingStatus(prev => ({
              ...prev,
              [suggestion.model]: { 
                status: 'completed', 
                message: 'Training completed successfully!',
                metrics: status.result?.metrics
              }
            }));
            
            if (status.result && status.result.model_id) {
              setCompletedModelId(status.result.model_id);
              setAutoTraining(false);
              
              // Navigate to visualization after a short delay
              setTimeout(() => {
                navigate(`/visualizations/${status.result.model_id}`);
              }, 2500);
            }
          } else if (status.status === 'failed') {
            clearInterval(taskStatusInterval);
            setAutoTraining(false);
            setTrainingStatus(prev => ({
              ...prev,
              [suggestion.model]: { status: 'error', message: 'Training failed' }
            }));
          }
        } catch (error) {
          console.error('Error checking task status:', error);
          clearInterval(taskStatusInterval);
          setAutoTraining(false);
        }
      }, 1000);
    } catch (error) {
      console.error('Error starting training:', error);
      setAutoTraining(false);
      setTrainingStatus(prev => ({
        ...prev,
        [suggestion.model]: { status: 'error', message: 'Failed to start training' }
      }));
    }
  };
  
  const handleAutoTrain = () => {
    if (recommendedModel) {
      setAutoTraining(true);
      handleTrainModel(recommendedModel);
    }
  };
  
  const formatTime = (seconds) => {
    if (!seconds) return '--';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  };
  
  const getStatusIcon = (status) => {
    switch (status?.status) {
      case 'completed':
        return <CheckCircle className="text-success-500 ml-2" size={20} />;
      case 'error':
        return <AlertCircle className="text-error-400 ml-2" size={20} />;
      case 'training':
      case 'running':
        return <Loader className="text-primary-500 animate-spin ml-2" size={20} />;
      default:
        return null;
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-success-500';
    if (score >= 60) return 'text-warning-500';
    return 'text-on-surface-variant';
  };
  
  return (
    <div className="section-card-elevated p-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div className="flex items-center space-x-4">
          <div className="w-14 h-14 bg-gradient-to-br from-success-500 to-success-600 rounded-xl flex items-center justify-center shadow-lg shadow-success-500/20">
            <TrendingUp className="text-white" size={26} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-on-surface">Model Training</h2>
            <p className="text-sm text-on-surface-variant mt-0.5">
              Select and train a machine learning model
            </p>
          </div>
        </div>
        
        {/* Auto Train Button */}
        {recommendedModel && !autoTraining && (
          <button
            onClick={handleAutoTrain}
            disabled={trainingStatus[recommendedModel.model]?.status === 'running'}
            className="btn-success flex items-center shadow-lg shadow-success-500/20"
          >
            <Zap size={18} className="mr-2" />
            Auto Train AI Pick
          </button>
        )}
      </div>
      
      {/* Dataset Analysis Summary */}
      {modelSuggestions.length > 0 && modelSuggestions[0].dataset_analysis && (
        <div className="mb-6 p-5 rounded-xl" style={{ background: 'rgba(77, 142, 255, 0.06)', border: '1px solid rgba(77, 142, 255, 0.12)' }}>
          <h3 className="font-semibold text-on-surface mb-4 flex items-center">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center mr-3" style={{ background: 'rgba(77, 142, 255, 0.12)' }}>
              <Target size={16} className="text-primary-500" />
            </div>
            Dataset Analysis Summary
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="metric-card">
              <span className="text-xs text-on-surface-variant block">Samples</span>
              <span className="text-xl font-bold text-on-surface">{modelSuggestions[0].dataset_analysis.samples?.toLocaleString()}</span>
            </div>
            <div className="metric-card">
              <span className="text-xs text-on-surface-variant block">Features</span>
              <span className="text-xl font-bold text-on-surface">{modelSuggestions[0].dataset_analysis.features}</span>
            </div>
            <div className="metric-card">
              <span className="text-xs text-on-surface-variant block">Problem Type</span>
              <span className="text-xl font-bold text-on-surface capitalize">{modelSuggestions[0].dataset_analysis.problem_type}</span>
            </div>
            {modelSuggestions[0].dataset_analysis.n_classes > 0 && (
              <div className="metric-card">
                <span className="text-xs text-on-surface-variant block">Classes</span>
                <span className="text-xl font-bold text-on-surface">{modelSuggestions[0].dataset_analysis.n_classes}</span>
              </div>
            )}
          </div>
        </div>
      )}
      
      {modelSuggestions.length === 0 ? (
        <EmptyState
          icon={TrendingUp}
          title="Complete preprocessing to see model suggestions"
          description="Run the data processing pipeline to analyze your dataset and get AI-powered model recommendations."
        />
      ) : (
        <div className="space-y-4">
          {/* AI Recommendation Banner */}
          {recommendedModel && (
            <div className="p-4 rounded-xl" style={{ background: 'rgba(77, 142, 255, 0.06)', border: '1px solid rgba(77, 142, 255, 0.15)' }}>
              <div className="flex items-start space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-success-500 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg shadow-primary-500/20">
                  <Sparkles size={20} className="text-white" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-1">
                    <h3 className="font-bold text-on-surface text-sm">AI Model Recommendation</h3>
                    <span className="px-2 py-0.5 text-xs font-bold rounded-full" style={{ background: 'rgba(77, 142, 255, 0.15)', color: '#adc6ff' }}>
                      Score: {recommendedModel.score?.toFixed(0) || Math.round(recommendedModel.confidence * 100)}/100
                    </span>
                  </div>
                  <p className="text-sm text-on-surface-variant">
                    {recommendedModel.ai_suggestion || recommendedModel.recommendation_reason || `${recommendedModel.model} is the best fit for your dataset.`}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Model Cards - show top 3 by default, rest behind toggle */}
          {(() => {
            const visibleModels = showAllModels ? modelSuggestions : modelSuggestions.slice(0, 3);
            const hiddenCount = modelSuggestions.length - 3;
            return (
              <>
                {visibleModels.map((suggestion, index) => {
            const status = trainingStatus[suggestion.model];
            const progress = trainingProgress[suggestion.model];
            const isRecommended = suggestion.is_recommended;
            
            return (
              <div 
                key={index} 
                className={`section-card overflow-hidden transition-all duration-200 ${
                  isRecommended 
                    ? 'ring-2 ring-success-500/30 shadow-lg shadow-success-500/5' 
                    : 'border-outline-variant hover:border-primary-200 hover:shadow-md'
                }`}
                style={isRecommended ? { borderColor: 'rgba(16, 185, 129, 0.3)' } : {}}
              >
                <div className="p-5">
                  {/* Recommended Badge */}
                  {isRecommended && (
                    <div className="flex items-center mb-4">
                      <span className="inline-flex items-center px-3 py-1.5 bg-gradient-to-r from-success-500 to-success-600 text-white text-xs font-bold rounded-lg shadow-sm">
                        <Sparkles size={14} className="mr-1.5" />
                        AI RECOMMENDED
                      </span>
                      {suggestion.recommendation_reason && (
                        <span className="ml-3 text-sm text-success-500 font-medium">
                          {suggestion.recommendation_reason}
                        </span>
                      )}
                    </div>
                  )}
                  
                  <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center">
                        <h3 className="font-semibold text-lg text-on-surface">{suggestion.model}</h3>
                        {getStatusIcon(status)}
                      </div>
                      
                      <p className="text-sm text-on-surface-variant mt-2 leading-relaxed">{suggestion.rationale}</p>
                      
                      <div className="mt-4 flex flex-wrap gap-3 text-sm">
                        <div className="flex items-center bg-surface-container-high px-3 py-1.5 rounded-lg">
                          <BarChart3 size={14} className="text-on-surface-variant mr-1.5" />
                          <span className="text-on-surface-variant">Score:</span>
                          <span className={`ml-1.5 font-semibold ${getScoreColor(suggestion.score || suggestion.confidence * 100)}`}>
                            {suggestion.score?.toFixed(0) || Math.round(suggestion.confidence * 100)}%
                          </span>
                        </div>
                        <div className="flex items-center bg-surface-container-low px-3 py-1.5 rounded-lg">
                          <Clock size={14} className="text-on-surface-variant mr-1.5" />
                          <span className="text-on-surface-variant">Est. Time:</span>
                          <span className="ml-1.5 font-semibold text-on-surface-variant">
                            {formatTime(suggestion.estimated_time_seconds)}
                          </span>
                        </div>
                        <div className="flex items-center bg-surface-container-low px-3 py-1.5 rounded-lg">
                          <span className="text-on-surface-variant">Iterations:</span>
                          <span className="ml-1.5 font-semibold text-on-surface-variant">
                            {suggestion.training_iterations || suggestion.params?.n_estimators || 100}
                          </span>
                        </div>
                      </div>
                      
                      {/* Training Progress Bar */}
                      {status && (status.status === 'training' || status.status === 'running') && (
                        <div className="mt-5">
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-primary-500 font-medium">
                              {progress?.currentStep || 'Training...'}
                            </span>
                            <span className="text-on-surface-variant">
                              ETA: {formatTime(progress?.eta)}
                            </span>
                          </div>
                          <div className="progress-bar">
                            <div 
                              className="progress-fill bg-primary-500 animate-pulse-subtle"
                              style={{ width: `${Math.max(progress?.percent || 10, 10)}%` }}
                            />
                          </div>
                        </div>
                      )}
                      
                      {/* Completed Metrics */}
                      {status?.status === 'completed' && status.metrics && (
                        <div className="mt-5 p-4 rounded-xl" style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                          <h4 className="font-semibold text-success-500 text-sm mb-3 flex items-center">
                            <CheckCircle size={16} className="mr-2" />
                            Training Completed Successfully
                          </h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {status.metrics.accuracy !== undefined && (
                              <div className="p-2 rounded-lg" style={{ background: 'rgba(16, 185, 129, 0.06)', border: '1px solid rgba(16, 185, 129, 0.12)' }}>
                                <span className="text-xs text-success-500 block">Accuracy</span>
                                <span className="text-lg font-bold text-success-400">
                                  {(status.metrics.accuracy * 100).toFixed(1)}%
                                </span>
                              </div>
                            )}
                            {status.metrics.f1_score !== undefined && (
                              <div className="p-2 rounded-lg" style={{ background: 'rgba(16, 185, 129, 0.06)', border: '1px solid rgba(16, 185, 129, 0.12)' }}>
                                <span className="text-xs text-success-500 block">F1 Score</span>
                                <span className="text-lg font-bold text-success-400">
                                  {(status.metrics.f1_score * 100).toFixed(1)}%
                                </span>
                              </div>
                            )}
                            {status.metrics.r2_score !== undefined && (
                              <div className="p-2 rounded-lg" style={{ background: 'rgba(16, 185, 129, 0.06)', border: '1px solid rgba(16, 185, 129, 0.12)' }}>
                                <span className="text-xs text-success-500 block">R² Score</span>
                                <span className="text-lg font-bold text-success-400">
                                  {status.metrics.r2_score.toFixed(4)}
                                </span>
                              </div>
                            )}
                          </div>
                          
                          {/* Navigation Buttons after training completes */}
                          <div className="flex flex-wrap items-center gap-3 mt-4 pt-4" style={{ borderTop: '1px solid rgba(16, 185, 129, 0.15)' }}>
                            {completedModelId && (
                              <button
                                onClick={() => navigate(`/visualizations/${completedModelId}`)}
                                className="btn-success text-sm flex items-center"
                              >
                                <Eye size={16} className="mr-2" />
                                View Visualizations
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {/* Status Message */}
                      {status && status.status !== 'completed' && (
                        <div className="mt-4">
                          <span className={`badge ${
                            status.status === 'training' ? 'badge-warning' :
                            status.status === 'running' ? 'badge-primary' :
                            status.status === 'error' ? 'badge-error' : 'badge-slate'
                          }`}>
                            {status.message}
                          </span>
                        </div>
                      )}
                    </div>
                    
                    <button 
                      onClick={() => handleTrainModel(suggestion)}
                      disabled={status && (status.status === 'training' || status.status === 'running')}
                      className={`flex-shrink-0 flex items-center ${
                        status && (status.status === 'training' || status.status === 'running')
                          ? 'btn-secondary opacity-50 cursor-not-allowed'
                          : isRecommended
                            ? 'btn-success'
                            : 'btn-primary'
                      }`}
                    >
                      {status && (status.status === 'training' || status.status === 'running') ? (
                        <>
                          <Loader size={16} className="mr-2 animate-spin" />
                          Training...
                        </>
                      ) : (
                        <>
                          <Play size={16} className="mr-2" />
                          {isRecommended ? 'Train Best' : 'Train Model'}
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}

                {/* Show More / Show Less toggle */}
                {hiddenCount > 0 && (
                  <button
                    onClick={() => setShowAllModels(!showAllModels)}
                    className="w-full py-3 px-4 bg-surface-container-low hover:bg-surface-container-high border border-outline-variant rounded-xl text-sm font-medium text-on-surface-variant hover:text-on-surface transition-all flex items-center justify-center space-x-2"
                  >
                    {showAllModels ? (
                      <>
                        <ChevronUp size={16} />
                        <span>Show Less Models</span>
                      </>
                    ) : (
                      <>
                        <ChevronDown size={16} />
                        <span>Show {hiddenCount} More Model{hiddenCount > 1 ? 's' : ''}</span>
                      </>
                    )}
                  </button>
                )}
              </>
            );
          })()}
        </div>
      )}
      
      {/* Training Console */}
      {trainingTaskId && (
        <div className="mt-6 animate-in">
          <h3 className="text-lg font-semibold text-on-surface mb-4 flex items-center">
            Training Logs
          </h3>
          <Console taskId={trainingTaskId} autoConnect={true} />
        </div>
      )}
    </div>
  );
};

export default TrainPanel;
