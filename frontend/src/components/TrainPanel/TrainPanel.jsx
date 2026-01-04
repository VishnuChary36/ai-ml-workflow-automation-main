import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Loader, Eye, Rocket, BarChart3, Star, Clock, Zap, CheckCircle, AlertCircle } from 'lucide-react';
import Console from '../Console/Console';
import { trainModel, getTaskStatus } from '../../api/client';

const TrainPanel = ({ datasetId, modelSuggestions = [], targetColumn, setDataset }) => {
  const [trainingTaskId, setTrainingTaskId] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState({});
  const [trainingProgress, setTrainingProgress] = useState({});
  const [completedModelId, setCompletedModelId] = useState(null);
  const [autoTraining, setAutoTraining] = useState(false);
  const navigate = useNavigate();
  
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
        return <CheckCircle className="text-green-500 ml-2" size={20} />;
      case 'error':
        return <AlertCircle className="text-red-500 ml-2" size={20} />;
      case 'training':
      case 'running':
        return <Loader className="text-blue-500 animate-spin ml-2" size={20} />;
      default:
        return null;
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Model Training</h2>
          <p className="text-gray-600 text-sm mt-1">
            AI-powered model selection based on your dataset characteristics
          </p>
        </div>
        
        {/* Auto Train Button */}
        {recommendedModel && !autoTraining && (
          <button
            onClick={handleAutoTrain}
            disabled={trainingStatus[recommendedModel.model]?.status === 'running'}
            className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 transition flex items-center shadow-lg"
          >
            <Zap size={20} className="mr-2" />
            Auto Train Best Model
          </button>
        )}
      </div>
      
      {/* Dataset Analysis Summary */}
      {modelSuggestions.length > 0 && modelSuggestions[0].dataset_analysis && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 className="font-semibold text-blue-800 mb-2">📊 Dataset Analysis</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-blue-600">Samples:</span>
              <span className="ml-2 font-semibold">{modelSuggestions[0].dataset_analysis.samples?.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-blue-600">Features:</span>
              <span className="ml-2 font-semibold">{modelSuggestions[0].dataset_analysis.features}</span>
            </div>
            <div>
              <span className="text-blue-600">Problem Type:</span>
              <span className="ml-2 font-semibold capitalize">{modelSuggestions[0].dataset_analysis.problem_type}</span>
            </div>
            {modelSuggestions[0].dataset_analysis.n_classes > 0 && (
              <div>
                <span className="text-blue-600">Classes:</span>
                <span className="ml-2 font-semibold">{modelSuggestions[0].dataset_analysis.n_classes}</span>
              </div>
            )}
          </div>
        </div>
      )}
      
      {modelSuggestions.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-gray-500">Complete preprocessing to see model suggestions</p>
        </div>
      ) : (
        <div className="space-y-4">
          {modelSuggestions.map((suggestion, index) => {
            const status = trainingStatus[suggestion.model];
            const progress = trainingProgress[suggestion.model];
            const isRecommended = suggestion.is_recommended;
            
            return (
              <div 
                key={index} 
                className={`border-2 rounded-lg p-4 transition-all ${
                  isRecommended 
                    ? 'border-green-400 bg-green-50 shadow-md' 
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              >
                {/* Recommended Badge */}
                {isRecommended && (
                  <div className="flex items-center mb-3">
                    <span className="inline-flex items-center px-3 py-1 bg-green-500 text-white text-xs font-bold rounded-full">
                      <Star size={12} className="mr-1" />
                      RECOMMENDED FOR YOUR DATA
                    </span>
                    {suggestion.recommendation_reason && (
                      <span className="ml-2 text-sm text-green-700">
                        {suggestion.recommendation_reason}
                      </span>
                    )}
                  </div>
                )}
                
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center">
                      <h3 className="font-semibold text-lg">{suggestion.model}</h3>
                      {getStatusIcon(status)}
                    </div>
                    
                    <p className="text-sm text-gray-600 mt-2">{suggestion.rationale}</p>
                    
                    <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div className="flex items-center">
                        <BarChart3 size={14} className="text-gray-400 mr-1" />
                        <span className="text-gray-500">Score:</span>
                        <span className={`ml-1 font-semibold ${
                          suggestion.score >= 80 ? 'text-green-600' :
                          suggestion.score >= 60 ? 'text-yellow-600' : 'text-gray-600'
                        }`}>
                          {suggestion.score?.toFixed(0) || Math.round(suggestion.confidence * 100)}%
                        </span>
                      </div>
                      <div className="flex items-center">
                        <Clock size={14} className="text-gray-400 mr-1" />
                        <span className="text-gray-500">Est. Time:</span>
                        <span className="ml-1 font-semibold">
                          {formatTime(suggestion.estimated_time_seconds)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">Iterations:</span>
                        <span className="ml-1 font-semibold">
                          {suggestion.training_iterations || suggestion.params?.n_estimators || 100}
                        </span>
                      </div>
                    </div>
                    
                    {/* Training Progress Bar */}
                    {status && (status.status === 'training' || status.status === 'running') && (
                      <div className="mt-4">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-blue-600 font-medium">
                            {progress?.currentStep || 'Training...'}
                          </span>
                          <span className="text-gray-500">
                            ETA: {formatTime(progress?.eta)}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-blue-600 h-2.5 rounded-full transition-all duration-500 animate-pulse"
                            style={{ width: `${Math.max(progress?.percent || 10, 10)}%` }}
                          />
                        </div>
                      </div>
                    )}
                    
                    {/* Completed Metrics */}
                    {status?.status === 'completed' && status.metrics && (
                      <div className="mt-4 p-3 bg-green-100 rounded-lg">
                        <h4 className="font-semibold text-green-800 text-sm mb-2">✅ Training Results</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                          {status.metrics.accuracy !== undefined && (
                            <div>
                              <span className="text-green-700">Accuracy:</span>
                              <span className="ml-1 font-bold text-green-800">
                                {(status.metrics.accuracy * 100).toFixed(1)}%
                              </span>
                            </div>
                          )}
                          {status.metrics.f1_score !== undefined && (
                            <div>
                              <span className="text-green-700">F1:</span>
                              <span className="ml-1 font-bold text-green-800">
                                {(status.metrics.f1_score * 100).toFixed(1)}%
                              </span>
                            </div>
                          )}
                          {status.metrics.r2_score !== undefined && (
                            <div>
                              <span className="text-green-700">R²:</span>
                              <span className="ml-1 font-bold text-green-800">
                                {status.metrics.r2_score.toFixed(4)}
                              </span>
                            </div>
                          )}
                        </div>
                        
                        {/* Navigation Buttons after training completes */}
                        <div className="flex items-center gap-3 mt-4 pt-3 border-t border-green-200">
                          <button
                            onClick={() => navigate('/dashboard')}
                            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-md text-sm font-medium"
                          >
                            <BarChart3 size={16} />
                            Go to Dashboard
                          </button>
                          {completedModelId && (
                            <button
                              onClick={() => navigate(`/visualizations/${completedModelId}`)}
                              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:from-green-700 hover:to-emerald-700 transition-all shadow-md text-sm font-medium"
                            >
                              <Eye size={16} />
                              View Visualizations
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Status Message */}
                    {status && status.status !== 'completed' && (
                      <div className="mt-3">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          status.status === 'training' ? 'bg-yellow-100 text-yellow-800' :
                          status.status === 'running' ? 'bg-blue-100 text-blue-800' :
                          status.status === 'error' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {status.message}
                        </span>
                      </div>
                    )}
                  </div>
                  
                  <button 
                    onClick={() => handleTrainModel(suggestion)}
                    disabled={status && (status.status === 'training' || status.status === 'running')}
                    className={`ml-4 px-4 py-2 rounded-lg transition flex items-center ${
                      status && (status.status === 'training' || status.status === 'running')
                        ? 'bg-gray-400 text-white cursor-not-allowed'
                        : isRecommended
                          ? 'bg-green-600 text-white hover:bg-green-700 shadow-md'
                          : 'bg-blue-600 text-white hover:bg-blue-700'
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
                        {isRecommended ? 'Train Best' : 'Train'}
                      </>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
      
      {/* Training Console */}
      {trainingTaskId && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3 flex items-center">
            <span className="mr-2">📋</span>
            Training Logs
          </h3>
          <Console taskId={trainingTaskId} autoConnect={true} />
        </div>
      )}
    </div>
  );
};

export default TrainPanel;
