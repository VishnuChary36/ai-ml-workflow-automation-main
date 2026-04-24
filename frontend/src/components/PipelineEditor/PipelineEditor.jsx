import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle, Circle, Eye, Play, Loader, ChevronDown, ChevronUp, BarChart3, Workflow, Settings, Percent } from 'lucide-react';
import Console from '../Console/Console';
import TrainPanel from '../TrainPanel/TrainPanel';
import { getSuggestedPipeline, runPipeline, getSuggestedModels, getTaskStatus } from '../../api/client';
import { useSessionStorage } from '../../hooks/useSessionStorage';
import toast from 'react-hot-toast';

const PipelineEditor = ({ datasetId, targetColumn, onPipelineComplete }) => {
  const navigate = useNavigate();
  const [suggestions, setSuggestions] = useState([]);
  const [selectedSteps, setSelectedSteps] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  const [pipelineState, setPipelineState] = useSessionStorage('ml_pipelineState', {});
  const taskId = pipelineState.taskId || null;
  const showTrainPanel = pipelineState.showTrainPanel || false;
  const modelSuggestions = pipelineState.modelSuggestions || [];
  const pipelineCompleted = pipelineState.pipelineCompleted || false;

  const setTaskId = (val) => setPipelineState(prev => ({...prev, taskId: val}));
  const setShowTrainPanel = (val) => setPipelineState(prev => ({...prev, showTrainPanel: val}));
  const setModelSuggestions = (val) => setPipelineState(prev => ({...prev, modelSuggestions: val}));
  const setPipelineCompleted = (val) => setPipelineState(prev => ({...prev, pipelineCompleted: val}));
  
  const [expandedStep, setExpandedStep] = useState(null);
  const [previewStep, setPreviewStep] = useState(null);

  useEffect(() => {
    loadSuggestions();
  }, [datasetId]);

  const loadSuggestions = async () => {
    setLoading(true);
    try {
      const result = await getSuggestedPipeline(datasetId, targetColumn);
      setSuggestions(result.suggestions);
      // Select all steps by default
      setSelectedSteps(new Set(result.suggestions.map(s => s.id)));
    } catch (error) {
      console.error('Error loading suggestions:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleStep = (stepId) => {
    const newSelected = new Set(selectedSteps);
    if (newSelected.has(stepId)) {
      newSelected.delete(stepId);
    } else {
      newSelected.add(stepId);
    }
    setSelectedSteps(newSelected);
  };

  const toggleExpand = (stepId) => {
    setExpandedStep(expandedStep === stepId ? null : stepId);
  };

  const handlePreview = (step) => {
    setPreviewStep(step);
  };

  const handleRunPipeline = async () => {
    const stepsToRun = suggestions.filter(s => selectedSteps.has(s.id));
    
    if (stepsToRun.length === 0) {
      toast.error('Please select at least one step');
      return;
    }

    setRunning(true);
    try {
      const result = await runPipeline(datasetId, stepsToRun);
      setTaskId(result.task_id);
      
      // After pipeline completes, fetch model suggestions
      const taskStatusInterval = setInterval(async () => {
        try {
          const status = await getTaskStatus(result.task_id);
          if (status.status === 'completed') {
            clearInterval(taskStatusInterval);
            setPipelineCompleted(true);
            
            // Fetch model suggestions
            if (targetColumn) {
              try {
                const modelSuggestionData = await getSuggestedModels(datasetId, targetColumn);
                setModelSuggestions(modelSuggestionData.suggestions);
                setShowTrainPanel(true);
              } catch (error) {
                console.error('Error fetching model suggestions:', error);
              }
            }
            setRunning(false);
          } else if (status.status === 'failed') {
            clearInterval(taskStatusInterval);
            setRunning(false);
          }
        } catch (error) {
          console.error('Error checking task status:', error);
          clearInterval(taskStatusInterval);
          setRunning(false);
        }
      }, 1000);
    } catch (error) {
      console.error('Error running pipeline:', error);
      toast.error('Failed to start pipeline execution');
      setRunning(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'badge-success';
    if (confidence >= 0.6) return 'badge-primary';
    return 'badge-warning';
  };

  if (loading) {
    return (
      <div className="section-card-elevated p-8 space-y-4">
        <div className="flex items-center space-x-4 mb-6">
          <div className="w-14 h-14 bg-surface-container-highest rounded-xl animate-pulse"></div>
          <div className="space-y-2">
            <div className="h-6 bg-surface-container-highest rounded w-48 animate-pulse"></div>
            <div className="h-4 bg-surface-container-high rounded w-32 animate-pulse"></div>
          </div>
        </div>
        <div className="space-y-3">
          <div className="h-24 bg-surface-container-low rounded-xl border border-outline-variant animate-pulse"></div>
          <div className="h-24 bg-surface-container-low rounded-xl border border-outline-variant animate-pulse"></div>
          <div className="h-24 bg-surface-container-low rounded-xl border border-outline-variant animate-pulse"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Pipeline Header Card */}
      <div className="section-card-elevated p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center space-x-4">
            <div className="w-14 h-14 bg-gradient-to-br from-primary-400 to-primary-500 rounded-xl flex items-center justify-center shadow-glow-blue">
              <Workflow className="text-white" size={26} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-on-surface">
                Data Processing Pipeline
              </h2>
              <p className="text-sm text-on-surface-variant mt-0.5">
                <span className="font-medium text-primary-500">{suggestions.length} steps</span> suggested • <span className="font-medium text-success-500">{selectedSteps.size} selected</span>
              </p>
            </div>
          </div>
          <button
            onClick={handleRunPipeline}
            disabled={running || selectedSteps.size === 0}
            className="btn-success flex items-center shadow-lg shadow-success-500/20"
          >
            {running ? (
              <>
                <Loader className="animate-spin mr-2" size={18} />
                Processing...
              </>
            ) : (
              <>
                <Play size={18} className="mr-2" />
                Run Pipeline
              </>
            )}
          </button>
        </div>
      </div>

      {/* Pipeline Steps */}
      <div className="space-y-3">
        {suggestions.map((step, index) => (
          <div
            key={step.id}
            className={`section-card transition-all duration-200 overflow-hidden hover:-translate-y-0.5 hover:shadow-md ${
              selectedSteps.has(step.id)
                ? 'border-primary-200 ring-2 ring-primary-100 shadow-lg shadow-primary-500/5'
                : 'border-outline-variant opacity-80 hover:opacity-100 hover:border-outline-variant'
            }`}
          >
            <div className="p-5">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4 flex-1">
                  {/* Checkbox */}
                  <button
                    onClick={() => toggleStep(step.id)}
                    className="mt-0.5 flex-shrink-0 transition-transform hover:scale-110"
                  >
                    {selectedSteps.has(step.id) ? (
                      <CheckCircle className="text-primary-500" size={24} />
                    ) : (
                      <Circle className="text-outline-variant" size={24} />
                    )}
                  </button>
                  
                  {/* Step Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2 mb-2">
                      <span className="text-xs font-medium text-on-surface-variant bg-surface-container-high px-2 py-0.5 rounded">
                        Step {index + 1}
                      </span>
                      <span className="badge badge-primary font-semibold">
                        {step.type.toUpperCase()}
                      </span>
                      <span className={`badge border ${getConfidenceColor(step.confidence)}`}>
                        <Percent size={12} className="mr-1" />
                        {(step.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                    
                    <p className="text-on-surface-variant leading-relaxed">{step.rationale}</p>
                    
                    {step.target_columns && step.target_columns.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-2">
                        <span className="text-xs font-medium text-on-surface-variant">Columns:</span>
                        {step.target_columns.slice(0, 5).map((col, i) => (
                          <span key={i} className="text-xs bg-surface-container-high text-on-surface-variant px-2 py-0.5 rounded">
                            {col}
                          </span>
                        ))}
                        {step.target_columns.length > 5 && (
                          <span className="text-xs text-on-surface-variant">
                            +{step.target_columns.length - 5} more
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center space-x-1 ml-4">
                  <button
                    onClick={() => handlePreview(step)}
                    className="btn-icon"
                    aria-label="Preview step output"
                    title="Preview output"
                  >
                    <Eye size={18} />
                  </button>
                  <button
                    onClick={() => toggleExpand(step.id)}
                    className="btn-icon"
                    aria-label="View parameters"
                    title="View parameters"
                  >
                    {expandedStep === step.id ? (
                      <ChevronUp size={18} />
                    ) : (
                      <ChevronDown size={18} />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Expanded Parameters */}
            {expandedStep === step.id && (
              <div className="px-5 pb-5 border-t border-outline-variant pt-4 bg-surface-container-low">
                <div className="flex items-center space-x-2 mb-3">
                  <Settings size={16} className="text-on-surface-variant" />
                  <h4 className="font-medium text-sm text-on-surface-variant">Parameters</h4>
                </div>
                <pre className="bg-surface-container-lowest text-slate-100 p-4 rounded-xl text-xs overflow-x-auto font-mono">
                  {JSON.stringify(step.params, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Preview Console */}
      {previewStep && (
        <div className="animate-in">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-on-surface">
              Preview: {previewStep.type}
            </h3>
            <button
              onClick={() => setPreviewStep(null)}
              className="btn-ghost text-sm"
            >
              Close Preview
            </button>
          </div>
          <Console
            showPreview={true}
            previewLogs={previewStep.console_preview.map((line, idx) => {
              const parts = line.split(' | ');
              return {
                timestamp: new Date().toISOString(),
                level: parts[0]?.split(' ')[0] || 'INFO',
                source: parts[1] || '',
                message: parts[2] || line,
              };
            })}
          />
        </div>
      )}

      {/* Live Console */}
      {taskId && (
        <div className="animate-in">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-on-surface">Pipeline Execution</h3>
            
            {/* Dashboard Button - Show when preprocessing completes */}
            {pipelineCompleted && (
              <button
                onClick={() => navigate(`/dashboard/${taskId}`)}
                className="btn-primary flex items-center"
              >
                <BarChart3 size={18} className="mr-2" />
                View Data Dashboard
              </button>
            )}
          </div>
          <Console taskId={taskId} autoConnect={true} />
        </div>
      )}
      
      {/* Train Panel - Show after preprocessing completes */}
      {showTrainPanel && (
        <div className="mt-6 animate-in">
          <TrainPanel 
            datasetId={datasetId} 
            modelSuggestions={modelSuggestions}
            targetColumn={targetColumn}
          />
        </div>
      )}
    </div>
  );
};

export default PipelineEditor;
