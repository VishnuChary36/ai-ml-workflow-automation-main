import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle, Circle, Eye, Play, Loader, ChevronDown, ChevronUp, BarChart3 } from 'lucide-react';
import Console from '../Console/Console';
import TrainPanel from '../TrainPanel/TrainPanel';
import { getSuggestedPipeline, runPipeline, getSuggestedModels, getTaskStatus } from '../../api/client';

const PipelineEditor = ({ datasetId, targetColumn, onPipelineComplete }) => {
  const navigate = useNavigate();
  const [suggestions, setSuggestions] = useState([]);
  const [selectedSteps, setSelectedSteps] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [expandedStep, setExpandedStep] = useState(null);
  const [previewStep, setPreviewStep] = useState(null);
  const [showTrainPanel, setShowTrainPanel] = useState(false);
  const [modelSuggestions, setModelSuggestions] = useState([]);
  const [pipelineCompleted, setPipelineCompleted] = useState(false);

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
      alert('Please select at least one step');
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
      alert('Failed to start pipeline execution');
      setRunning(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8 flex items-center justify-center">
        <Loader className="animate-spin mr-3" size={24} />
        <span>Loading AI suggestions...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">
            AI-Suggested Pipeline
          </h2>
          <button
            onClick={handleRunPipeline}
            disabled={running || selectedSteps.size === 0}
            className="px-6 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition flex items-center"
          >
            {running ? (
              <>
                <Loader className="animate-spin mr-2" size={18} />
                Running...
              </>
            ) : (
              <>
                <Play size={18} className="mr-2" />
                Run Pipeline
              </>
            )}
          </button>
        </div>

        <div className="space-y-3">
          {suggestions.map((step, index) => (
            <div
              key={step.id}
              className={`border rounded-lg overflow-hidden transition ${
                selectedSteps.has(step.id)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 bg-white'
              }`}
            >
              <div className="p-4 flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1">
                  <button
                    onClick={() => toggleStep(step.id)}
                    className="mt-1 flex-shrink-0"
                  >
                    {selectedSteps.has(step.id) ? (
                      <CheckCircle className="text-blue-600" size={24} />
                    ) : (
                      <Circle className="text-gray-400" size={24} />
                    )}
                  </button>
                  
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <span className="font-mono text-sm text-gray-500">
                        Step {index + 1}
                      </span>
                      <span className="px-2 py-1 bg-gray-200 text-gray-700 rounded text-xs font-semibold">
                        {step.type.toUpperCase()}
                      </span>
                      <span className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-semibold">
                        {(step.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                    
                    <p className="mt-2 text-gray-700">{step.rationale}</p>
                    
                    {step.target_columns && step.target_columns.length > 0 && (
                      <div className="mt-2 text-sm text-gray-600">
                        <span className="font-semibold">Columns:</span>{' '}
                        {step.target_columns.join(', ')}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center space-x-2 ml-4">
                  <button
                    onClick={() => handlePreview(step)}
                    className="p-2 hover:bg-gray-200 rounded text-gray-600 hover:text-gray-800 transition"
                    title="Preview console output"
                  >
                    <Eye size={18} />
                  </button>
                  <button
                    onClick={() => toggleExpand(step.id)}
                    className="p-2 hover:bg-gray-200 rounded text-gray-600 hover:text-gray-800 transition"
                  >
                    {expandedStep === step.id ? (
                      <ChevronUp size={18} />
                    ) : (
                      <ChevronDown size={18} />
                    )}
                  </button>
                </div>
              </div>

              {expandedStep === step.id && (
                <div className="px-4 pb-4 border-t border-gray-200 pt-4 bg-gray-50">
                  <h4 className="font-semibold text-sm mb-2">Parameters:</h4>
                  <pre className="bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto">
                    {JSON.stringify(step.params, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Preview Console */}
      {previewStep && (
        <div>
          <h3 className="text-lg font-semibold mb-3">
            Preview: {previewStep.type}
          </h3>
          <Console
            showPreview={true}
            previewLogs={previewStep.console_preview.map((line, idx) => {
              // Parse preview line (simplified)
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
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold">Pipeline Execution Logs</h3>
            
            {/* Dashboard Button - Show when preprocessing completes */}
            {pipelineCompleted && (
              <button
                onClick={() => navigate(`/dashboard/${taskId}`)}
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 transition-all shadow-md hover:shadow-lg"
              >
                <BarChart3 size={18} />
                View Data Dashboard
              </button>
            )}
          </div>
          <Console taskId={taskId} autoConnect={true} />
        </div>
      )}
      
      {/* Train Panel - Show after preprocessing completes */}
      {showTrainPanel && (
        <div className="mt-6">
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
