import React, { useState, useEffect, useRef } from 'react';
import { 
  ArrowLeft, 
  Download, 
  Rocket, 
  Server, 
  Cloud, 
  Container, 
  CheckCircle, 
  AlertTriangle,
  Copy,
  Terminal,
  FileCode,
  Package,
  Activity,
  Clock,
  Shield,
  Globe,
  Zap,
  ExternalLink,
  Info,
  ChevronDown,
  ChevronUp,
  Key,
  Lock,
  RefreshCw,
  BarChart3,
  AlertCircle,
  Check,
  Code
} from 'lucide-react';
import { deployModel, getDeployment, getModelDeployments, downloadDeploymentPackage, getTaskStatus, getLogs } from '../../api/client';

const PLATFORM_INFO = {
  local: {
    name: 'Local Server',
    icon: Server,
    description: 'Deploy to local FastAPI server for development and testing',
    color: 'blue',
    features: ['Quick setup', 'Debug mode', 'Hot reload']
  },
  docker: {
    name: 'Docker Container',
    icon: Container,
    description: 'Containerized deployment with Docker for production',
    color: 'cyan',
    features: ['Portable', 'Isolated', 'Reproducible']
  },
  cloud: {
    name: 'Cloud Platform',
    icon: Cloud,
    description: 'Deploy to AWS, GCP, or Azure cloud services',
    color: 'purple',
    features: ['Auto-scaling', 'HTTPS', 'Global CDN']
  }
};

const CLOUD_PROVIDERS = [
  {
    id: 'gcp-cloudrun',
    name: 'Google Cloud Run',
    icon: '🚀',
    description: 'Simple container deployment with automatic HTTPS and autoscaling',
    difficulty: 'Easy',
    cost: 'Pay-per-use',
    commands: [
      'gcloud auth configure-docker',
      'docker build -t gcr.io/PROJECT_ID/ml-model .',
      'docker push gcr.io/PROJECT_ID/ml-model',
      'gcloud run deploy ml-model --image gcr.io/PROJECT_ID/ml-model --platform managed --allow-unauthenticated'
    ]
  },
  {
    id: 'aws-fargate',
    name: 'AWS Fargate / ECS',
    icon: '☁️',
    description: 'Serverless containers on AWS with load balancing',
    difficulty: 'Medium',
    cost: 'Pay-per-use',
    commands: [
      'aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com',
      'docker build -t ml-model .',
      'docker tag ml-model:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ml-model:latest',
      'docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ml-model:latest',
      '# Then create ECS service via AWS Console or CLI'
    ]
  },
  {
    id: 'azure-aci',
    name: 'Azure Container Instances',
    icon: '🔷',
    description: 'Simple container deployment on Azure',
    difficulty: 'Easy',
    cost: 'Pay-per-use',
    commands: [
      'az acr login --name myregistry',
      'docker build -t myregistry.azurecr.io/ml-model .',
      'docker push myregistry.azurecr.io/ml-model',
      'az container create --resource-group myResourceGroup --name ml-model --image myregistry.azurecr.io/ml-model --ports 8080'
    ]
  },
  {
    id: 'render',
    name: 'Render',
    icon: '🎯',
    description: 'Simple UI-driven deployment, great for quick projects',
    difficulty: 'Very Easy',
    cost: 'Free tier available',
    commands: [
      '# 1. Push your code to GitHub',
      '# 2. Connect repository in Render dashboard',
      '# 3. Select "Docker" environment',
      '# 4. Click Deploy - Render handles the rest!'
    ]
  },
  {
    id: 'railway',
    name: 'Railway',
    icon: '🚂',
    description: 'Developer-friendly platform with instant deployments',
    difficulty: 'Very Easy',
    cost: 'Free tier available',
    commands: [
      '# Install Railway CLI',
      'npm install -g @railway/cli',
      'railway login',
      'railway init',
      'railway up'
    ]
  },
  {
    id: 'huggingface',
    name: 'Hugging Face Spaces',
    icon: '🤗',
    description: 'Free hosting for ML demos with Gradio/Streamlit',
    difficulty: 'Very Easy',
    cost: 'Free',
    commands: [
      '# 1. Create new Space at huggingface.co/spaces',
      '# 2. Select Docker SDK',
      '# 3. Push your code with git:',
      'git remote add hf https://huggingface.co/spaces/USERNAME/SPACE_NAME',
      'git push hf main'
    ]
  }
];

const PRODUCTION_CHECKLIST = [
  { id: 'model', text: 'Model file exported and tested locally', category: 'Model' },
  { id: 'api', text: 'API implements preprocessing & returns correct JSON', category: 'API' },
  { id: 'docker', text: 'Docker image builds and runs locally', category: 'Docker' },
  { id: 'deploy', text: 'Deploy to chosen host → obtain https://... URL', category: 'Deploy' },
  { id: 'frontend', text: 'Frontend fetch() points to that URL', category: 'Frontend' },
  { id: 'cors', text: 'CORS configured for your domain', category: 'Security' },
  { id: 'auth', text: 'Add API key/auth & enable HTTPS', category: 'Security' },
  { id: 'health', text: 'Add health endpoint (/health)', category: 'Monitoring' },
  { id: 'logging', text: 'Add logging & basic monitoring', category: 'Monitoring' },
  { id: 'alerts', text: 'Set up alerts for errors', category: 'Monitoring' }
];

const Deployment = ({ modelId, modelName, onBack }) => {
  const [selectedPlatform, setSelectedPlatform] = useState(null);
  const [deploying, setDeploying] = useState(false);
  const [deploymentResult, setDeploymentResult] = useState(null);
  const [error, setError] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [existingDeployments, setExistingDeployments] = useState([]);
  const [loadingDeployments, setLoadingDeployments] = useState(true);
  const [logs, setLogs] = useState([]);
  const [expandedProvider, setExpandedProvider] = useState(null);
  const [expandedSection, setExpandedSection] = useState('platform');
  const [checklist, setChecklist] = useState({});
  const [copied, setCopied] = useState(null);
  const logsEndRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    fetchExistingDeployments();
  }, [modelId]);

  // WebSocket connection for real-time logs
  useEffect(() => {
    if (taskId && taskStatus?.toUpperCase() === 'RUNNING') {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = window.location.hostname;
      const wsPort = '8000';
      const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws/logs?task_id=${taskId}`;
      
      console.log('Connecting to WebSocket:', wsUrl);
      
      try {
        wsRef.current = new WebSocket(wsUrl);
        
        wsRef.current.onopen = () => {
          console.log('WebSocket connected for deployment logs');
        };
        
        wsRef.current.onmessage = (event) => {
          try {
            const logData = JSON.parse(event.data);
            setLogs(prev => [...prev, {
              timestamp: logData.ts || new Date().toISOString(),
              level: logData.level || 'INFO',
              message: logData.message,
              source: logData.source
            }]);
          } catch (e) {
            console.log('Log message:', event.data);
          }
        };
        
        wsRef.current.onerror = (err) => {
          console.error('WebSocket error:', err);
        };
        
        wsRef.current.onclose = () => {
          console.log('WebSocket closed');
        };
      } catch (err) {
        console.error('WebSocket connection failed:', err);
      }
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [taskId, taskStatus]);

  // Auto-scroll logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  useEffect(() => {
    let interval;
    const normalizedStatus = taskStatus?.toUpperCase();
    if (taskId && normalizedStatus !== 'COMPLETED' && normalizedStatus !== 'FAILED') {
      interval = setInterval(async () => {
        try {
          const status = await getTaskStatus(taskId);
          const currentStatus = status.status?.toUpperCase();
          setTaskStatus(currentStatus);
          
          // Also fetch logs via HTTP
          try {
            const logsData = await getLogs(taskId);
            if (logsData.logs && logsData.logs.length > 0) {
              setLogs(logsData.logs.map(log => ({
                timestamp: log.ts || log.timestamp || new Date().toISOString(),
                level: log.level || 'INFO',
                message: log.message,
                source: log.source
              })));
            }
          } catch (logErr) {
            // Logs may not be available yet
          }
          
          if (currentStatus === 'COMPLETED') {
            setDeploymentResult(status.result);
            setDeploying(false);
            fetchExistingDeployments();
          } else if (currentStatus === 'FAILED') {
            setError(status.error || 'Deployment failed');
            setDeploying(false);
          }
        } catch (err) {
          console.error('Error polling task status:', err);
        }
      }, 1000); // Poll every second
    }
    return () => clearInterval(interval);
  }, [taskId, taskStatus]);

  const fetchExistingDeployments = async () => {
    try {
      setLoadingDeployments(true);
      const data = await getModelDeployments(modelId);
      setExistingDeployments(data.deployments || []);
    } catch (err) {
      console.error('Error fetching deployments:', err);
    } finally {
      setLoadingDeployments(false);
    }
  };

  const handleDeploy = async () => {
    if (!selectedPlatform) return;
    
    try {
      setDeploying(true);
      setError(null);
      setDeploymentResult(null);
      setLogs([]);
      
      const result = await deployModel(modelId, selectedPlatform);
      setTaskId(result.task_id);
      setTaskStatus('RUNNING');
      
      // Add initial log entry
      setLogs([{
        timestamp: new Date().toISOString(),
        level: 'INFO',
        message: `🚀 Deployment started to ${PLATFORM_INFO[selectedPlatform].name}`,
        source: 'deployment.init'
      }]);
    } catch (err) {
      setError(err.response?.data?.detail || 'Deployment failed');
      setDeploying(false);
    }
  };

  const handleDownload = async (deploymentId) => {
    try {
      await downloadDeploymentPackage(deploymentId);
    } catch (err) {
      console.error('Error downloading package:', err);
      alert('Failed to download deployment package');
    }
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const toggleChecklist = (id) => {
    setChecklist(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const completedItems = Object.values(checklist).filter(Boolean).length;
  const checklistProgress = (completedItems / PRODUCTION_CHECKLIST.length) * 100;

  return (
    <div className="page-container">
      <div className="content-wrapper">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-br from-success-500 to-success-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <Rocket className="text-white" size={20} />
              </div>
              <h1 className="text-2xl font-bold text-on-surface">Deploy Model</h1>
            </div>
            <p className="text-on-surface-variant ml-13 mt-1">
              Package and deploy <span className="font-semibold text-primary-500">{modelName || modelId}</span> to production
            </p>
          </div>
          <button
            onClick={onBack}
            className="btn-secondary"
          >
            <ArrowLeft size={18} />
            Back
          </button>
        </div>

        {/* High-Level Overview Banner */}
        <div className="card p-6 mb-6" style={{ background: 'rgba(16, 185, 129, 0.06)', borderColor: 'rgba(16, 185, 129, 0.2)' }}>
          <div className="flex items-start">
            <Info size={24} className="text-success-500 mr-3 flex-shrink-0 mt-0.5" />
            <div>
              <h2 className="text-xl font-bold text-on-surface mb-2">Deployment Flow Overview</h2>
              <p className="text-on-surface-variant">
                Package your model inside a web service (API endpoint), deploy to a public host (real URL + HTTPS), 
                and connect your website to call the service from frontend.
              </p>
            </div>
          </div>
        </div>

        {/* Deployment Pipeline Overview */}
        <div className="card p-6 mb-6">
          <h2 className="text-lg font-semibold text-on-surface mb-4">Deployment Pipeline</h2>
          <div className="flex items-center justify-between">
            {['Export Model', 'Create API', 'Containerize', 'Live Endpoint', 'Frontend Ready'].map((step, idx) => {
              // Calculate completion based on deployment result - ALL steps complete when deployed
              const isCompleted = deploymentResult ? true : (idx <= 2 && deploying);
              const isActive = deploying && !deploymentResult && idx === 3;
              
              return (
                <React.Fragment key={step}>
                  <div className="flex flex-col items-center">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold ${
                      isCompleted ? 'bg-success-500 text-white' : 
                      isActive ? 'bg-primary-500 text-white animate-pulse' : 
                      'bg-surface-container-high text-on-surface-variant border border-outline-variant'
                    }`}>
                      {isCompleted ? <CheckCircle size={20} /> : idx + 1}
                    </div>
                    <span className="text-xs mt-2 text-on-surface-variant text-center max-w-[80px]">{step}</span>
                  </div>
                  {idx < 4 && (
                    <div className={`flex-1 h-1 mx-2 rounded-full ${
                      deploymentResult ? 'bg-success-500' : (isCompleted && idx < 3 ? 'bg-success-500' : 'bg-surface-container-highest')
                    }`} />
                  )}
                </React.Fragment>
              );
            })}
          </div>
          {deploymentResult && (
            <div className="mt-4 p-3 rounded-xl" style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <p className="text-success-500 text-sm">
                🎉 <strong>All steps complete!</strong> Your model is live and ready to accept predictions. Use the code snippets below to integrate with your application.
              </p>
            </div>
          )}
        </div>

        {/* Platform Selection */}
        <div className="section-card p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'platform' ? null : 'platform')}
          >
            <h2 className="text-lg font-semibold text-on-surface flex items-center">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-100 to-primary-50 rounded-lg flex items-center justify-center mr-3">
                <Server size={16} className="text-primary-500" />
              </div>
              Step 1: Select Deployment Platform
            </h2>
            {expandedSection === 'platform' ? <ChevronUp size={20} className="text-on-surface-variant" /> : <ChevronDown size={20} className="text-on-surface-variant" />}
          </div>
          
          {expandedSection === 'platform' && (
            <div className="mt-5">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(PLATFORM_INFO).map(([key, platform]) => {
                  const Icon = platform.icon;
                  const isSelected = selectedPlatform === key;
                  
                  return (
                    <button
                      key={key}
                      onClick={(e) => { e.stopPropagation(); setSelectedPlatform(key); }}
                      disabled={deploying}
                      className={`deploy-option ${isSelected ? 'deploy-option-selected' : ''} ${deploying ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className="flex items-center mb-3">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${isSelected ? 'bg-primary-100' : 'bg-surface-container-high'}`}>
                          <Icon size={22} className={isSelected ? 'text-primary-500' : 'text-on-surface-variant'} />
                        </div>
                        <span className={`ml-3 font-semibold ${isSelected ? 'text-primary-500' : 'text-on-surface'}`}>
                          {platform.name}
                        </span>
                      </div>
                      <p className="text-sm text-on-surface-variant mb-3">{platform.description}</p>
                      <div className="flex flex-wrap gap-1">
                        {platform.features.map((f, i) => (
                          <span key={i} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium" style={{ background: 'rgba(77, 142, 255, 0.12)', color: '#adc6ff' }}>
                            {f}
                          </span>
                        ))}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Package Contents */}
        <div className="section-card p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'package' ? null : 'package')}
          >
            <h2 className="text-lg font-semibold text-on-surface flex items-center">
              <div className="w-8 h-8 bg-surface-container-high rounded-lg flex items-center justify-center mr-3">
                <Package size={16} className="text-on-surface-variant" />
              </div>
              Deployment Package Contents
            </h2>
            {expandedSection === 'package' ? <ChevronUp size={20} className="text-on-surface-variant" /> : <ChevronDown size={20} className="text-on-surface-variant" />}
          </div>
          
          {expandedSection === 'package' && (
            <div className="mt-5 grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { name: 'Trained Model', desc: 'model.joblib', icon: FileCode, tip: 'Your trained ML model' },
                { name: 'Encoders', desc: 'encoders.joblib', icon: Activity, tip: 'Label encoders for categorical features' },
                { name: 'Inference API', desc: 'app.py (FastAPI)', icon: Server, tip: 'Production-ready REST API' },
                { name: 'Docker Config', desc: 'Dockerfile', icon: Container, tip: 'Container configuration' },
                { name: 'Metadata', desc: 'metadata.json', icon: FileCode, tip: 'Model info and feature specs' },
                { name: 'Preprocessing', desc: 'preprocessing.json', icon: Activity, tip: 'Data transformation config' },
                { name: 'Docker Compose', desc: 'docker-compose.yml', icon: Container, tip: 'Multi-container orchestration' },
                { name: 'Requirements', desc: 'requirements.txt', icon: FileCode, tip: 'Python dependencies' },
              ].map((item, idx) => (
                <div key={idx} className="p-4 bg-surface-container-low rounded-xl hover:bg-surface-container-high transition group border border-outline-variant">
                  <item.icon size={18} className="text-primary-500 mb-2" />
                  <div className="font-medium text-on-surface text-sm">{item.name}</div>
                  <div className="text-xs text-on-surface-variant">{item.desc}</div>
                  <div className="text-xs text-primary-500 opacity-0 group-hover:opacity-100 transition mt-1">{item.tip}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Deploy Button */}
        <div className="card p-6 mb-6" style={{ background: 'rgba(16, 185, 129, 0.06)', borderColor: 'rgba(16, 185, 129, 0.2)' }}>
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-on-surface mb-1">Create Deployment Package</h2>
              <p className="text-on-surface-variant">
                {selectedPlatform 
                  ? `Generate package for ${PLATFORM_INFO[selectedPlatform].name}` 
                  : 'Select a platform above to get started'}
              </p>
            </div>
            <button
              onClick={handleDeploy}
              disabled={!selectedPlatform || deploying}
              className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition ${
                selectedPlatform && !deploying
                  ? 'btn-success'
                  : 'bg-surface-container-high text-on-surface-variant cursor-not-allowed border border-outline-variant'
              }`}
            >
              {deploying ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Creating Package...
                </>
              ) : (
                <>
                  <Rocket size={18} />
                  Generate Package
                </>
              )}
            </button>
          </div>
        </div>

        {/* Task Progress with Real-Time Logs */}
        {taskId && taskStatus?.toUpperCase() === 'RUNNING' && (
          <div className="card p-6 mb-6" style={{ background: 'rgba(77, 142, 255, 0.06)', borderColor: 'rgba(77, 142, 255, 0.2)' }}>
            <div className="flex items-center mb-4">
              <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mr-3" />
              <div>
                <h3 className="font-semibold text-primary-500">Creating Deployment Package</h3>
                <p className="text-on-surface-variant text-sm">Packaging model with all artifacts...</p>
              </div>
            </div>
            
            {/* Real-Time Log Console */}
            <div className="console-container rounded-xl p-4 max-h-64 overflow-y-auto text-sm">
              {logs.length === 0 ? (
                <div className="text-on-surface-variant">Waiting for deployment logs...</div>
              ) : (
                logs.map((log, idx) => (
                  <div key={idx} className={`mb-1 ${
                    log.level === 'ERROR' ? 'text-error-400' :
                    log.level === 'WARNING' ? 'text-warning-500' :
                    'text-success-500'
                  }`}>
                    <span className="text-on-surface-variant">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
                    {log.message}
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="card p-6 mb-6" style={{ background: 'rgba(239, 68, 68, 0.06)', borderColor: 'rgba(239, 68, 68, 0.2)' }}>
            <div className="flex items-center">
              <AlertTriangle size={24} className="text-error-400 mr-3" />
              <div>
                <h3 className="font-semibold text-error-400">Deployment Failed</h3>
                <p className="text-on-surface-variant text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Deployment Result */}
        {deploymentResult && (
          <div className="card p-6 mb-6" style={{ background: 'rgba(16, 185, 129, 0.06)', borderColor: 'rgba(16, 185, 129, 0.2)' }}>
            <div className="flex items-start mb-4">
              <CheckCircle size={24} className="text-success-500 mr-3 mt-0.5" />
              <div>
                <h3 className="font-semibold text-success-500">🎉 Model Deployed & Ready!</h3>
                <p className="text-on-surface-variant text-sm">Your model is LIVE and ready to serve predictions</p>
              </div>
            </div>
            
            <div className="space-y-4">
              {/* LIVE Prediction Endpoint */}
              <div className="bg-surface-container rounded-xl p-4 border-2 border-success-300">
                <div className="flex items-center mb-2">
                  <span className="inline-block w-3 h-3 bg-success-500 rounded-full animate-pulse mr-2"></span>
                  <span className="text-sm font-semibold text-success-500">LIVE Prediction Endpoint</span>
                </div>
                <div className="flex items-center justify-between bg-surface-container-low rounded-lg p-2">
                  <code className="text-sm text-on-surface">
                    {deploymentResult.live_prediction_url || deploymentResult.deployment_url}
                  </code>
                  <button 
                    onClick={() => copyToClipboard(deploymentResult.live_prediction_url || deploymentResult.deployment_url, 'live-url')}
                    className="ml-2 text-on-surface-variant hover:text-primary-500 flex items-center"
                  >
                    {copied === 'live-url' ? <Check size={16} className="text-success-500" /> : <Copy size={16} />}
                  </button>
                </div>
                <p className="text-xs text-on-surface-variant mt-2">
                  ✅ This endpoint is ready to use NOW - no additional setup required!
                </p>
              </div>
              
              {/* Frontend Integration Code */}
              <div className="bg-surface-container-lowest rounded-lg p-4 text-white">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <Code size={16} className="mr-2 text-blue-400" />
                    <span className="text-sm font-semibold text-white">Frontend Integration (JavaScript)</span>
                  </div>
                  <button 
                    onClick={() => copyToClipboard(`// Make predictions from your frontend
const predict = async (features) => {
  const response = await fetch("${deploymentResult.live_prediction_url || deploymentResult.deployment_url}", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features })
  });
  return await response.json();
};

// Example usage:
const result = await predict({
  // Add your feature values here
  feature1: "value1",
  feature2: 123
});
console.log("Prediction:", result.prediction);`, 'js-code')}
                    className="text-on-surface-variant hover:text-on-surface"
                  >
                    {copied === 'js-code' ? <Check size={14} className="text-success-400" /> : <Copy size={14} />}
                  </button>
                </div>
                <pre className="text-sm text-success-400 overflow-x-auto">
{`// Make predictions from your frontend
const predict = async (features) => {
  const response = await fetch("${deploymentResult.live_prediction_url || deploymentResult.deployment_url}", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features })
  });
  return await response.json();
};

// Example usage:
const result = await predict({
  // Add your feature values here
  feature1: "value1",
  feature2: 123
});
console.log("Prediction:", result.prediction);`}
                </pre>
              </div>
              
              {/* Python Integration Code */}
              <div className="bg-surface-container-lowest rounded-lg p-4 text-white">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <Code size={16} className="mr-2 text-warning-500" />
                    <span className="text-sm font-semibold text-white">Python Integration</span>
                  </div>
                  <button 
                    onClick={() => copyToClipboard(`import requests

# Make a prediction
response = requests.post(
    "${deploymentResult.live_prediction_url || deploymentResult.deployment_url}",
    json={"features": {"feature1": "value1", "feature2": 123}}
)
result = response.json()
print("Prediction:", result["prediction"])`, 'py-code')}
                    className="text-on-surface-variant hover:text-on-surface"
                  >
                    {copied === 'py-code' ? <Check size={14} className="text-success-400" /> : <Copy size={14} />}
                  </button>
                </div>
                <pre className="text-sm text-warning-500 overflow-x-auto">
{`import requests

# Make a prediction
response = requests.post(
    "${deploymentResult.live_prediction_url || deploymentResult.deployment_url}",
    json={"features": {"feature1": "value1", "feature2": 123}}
)
result = response.json()
print("Prediction:", result["prediction"])`}
                </pre>
              </div>
              
              {/* cURL Example */}
              <div className="bg-surface-container-lowest rounded-lg p-4 text-white">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <Terminal size={16} className="mr-2 text-on-surface-variant" />
                    <span className="text-sm font-semibold text-white">cURL / Command Line</span>
                  </div>
                  <button 
                    onClick={() => copyToClipboard(`curl -X POST "${deploymentResult.live_prediction_url || deploymentResult.deployment_url}" \\
  -H "Content-Type: application/json" \\
  -d '{"features": {"feature1": "value1", "feature2": 123}}'`, 'curl-code')}
                    className="text-on-surface-variant hover:text-on-surface"
                  >
                    {copied === 'curl-code' ? <Check size={14} className="text-success-400" /> : <Copy size={14} />}
                  </button>
                </div>
                <pre className="text-sm text-accent-cyan overflow-x-auto">
{`curl -X POST "${deploymentResult.live_prediction_url || deploymentResult.deployment_url}" \\
  -H "Content-Type: application/json" \\
  -d '{"features": {"feature1": "value1", "feature2": 123}}'`}
                </pre>
              </div>
              
              {/* Get Feature Info Button */}
              <div className="rounded-lg p-4" style={{ background: 'rgba(77, 142, 255, 0.06)', border: '1px solid rgba(77, 142, 255, 0.15)' }}>
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold text-primary-500">Need exact feature names?</h4>
                    <p className="text-on-surface-variant text-sm">Get the full API documentation including all required features</p>
                  </div>
                  <a 
                    href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/predict/${modelId}/info`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-4 py-2 bg-primary-400 text-white rounded-lg hover:bg-primary-500 transition flex items-center"
                  >
                    <ExternalLink size={16} className="mr-2" />
                    View API Docs
                  </a>
                </div>
              </div>
              
              <hr className="border-outline-variant" />
              
              {/* Standalone Package Download */}
              <div className="bg-surface-container-low rounded-xl p-4 border border-outline-variant">
                <h4 className="font-semibold text-on-surface mb-2">📦 Standalone Deployment Package</h4>
                <p className="text-on-surface-variant text-sm mb-3">
                  Download the complete package to deploy as a separate service on your own infrastructure.
                </p>
                
                {/* Download Button */}
                <button
                  onClick={() => handleDownload(deploymentResult.deployment_id)}
                  className="w-full btn-primary flex items-center justify-center"
                >
                  <Download size={18} className="mr-2" />
                  Download Deployment Package (.zip)
                </button>
                
                {/* Package Files */}
                {deploymentResult.files && deploymentResult.files.length > 0 && (
                  <div className="mt-3">
                    <div className="text-sm font-medium text-on-surface mb-2">Package Contents:</div>
                    <div className="flex flex-wrap gap-2">
                      {deploymentResult.files.map((file, idx) => (
                        <span key={idx} className="badge badge-primary text-xs">
                          {file}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Cloud Deployment Options */}
        <div className="card p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'cloud' ? null : 'cloud')}
          >
            <h2 className="text-lg font-semibold text-on-surface">
              <Cloud size={20} className="inline mr-2 text-primary-500" />
              Step 2: Deploy to Cloud (Choose Provider)
            </h2>
            {expandedSection === 'cloud' ? <ChevronUp size={20} className="text-on-surface-variant" /> : <ChevronDown size={20} className="text-on-surface-variant" />}
          </div>
          
          {expandedSection === 'cloud' && (
            <div className="mt-4 space-y-3">
              <p className="text-on-surface-variant text-sm mb-4">
                After generating your package, deploy to any of these hosting services to get a public HTTPS URL:
              </p>
              
              {CLOUD_PROVIDERS.map((provider) => (
                <div 
                  key={provider.id}
                  className="border border-outline-variant rounded-xl overflow-hidden bg-surface-container"
                >
                  <button
                    onClick={() => setExpandedProvider(expandedProvider === provider.id ? null : provider.id)}
                    className="w-full p-4 flex items-center justify-between hover:bg-surface-container-low transition"
                  >
                    <div className="flex items-center">
                      <span className="text-2xl mr-3">{provider.icon}</span>
                      <div className="text-left">
                        <div className="font-semibold text-on-surface">{provider.name}</div>
                        <div className="text-sm text-on-surface-variant">{provider.description}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`px-2 py-1 text-xs rounded-lg font-medium ${                        provider.difficulty === 'Very Easy' ? 'text-success-500' :
                        provider.difficulty === 'Easy' ? 'text-primary-500' :
                        'text-warning-500'
                      }`} style={{ background: provider.difficulty === 'Very Easy' ? 'rgba(16,185,129,0.1)' : provider.difficulty === 'Easy' ? 'rgba(77,142,255,0.1)' : 'rgba(245,158,11,0.1)' }}>
                        {provider.difficulty}
                      </span>
                      <span className="px-2 py-1 bg-surface-container-high text-on-surface-variant text-xs rounded-lg font-medium">
                        {provider.cost}
                      </span>
                      {expandedProvider === provider.id ? <ChevronUp size={18} className="text-on-surface-variant" /> : <ChevronDown size={18} className="text-on-surface-variant" />}
                    </div>
                  </button>
                  
                  {expandedProvider === provider.id && (
                    <div className="px-4 pb-4 bg-surface-container-low">
                      <div className="console-container rounded-xl p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-on-surface-variant">Deployment Commands</span>
                          <button 
                            onClick={() => copyToClipboard(provider.commands.join('\n'), provider.id)}
                            className="text-on-surface-variant hover:text-primary-500"
                          >
                            {copied === provider.id ? <Check size={14} className="text-success-500" /> : <Copy size={14} />}
                          </button>
                        </div>
                        <pre className="text-sm text-success-500 overflow-x-auto whitespace-pre-wrap">
                          {provider.commands.join('\n')}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Security & CORS */}
        <div className="card p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'security' ? null : 'security')}
          >
            <h2 className="text-lg font-semibold text-on-surface">
              <Shield size={20} className="inline mr-2 text-warning-500" />
              Security & Production Considerations
            </h2>
            {expandedSection === 'security' ? <ChevronUp size={20} className="text-on-surface-variant" /> : <ChevronDown size={20} className="text-on-surface-variant" />}
          </div>
          
          {expandedSection === 'security' && (
            <div className="mt-4 space-y-4">
              {/* CORS Configuration */}
              <div className="p-4 rounded-xl" style={{ background: 'rgba(77, 142, 255, 0.06)', border: '1px solid rgba(77, 142, 255, 0.15)' }}>
                <div className="flex items-center mb-2">
                  <Globe size={18} className="text-primary-500 mr-2" />
                  <h3 className="font-semibold text-primary-500">CORS Configuration</h3>
                </div>
                <p className="text-sm text-on-surface-variant mb-3">
                  Enable CORS so browser requests from your website domain succeed:
                </p>
                <div className="console-container rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-on-surface-variant">FastAPI CORS Middleware</span>
                    <button 
                      onClick={() => copyToClipboard(`from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-website.com"],  # restrict to your site
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)`, 'cors')}
                      className="text-on-surface-variant hover:text-primary-500"
                    >
                      {copied === 'cors' ? <Check size={14} className="text-success-500" /> : <Copy size={14} />}
                    </button>
                  </div>
                  <pre className="text-sm text-success-500 overflow-x-auto">
{`from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-website.com"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)`}
                  </pre>
                </div>
              </div>

              {/* API Key Security */}
              <div className="p-4 rounded-xl" style={{ background: 'rgba(245, 158, 11, 0.06)', border: '1px solid rgba(245, 158, 11, 0.15)' }}>
                <div className="flex items-center mb-2">
                  <Key size={18} className="text-warning-500 mr-2" />
                  <h3 className="font-semibold text-warning-500">API Key Authentication</h3>
                </div>
                <p className="text-sm text-on-surface-variant mb-3">
                  Add API key check to prevent abuse:
                </p>
                <div className="console-container rounded-xl p-4">
                  <pre className="text-sm text-success-500 overflow-x-auto">
{`from fastapi import Header, HTTPException

API_KEYS = {"your-secret-key-here"}

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.post("/predict")
async def predict(request: PredictRequest, api_key: str = Depends(verify_api_key)):
    # Your prediction logic
    pass`}
                  </pre>
                </div>
              </div>

              {/* Important Considerations Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-surface-container-low rounded-xl border border-outline-variant">
                  <div className="flex items-center mb-2">
                    <Lock size={18} className="text-primary-500 mr-2" />
                    <h4 className="font-medium text-on-surface">HTTPS & Domain</h4>
                  </div>
                  <p className="text-sm text-on-surface-variant">
                    Cloud hosts provide HTTPS by default. Point your domain to the service for custom URLs.
                  </p>
                </div>
                
                <div className="p-4 bg-surface-container-low rounded-xl border border-outline-variant">
                  <div className="flex items-center mb-2">
                    <RefreshCw size={18} className="text-success-500 mr-2" />
                    <h4 className="font-medium text-on-surface">Preprocessing Parity</h4>
                  </div>
                  <p className="text-sm text-on-surface-variant">
                    Include the exact preprocessing used during training (normalization, encoding, etc.)
                  </p>
                </div>
                
                <div className="p-4 bg-surface-container-low rounded-xl border border-outline-variant">
                  <div className="flex items-center mb-2">
                    <Zap size={18} className="text-warning-500 mr-2" />
                    <h4 className="font-medium text-on-surface">Model Optimization</h4>
                  </div>
                  <p className="text-sm text-on-surface-variant">
                    For large models, consider quantization, ONNX, or TorchScript for better latency.
                  </p>
                </div>
                
                <div className="p-4 bg-surface-container-low rounded-xl border border-outline-variant">
                  <div className="flex items-center mb-2">
                    <BarChart3 size={18} className="text-success-500 mr-2" />
                    <h4 className="font-medium text-on-surface">Monitoring</h4>
                  </div>
                  <p className="text-sm text-on-surface-variant">
                    Collect logs, errors, and prediction distributions to detect drift.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Frontend Integration */}
        <div className="card p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'frontend' ? null : 'frontend')}
          >
            <h2 className="text-lg font-semibold text-on-surface">
              <Globe size={20} className="inline mr-2 text-success-500" />
              Step 3: Connect from Your Website
            </h2>
            {expandedSection === 'frontend' ? <ChevronUp size={20} className="text-on-surface-variant" /> : <ChevronDown size={20} className="text-on-surface-variant" />}
          </div>
          
          {expandedSection === 'frontend' && (
            <div className="mt-4">
              <p className="text-on-surface-variant text-sm mb-4">
                Call your deployed API from client-side JavaScript:
              </p>
              <div className="console-container rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-on-surface-variant">Frontend JavaScript</span>
                  <button 
                    onClick={() => copyToClipboard(`async function getPrediction(features) {
  const resp = await fetch("https://your-api-domain.com/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": "YOUR_API_KEY"   // optional security
    },
    body: JSON.stringify({ data: features })
  });
  
  if (!resp.ok) throw new Error('Prediction failed');
  
  const result = await resp.json();
  return result;
}

// Usage:
const prediction = await getPrediction({
  feature1: 100,
  feature2: "value"
});
console.log(prediction.label, prediction.confidence);`, 'frontend')}
                    className="text-on-surface-variant hover:text-primary-500"
                  >
                    {copied === 'frontend' ? <Check size={14} className="text-success-500" /> : <Copy size={14} />}
                  </button>
                </div>
                <pre className="text-sm text-success-500 overflow-x-auto">
{`async function getPrediction(features) {
  const resp = await fetch("https://your-api-domain.com/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": "YOUR_API_KEY"   // optional security
    },
    body: JSON.stringify({ data: features })
  });
  
  if (!resp.ok) throw new Error('Prediction failed');
  
  const result = await resp.json();
  return result;
}

// Usage:
const prediction = await getPrediction({
  feature1: 100,
  feature2: "value"
});
console.log(prediction.label, prediction.confidence);`}
                </pre>
              </div>
            </div>
          )}
        </div>

        {/* Production Checklist */}
        <div className="card p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'checklist' ? null : 'checklist')}
          >
            <div className="flex items-center">
              <h2 className="text-lg font-semibold text-on-surface">
                <CheckCircle size={20} className="inline mr-2 text-success-500" />
                Production Checklist
              </h2>
              <span className="ml-3 text-sm text-on-surface-variant">
                {completedItems}/{PRODUCTION_CHECKLIST.length} completed
              </span>
            </div>
            {expandedSection === 'checklist' ? <ChevronUp size={20} className="text-on-surface-variant" /> : <ChevronDown size={20} className="text-on-surface-variant" />}
          </div>
          
          {expandedSection === 'checklist' && (
            <div className="mt-4">
              {/* Progress Bar */}
              <div className="mb-4">
                <div className="w-full bg-surface-container-high rounded-full h-2 overflow-hidden">
                  <div 
                    className="bg-success-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${checklistProgress}%` }}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                {PRODUCTION_CHECKLIST.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => toggleChecklist(item.id)}
                    className={`w-full flex items-center p-3 rounded-xl transition ${
                      checklist[item.id] 
                        ? 'border border-success-500/30' 
                        : 'bg-surface-container-low hover:bg-surface-container-high border border-transparent'
                    }`}
                  >
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-3 ${
                      checklist[item.id] 
                        ? 'bg-success-500 text-white' 
                        : 'bg-surface-container-high text-on-surface-variant border border-outline-variant'
                    }`}>
                      {checklist[item.id] ? <Check size={14} /> : null}
                    </div>
                    <span className={`flex-1 text-left ${checklist[item.id] ? 'text-success-500' : 'text-on-surface-variant'}`}>
                      {item.text}
                    </span>
                    <span className="badge badge-primary text-xs">
                      {item.category}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* API Usage Example */}
        <div className="card p-6 mb-6">
          <h2 className="text-lg font-semibold text-on-surface mb-4">
            <FileCode size={20} className="inline mr-2 text-primary-500" />
            API Usage Example
          </h2>
          <div className="console-container rounded-xl p-4 overflow-x-auto">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-on-surface-variant">cURL Example</span>
              <button 
                onClick={() => copyToClipboard(`curl -X POST "http://localhost:8080/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": {"feature1": 100, "feature2": "value"}
  }'`, 'curl')}
                className="text-on-surface-variant hover:text-primary-500"
              >
                {copied === 'curl' ? <Check size={14} className="text-success-500" /> : <Copy size={14} />}
              </button>
            </div>
            <pre className="text-sm text-success-500">
{`# Make a prediction
curl -X POST "http://localhost:8080/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": {"feature1": 100, "feature2": "value"}
  }'

# Response:
{
  "prediction": "Class A",
  "confidence": 0.91,
  "probabilities": {"Class A": 0.91, "Class B": 0.09},
  "label": "Class A"
}`}
            </pre>
          </div>
        </div>

        {/* Existing Deployments */}
        {existingDeployments.length > 0 && (
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-on-surface mb-4">
              <Clock size={20} className="inline mr-2 text-success-500" />
              Previous Deployments
            </h2>
            <div className="space-y-3">
              {existingDeployments.map((dep) => (
                <div key={dep.id} className="flex items-center justify-between p-3 bg-surface-container-low rounded-xl border border-outline-variant">
                  <div>
                    <div className="font-medium text-on-surface capitalize">{dep.platform}</div>
                    <div className="text-sm text-on-surface-variant">{dep.created_at}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <code className="text-xs bg-surface-container-high px-2 py-1 rounded-lg text-on-surface-variant">{dep.url}</code>
                    <button
                      onClick={() => handleDownload(dep.id)}
                      className="p-2 text-primary-500 hover:bg-surface-container-high rounded-lg transition"
                      title="Download Package"
                    >
                      <Download size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Deployment;
