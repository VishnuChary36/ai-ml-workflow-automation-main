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
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center">
              <Rocket className="text-green-600 mr-3" size={28} />
              <h1 className="text-2xl font-bold text-gray-800">Deploy Model</h1>
            </div>
            <p className="text-gray-600 ml-10">
              Package and deploy <span className="font-semibold">{modelName || modelId}</span> to production
            </p>
          </div>
          <button
            onClick={onBack}
            className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 border border-gray-300 rounded hover:bg-gray-50 transition"
          >
            <ArrowLeft size={18} className="mr-2" />
            Back
          </button>
        </div>

        {/* High-Level Overview Banner */}
        <div className="bg-gradient-to-r from-green-600 to-emerald-700 rounded-xl shadow-lg p-6 mb-6 text-white">
          <div className="flex items-start">
            <Info size={24} className="mr-3 flex-shrink-0 mt-0.5" />
            <div>
              <h2 className="text-xl font-bold mb-2">Deployment Flow Overview</h2>
              <p className="opacity-90">
                Package your model inside a web service (API endpoint), deploy to a public host (real URL + HTTPS), 
                and connect your website to call the service from frontend.
              </p>
            </div>
          </div>
        </div>

        {/* Deployment Pipeline Overview */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Deployment Pipeline</h2>
          <div className="flex items-center justify-between">
            {['Export Model', 'Create API', 'Containerize', 'Live Endpoint', 'Frontend Ready'].map((step, idx) => {
              // Calculate completion based on deployment result - ALL steps complete when deployed
              const isCompleted = deploymentResult ? true : (idx <= 2 && deploying);
              const isActive = deploying && !deploymentResult && idx === 3;
              
              return (
                <React.Fragment key={step}>
                  <div className="flex flex-col items-center">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                      isCompleted ? 'bg-green-500 text-white' : 
                      isActive ? 'bg-blue-500 text-white animate-pulse' : 
                      'bg-gray-200 text-gray-500'
                    }`}>
                      {isCompleted ? <CheckCircle size={20} /> : idx + 1}
                    </div>
                    <span className="text-xs mt-2 text-gray-600 text-center max-w-[80px]">{step}</span>
                  </div>
                  {idx < 4 && (
                    <div className={`flex-1 h-1 mx-2 ${
                      deploymentResult ? 'bg-green-500' : (isCompleted && idx < 3 ? 'bg-green-500' : 'bg-gray-200')
                    }`} />
                  )}
                </React.Fragment>
              );
            })}
          </div>
          {deploymentResult && (
            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-green-700 text-sm">
                🎉 <strong>All steps complete!</strong> Your model is live and ready to accept predictions. Use the code snippets below to integrate with your application.
              </p>
            </div>
          )}
        </div>

        {/* Platform Selection */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'platform' ? null : 'platform')}
          >
            <h2 className="text-lg font-semibold text-gray-800">
              <Server size={20} className="inline mr-2" />
              Step 1: Select Deployment Platform
            </h2>
            {expandedSection === 'platform' ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>
          
          {expandedSection === 'platform' && (
            <div className="mt-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(PLATFORM_INFO).map(([key, platform]) => {
                  const Icon = platform.icon;
                  const isSelected = selectedPlatform === key;
                  
                  return (
                    <button
                      key={key}
                      onClick={(e) => { e.stopPropagation(); setSelectedPlatform(key); }}
                      disabled={deploying}
                      className={`p-4 rounded-lg border-2 transition text-left ${
                        isSelected 
                          ? 'border-green-500 bg-green-50' 
                          : 'border-gray-200 hover:border-gray-300'
                      } ${deploying ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <div className="flex items-center mb-2">
                        <Icon size={24} className={isSelected ? 'text-green-600' : 'text-gray-500'} />
                        <span className={`ml-2 font-semibold ${isSelected ? 'text-green-700' : 'text-gray-700'}`}>
                          {platform.name}
                        </span>
                      </div>
                      <p className="text-sm text-gray-500 mb-3">{platform.description}</p>
                      <div className="flex flex-wrap gap-1">
                        {platform.features.map((f, i) => (
                          <span key={i} className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded">
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
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'package' ? null : 'package')}
          >
            <h2 className="text-lg font-semibold text-gray-800">
              <Package size={20} className="inline mr-2" />
              Deployment Package Contents
            </h2>
            {expandedSection === 'package' ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>
          
          {expandedSection === 'package' && (
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
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
                <div key={idx} className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition group">
                  <item.icon size={18} className="text-gray-500 mb-1" />
                  <div className="font-medium text-gray-700 text-sm">{item.name}</div>
                  <div className="text-xs text-gray-500">{item.desc}</div>
                  <div className="text-xs text-blue-600 opacity-0 group-hover:opacity-100 transition mt-1">{item.tip}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Deploy Button */}
        <div className="bg-gradient-to-r from-green-600 to-emerald-700 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="text-white">
              <h2 className="text-xl font-bold mb-1">Create Deployment Package</h2>
              <p className="opacity-90">
                {selectedPlatform 
                  ? `Generate package for ${PLATFORM_INFO[selectedPlatform].name}` 
                  : 'Select a platform above to get started'}
              </p>
            </div>
            <button
              onClick={handleDeploy}
              disabled={!selectedPlatform || deploying}
              className={`flex items-center px-6 py-3 rounded-lg font-semibold transition shadow-md ${
                selectedPlatform && !deploying
                  ? 'bg-white text-green-700 hover:bg-green-50'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              {deploying ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-green-700 mr-2" />
                  Creating Package...
                </>
              ) : (
                <>
                  <Rocket size={18} className="mr-2" />
                  Generate Package
                </>
              )}
            </button>
          </div>
        </div>

        {/* Task Progress with Real-Time Logs */}
        {taskId && taskStatus?.toUpperCase() === 'RUNNING' && (
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 mb-6">
            <div className="flex items-center mb-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-3" />
              <div>
                <h3 className="font-semibold text-blue-800">Creating Deployment Package</h3>
                <p className="text-blue-600 text-sm">Packaging model with all artifacts...</p>
              </div>
            </div>
            
            {/* Real-Time Log Console */}
            <div className="bg-gray-900 rounded-lg p-4 max-h-64 overflow-y-auto font-mono text-sm">
              {logs.length === 0 ? (
                <div className="text-gray-400">Waiting for deployment logs...</div>
              ) : (
                logs.map((log, idx) => (
                  <div key={idx} className={`mb-1 ${
                    log.level === 'ERROR' ? 'text-red-400' :
                    log.level === 'WARNING' ? 'text-yellow-400' :
                    'text-green-400'
                  }`}>
                    <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
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
          <div className="bg-red-50 border border-red-200 rounded-xl p-6 mb-6">
            <div className="flex items-center">
              <AlertTriangle size={24} className="text-red-500 mr-3" />
              <div>
                <h3 className="font-semibold text-red-800">Deployment Failed</h3>
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Deployment Result */}
        {deploymentResult && (
          <div className="bg-green-50 border border-green-200 rounded-xl p-6 mb-6">
            <div className="flex items-start mb-4">
              <CheckCircle size={24} className="text-green-500 mr-3 mt-0.5" />
              <div>
                <h3 className="font-semibold text-green-800">🎉 Model Deployed & Ready!</h3>
                <p className="text-green-600 text-sm">Your model is LIVE and ready to serve predictions</p>
              </div>
            </div>
            
            <div className="space-y-4">
              {/* LIVE Prediction Endpoint */}
              <div className="bg-white rounded-lg p-4 border-2 border-green-300">
                <div className="flex items-center mb-2">
                  <span className="inline-block w-3 h-3 bg-green-500 rounded-full animate-pulse mr-2"></span>
                  <span className="text-sm font-semibold text-green-700">LIVE Prediction Endpoint</span>
                </div>
                <div className="flex items-center justify-between bg-gray-100 rounded p-2">
                  <code className="text-sm font-mono text-gray-800">
                    {deploymentResult.live_prediction_url || deploymentResult.deployment_url}
                  </code>
                  <button 
                    onClick={() => copyToClipboard(deploymentResult.live_prediction_url || deploymentResult.deployment_url, 'live-url')}
                    className="ml-2 text-gray-500 hover:text-gray-700 flex items-center"
                  >
                    {copied === 'live-url' ? <Check size={16} className="text-green-500" /> : <Copy size={16} />}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  ✅ This endpoint is ready to use NOW - no additional setup required!
                </p>
              </div>
              
              {/* Frontend Integration Code */}
              <div className="bg-gray-900 rounded-lg p-4 text-white">
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
                    className="text-gray-400 hover:text-white"
                  >
                    {copied === 'js-code' ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                  </button>
                </div>
                <pre className="text-sm text-green-400 overflow-x-auto">
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
              <div className="bg-gray-900 rounded-lg p-4 text-white">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <Code size={16} className="mr-2 text-yellow-400" />
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
                    className="text-gray-400 hover:text-white"
                  >
                    {copied === 'py-code' ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                  </button>
                </div>
                <pre className="text-sm text-yellow-400 overflow-x-auto">
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
              <div className="bg-gray-900 rounded-lg p-4 text-white">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <Terminal size={16} className="mr-2 text-gray-400" />
                    <span className="text-sm font-semibold text-white">cURL / Command Line</span>
                  </div>
                  <button 
                    onClick={() => copyToClipboard(`curl -X POST "${deploymentResult.live_prediction_url || deploymentResult.deployment_url}" \\
  -H "Content-Type: application/json" \\
  -d '{"features": {"feature1": "value1", "feature2": 123}}'`, 'curl-code')}
                    className="text-gray-400 hover:text-white"
                  >
                    {copied === 'curl-code' ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                  </button>
                </div>
                <pre className="text-sm text-cyan-400 overflow-x-auto">
{`curl -X POST "${deploymentResult.live_prediction_url || deploymentResult.deployment_url}" \\
  -H "Content-Type: application/json" \\
  -d '{"features": {"feature1": "value1", "feature2": 123}}'`}
                </pre>
              </div>
              
              {/* Get Feature Info Button */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold text-blue-800">Need exact feature names?</h4>
                    <p className="text-blue-600 text-sm">Get the full API documentation including all required features</p>
                  </div>
                  <a 
                    href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/predict/${modelId}/info`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition flex items-center"
                  >
                    <ExternalLink size={16} className="mr-2" />
                    View API Docs
                  </a>
                </div>
              </div>
              
              <hr className="border-gray-200" />
              
              {/* Standalone Package Download */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-700 mb-2">📦 Standalone Deployment Package</h4>
                <p className="text-gray-600 text-sm mb-3">
                  Download the complete package to deploy as a separate service on your own infrastructure.
                </p>
                
                {/* Download Button */}
                <button
                  onClick={() => handleDownload(deploymentResult.deployment_id)}
                  className="w-full flex items-center justify-center px-4 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-800 transition"
                >
                  <Download size={18} className="mr-2" />
                  Download Deployment Package (.zip)
                </button>
                
                {/* Package Files */}
                {deploymentResult.files && deploymentResult.files.length > 0 && (
                  <div className="mt-3">
                    <div className="text-sm font-medium text-gray-700 mb-2">Package Contents:</div>
                    <div className="flex flex-wrap gap-2">
                      {deploymentResult.files.map((file, idx) => (
                        <span key={idx} className="px-2 py-1 bg-white border border-gray-200 text-gray-600 text-xs rounded">
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
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'cloud' ? null : 'cloud')}
          >
            <h2 className="text-lg font-semibold text-gray-800">
              <Cloud size={20} className="inline mr-2" />
              Step 2: Deploy to Cloud (Choose Provider)
            </h2>
            {expandedSection === 'cloud' ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>
          
          {expandedSection === 'cloud' && (
            <div className="mt-4 space-y-3">
              <p className="text-gray-600 text-sm mb-4">
                After generating your package, deploy to any of these hosting services to get a public HTTPS URL:
              </p>
              
              {CLOUD_PROVIDERS.map((provider) => (
                <div 
                  key={provider.id}
                  className="border border-gray-200 rounded-lg overflow-hidden"
                >
                  <button
                    onClick={() => setExpandedProvider(expandedProvider === provider.id ? null : provider.id)}
                    className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition"
                  >
                    <div className="flex items-center">
                      <span className="text-2xl mr-3">{provider.icon}</span>
                      <div className="text-left">
                        <div className="font-semibold text-gray-800">{provider.name}</div>
                        <div className="text-sm text-gray-500">{provider.description}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`px-2 py-1 text-xs rounded ${
                        provider.difficulty === 'Very Easy' ? 'bg-green-100 text-green-700' :
                        provider.difficulty === 'Easy' ? 'bg-blue-100 text-blue-700' :
                        'bg-yellow-100 text-yellow-700'
                      }`}>
                        {provider.difficulty}
                      </span>
                      <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
                        {provider.cost}
                      </span>
                      {expandedProvider === provider.id ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                    </div>
                  </button>
                  
                  {expandedProvider === provider.id && (
                    <div className="px-4 pb-4 bg-gray-50">
                      <div className="bg-gray-900 rounded-lg p-4 text-white">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-400">Deployment Commands</span>
                          <button 
                            onClick={() => copyToClipboard(provider.commands.join('\n'), provider.id)}
                            className="text-gray-400 hover:text-white"
                          >
                            {copied === provider.id ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                          </button>
                        </div>
                        <pre className="text-sm text-green-400 overflow-x-auto whitespace-pre-wrap">
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
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'security' ? null : 'security')}
          >
            <h2 className="text-lg font-semibold text-gray-800">
              <Shield size={20} className="inline mr-2" />
              Security & Production Considerations
            </h2>
            {expandedSection === 'security' ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>
          
          {expandedSection === 'security' && (
            <div className="mt-4 space-y-4">
              {/* CORS Configuration */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center mb-2">
                  <Globe size={18} className="text-blue-600 mr-2" />
                  <h3 className="font-semibold text-blue-800">CORS Configuration</h3>
                </div>
                <p className="text-sm text-blue-700 mb-3">
                  Enable CORS so browser requests from your website domain succeed:
                </p>
                <div className="bg-gray-900 rounded-lg p-4 text-white">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-400">FastAPI CORS Middleware</span>
                    <button 
                      onClick={() => copyToClipboard(`from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-website.com"],  # restrict to your site
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)`, 'cors')}
                      className="text-gray-400 hover:text-white"
                    >
                      {copied === 'cors' ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                    </button>
                  </div>
                  <pre className="text-sm text-green-400 overflow-x-auto">
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
              <div className="p-4 bg-yellow-50 rounded-lg">
                <div className="flex items-center mb-2">
                  <Key size={18} className="text-yellow-600 mr-2" />
                  <h3 className="font-semibold text-yellow-800">API Key Authentication</h3>
                </div>
                <p className="text-sm text-yellow-700 mb-3">
                  Add API key check to prevent abuse:
                </p>
                <div className="bg-gray-900 rounded-lg p-4 text-white">
                  <pre className="text-sm text-green-400 overflow-x-auto">
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
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center mb-2">
                    <Lock size={18} className="text-gray-600 mr-2" />
                    <h4 className="font-medium text-gray-800">HTTPS & Domain</h4>
                  </div>
                  <p className="text-sm text-gray-600">
                    Cloud hosts provide HTTPS by default. Point your domain to the service for custom URLs.
                  </p>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center mb-2">
                    <RefreshCw size={18} className="text-gray-600 mr-2" />
                    <h4 className="font-medium text-gray-800">Preprocessing Parity</h4>
                  </div>
                  <p className="text-sm text-gray-600">
                    Include the exact preprocessing used during training (normalization, encoding, etc.)
                  </p>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center mb-2">
                    <Zap size={18} className="text-gray-600 mr-2" />
                    <h4 className="font-medium text-gray-800">Model Optimization</h4>
                  </div>
                  <p className="text-sm text-gray-600">
                    For large models, consider quantization, ONNX, or TorchScript for better latency.
                  </p>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center mb-2">
                    <BarChart3 size={18} className="text-gray-600 mr-2" />
                    <h4 className="font-medium text-gray-800">Monitoring</h4>
                  </div>
                  <p className="text-sm text-gray-600">
                    Collect logs, errors, and prediction distributions to detect drift.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Frontend Integration */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'frontend' ? null : 'frontend')}
          >
            <h2 className="text-lg font-semibold text-gray-800">
              <Globe size={20} className="inline mr-2" />
              Step 3: Connect from Your Website
            </h2>
            {expandedSection === 'frontend' ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>
          
          {expandedSection === 'frontend' && (
            <div className="mt-4">
              <p className="text-gray-600 text-sm mb-4">
                Call your deployed API from client-side JavaScript:
              </p>
              <div className="bg-gray-900 rounded-lg p-4 text-white">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Frontend JavaScript</span>
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
                    className="text-gray-400 hover:text-white"
                  >
                    {copied === 'frontend' ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                  </button>
                </div>
                <pre className="text-sm text-green-400 overflow-x-auto">
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
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <div 
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setExpandedSection(expandedSection === 'checklist' ? null : 'checklist')}
          >
            <div className="flex items-center">
              <h2 className="text-lg font-semibold text-gray-800">
                <CheckCircle size={20} className="inline mr-2" />
                Production Checklist
              </h2>
              <span className="ml-3 text-sm text-gray-500">
                {completedItems}/{PRODUCTION_CHECKLIST.length} completed
              </span>
            </div>
            {expandedSection === 'checklist' ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </div>
          
          {expandedSection === 'checklist' && (
            <div className="mt-4">
              {/* Progress Bar */}
              <div className="mb-4">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${checklistProgress}%` }}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                {PRODUCTION_CHECKLIST.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => toggleChecklist(item.id)}
                    className={`w-full flex items-center p-3 rounded-lg transition ${
                      checklist[item.id] 
                        ? 'bg-green-50 border border-green-200' 
                        : 'bg-gray-50 hover:bg-gray-100 border border-transparent'
                    }`}
                  >
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-3 ${
                      checklist[item.id] 
                        ? 'bg-green-500 text-white' 
                        : 'bg-gray-200 text-gray-400'
                    }`}>
                      {checklist[item.id] ? <Check size={14} /> : null}
                    </div>
                    <span className={`flex-1 text-left ${checklist[item.id] ? 'text-green-800' : 'text-gray-700'}`}>
                      {item.text}
                    </span>
                    <span className="px-2 py-0.5 bg-gray-200 text-gray-600 text-xs rounded">
                      {item.category}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* API Usage Example */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            <FileCode size={20} className="inline mr-2" />
            API Usage Example
          </h2>
          <div className="bg-gray-900 rounded-lg p-4 text-white overflow-x-auto">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">cURL Example</span>
              <button 
                onClick={() => copyToClipboard(`curl -X POST "http://localhost:8080/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": {"feature1": 100, "feature2": "value"}
  }'`, 'curl')}
                className="text-gray-400 hover:text-white"
              >
                {copied === 'curl' ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
              </button>
            </div>
            <pre className="text-sm">
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
          <div className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              <Clock size={20} className="inline mr-2" />
              Previous Deployments
            </h2>
            <div className="space-y-3">
              {existingDeployments.map((dep) => (
                <div key={dep.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <div className="font-medium text-gray-700 capitalize">{dep.platform}</div>
                    <div className="text-sm text-gray-500">{dep.created_at}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <code className="text-xs bg-gray-200 px-2 py-1 rounded">{dep.url}</code>
                    <button
                      onClick={() => handleDownload(dep.id)}
                      className="p-2 text-blue-600 hover:bg-blue-50 rounded"
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
