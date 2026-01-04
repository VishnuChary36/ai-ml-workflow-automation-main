import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Dataset API
export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await client.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const getDataset = async (datasetId) => {
  const response = await client.get(`/api/datasets/${datasetId}`);
  return response.data;
};

// Suggestions API
export const getSuggestedPipeline = async (datasetId, targetColumn = null) => {
  const params = { dataset_id: datasetId };
  if (targetColumn) params.target_column = targetColumn;
  
  const response = await client.get('/api/suggest/pipeline', { params });
  return response.data;
};

export const getSuggestedModels = async (datasetId, targetColumn, problemType = 'auto') => {
  const params = {
    dataset_id: datasetId,
    target_column: targetColumn,
    problem_type: problemType,
  };
  
  const response = await client.get('/api/suggest/models', { params });
  return response.data;
};

// Pipeline Execution API
export const runPipeline = async (datasetId, steps) => {
  const response = await client.post('/api/run_pipeline', steps, {
    params: { dataset_id: datasetId },
  });
  
  return response.data;
};

// Task Management API
export const getTaskStatus = async (taskId) => {
  const response = await client.get(`/api/task/${taskId}/status`);
  return response.data;
};

export const listTasks = async (limit = 50, taskType = null) => {
  const params = { limit };
  if (taskType) params.task_type = taskType;
  
  const response = await client.get('/api/tasks', { params });
  return response.data;
};

export const cancelTask = async (taskId) => {
  const response = await client.post(`/api/cancel/${taskId}`);
  return response.data;
};

// Logs API
export const getLogs = async (taskId) => {
  const response = await client.get(`/api/logs/${taskId}`);
  return response.data;
};

export const downloadLogsText = async (taskId) => {
  const response = await client.get(`/api/logs/${taskId}.txt`, {
    responseType: 'blob',
  });
  
  // Create download
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `${taskId}.log`);
  document.body.appendChild(link);
  link.click();
  link.remove();
};

// Models API
export const listModels = async () => {
  const response = await client.get('/api/models');
  return response.data;
};

// Model Training API
export const trainModel = async (datasetId, modelConfig, targetColumn) => {
  const response = await client.post('/api/train_model', modelConfig, {
    params: { dataset_id: datasetId, target_column: targetColumn },
  });
  
  return response.data;
};

export const deployModel = async (modelId, platform) => {
  const response = await client.post('/api/deploy_model', {}, {
    params: { model_id: modelId, platform },
  });
  
  return response.data;
};

// Deployment API
export const getDeployment = async (deploymentId) => {
  const response = await client.get(`/api/deployment/${deploymentId}`);
  return response.data;
};

export const getModelDeployments = async (modelId) => {
  const response = await client.get(`/api/model/${modelId}/deployments`);
  return response.data;
};

export const downloadDeploymentPackage = async (deploymentId) => {
  const response = await client.get(`/api/deployment/${deploymentId}/download`, {
    responseType: 'blob',
  });
  
  // Create download
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `deployment_${deploymentId}.zip`);
  document.body.appendChild(link);
  link.click();
  link.remove();
};

// Visualization API
export const getVisualizations = async (modelId) => {
  const response = await client.get(`/api/visualizations/${modelId}`);
  
  return response.data;
};

// Explainability API
export const getExplainability = async (modelId) => {
  const response = await client.get(`/api/explainability/${modelId}`);
  return response.data;
};

export default client;
