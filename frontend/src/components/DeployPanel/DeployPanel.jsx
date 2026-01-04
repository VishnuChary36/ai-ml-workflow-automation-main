import React from 'react';
import { Rocket, Download } from 'lucide-react';

const DeployPanel = ({ modelId, modelName }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Deploy Model</h2>
      
      {!modelId ? (
        <div className="text-center py-8">
          <p className="text-gray-500">Train a model first to enable deployment</p>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-800">
              <span className="font-semibold">Model:</span> {modelName || modelId}
            </p>
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Deployment Platform
            </label>
            <select className="w-full px-4 py-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
              <option value="docker">Local Docker</option>
              <option value="hf">Hugging Face Spaces</option>
              <option value="aws">AWS ECS</option>
              <option value="gcp">GCP Cloud Run</option>
            </select>
          </div>
          
          <div className="flex space-x-4">
            <button className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition flex items-center justify-center">
              <Rocket size={20} className="mr-2" />
              Deploy
            </button>
            <button className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 transition flex items-center">
              <Download size={20} className="mr-2" />
              Export
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DeployPanel;
