import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import GrafanaDashboard from '../components/GrafanaDashboard/GrafanaDashboard';

export default function GrafanaDashboardPage() {
  const { modelId } = useParams();
  const navigate = useNavigate();
  
  if (!modelId || modelId === 'undefined') {
    return (
      <div className="page-container flex items-center justify-center min-h-[60vh]">
        <div className="text-center p-8 bg-surface-container rounded-xl shadow-sm border border-outline-variant">
          <h2 className="text-2xl font-bold text-on-surface mb-2">Model Not Found</h2>
          <p className="text-on-surface-variant mb-6">The requested Grafana dashboard could not be loaded.</p>
          <button onClick={() => navigate('/')} className="btn-primary">Return Home</button>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <GrafanaDashboard 
        modelId={modelId} 
        onBack={() => navigate(`/visualizations/${modelId}`)} 
      />
    </div>
  );
}
