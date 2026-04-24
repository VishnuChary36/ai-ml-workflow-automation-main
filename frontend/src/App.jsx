import React from 'react';
import { Routes, Route } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary/ErrorBoundary';
import AppLayout from './components/Layout/AppLayout';

import MainDashboard from './pages/MainDashboard';
import VisualizationPage from './pages/VisualizationPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import DeploymentPage from './pages/DeploymentPage';
import GrafanaDashboardPage from './pages/GrafanaDashboardPage';

// You will need react-hot-toast for toast notifications
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <ErrorBoundary>
      <Toaster 
        position="top-right" 
        toastOptions={{
          style: {
            borderRadius: '12px',
            background: 'rgba(25, 31, 47, 0.95)',
            color: '#dde2f8',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            backdropFilter: 'blur(12px)',
            fontSize: '14px',
          },
          success: {
            iconTheme: { primary: '#10b981', secondary: '#dde2f8' },
          },
          error: {
            iconTheme: { primary: '#ef4444', secondary: '#dde2f8' },
          },
        }}
      />
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<MainDashboard />} />
          <Route path="/visualizations/:modelId" element={<VisualizationPage />} />
          <Route path="/explainability/:modelId" element={<ExplainabilityPage />} />
          <Route path="/deploy/:modelId" element={<DeploymentPage />} />
          <Route path="/grafana/:modelId" element={<GrafanaDashboardPage />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  );
}

export default App;
