import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BarChart3,
  PieChart,
  LineChart,
  TrendingUp,
  TrendingDown,
  Database,
  CheckCircle,
  AlertTriangle,
  Layers,
  Hash,
  Tag,
  RefreshCw,
  Download,
  ChevronDown,
  ChevronUp,
  Lightbulb,
  Target,
  Sparkles,
  ArrowRight,
  ArrowLeft,
  Info,
  FileText,
  Activity,
  Zap,
  Award,
  BookOpen,
  Eye,
  Brain
} from 'lucide-react';

/**
 * DataDashboard - Power BI/Excel-style storytelling dashboard
 * 
 * Features:
 * - Executive Summary with narrative
 * - KPI cards
 * - Interactive charts (Bar, Pie, Histogram, Scatter, Box plots)
 * - Correlation heatmap
 * - Insights and recommendations
 * - Data quality scorecard
 */
const DataDashboard = ({ taskId: propTaskId, onBack }) => {
  const navigate = useNavigate();
  const [taskId, setTaskId] = useState(propTaskId || '');
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [trainedModels, setTrainedModels] = useState([]);
  const [expandedSections, setExpandedSections] = useState({
    summary: true,
    kpis: true,
    quality: true,
    distributions: true,
    correlations: true,
    categorical: true,
    insights: true,
    recommendations: true
  });
  const [selectedChart, setSelectedChart] = useState(null);

  // Fetch available datasets
  useEffect(() => {
    fetchAvailableDatasets();
  }, []);

  const fetchAvailableDatasets = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/processed-datasets');
      const data = await response.json();
      setAvailableDatasets(data.datasets || []);
      
      // Auto-select first dataset if none selected
      if (!taskId && data.datasets?.length > 0) {
        setTaskId(data.datasets[0].task_id);
      }
    } catch (err) {
      console.error('Failed to fetch datasets:', err);
    }
  };

  // Fetch dashboard data
  const fetchDashboard = useCallback(async () => {
    if (!taskId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const url = targetColumn
        ? `http://localhost:8000/api/dashboard/${taskId}?target_column=${encodeURIComponent(targetColumn)}`
        : `http://localhost:8000/api/dashboard/${taskId}`;
      
      const response = await fetch(url);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to load dashboard');
      }
      
      const data = await response.json();
      setDashboard(data);
      
      // Set target column from response if not already set
      if (!targetColumn && data.target_column) {
        setTargetColumn(data.target_column);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [taskId, targetColumn]);

  useEffect(() => {
    if (taskId) {
      fetchDashboard();
    }
  }, [taskId, fetchDashboard]);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Color schemes
  const colors = {
    primary: '#3B82F6',
    success: '#10B981',
    warning: '#F59E0B',
    danger: '#EF4444',
    purple: '#8B5CF6',
    pink: '#EC4899',
    indigo: '#6366F1',
    teal: '#14B8A6',
    chartColors: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#6366F1', '#F97316', '#06B6D4']
  };

  // Get icon by name
  const getIcon = (iconName, className = "") => {
    const iconMap = {
      'database': <Database className={className} />,
      'check-circle': <CheckCircle className={className} />,
      'layers': <Layers className={className} />,
      'hash': <Hash className={className} />,
      'tag': <Tag className={className} />,
      'pie-chart': <PieChart className={className} />,
      'trending-up': <TrendingUp className={className} />,
      'trending-down': <TrendingDown className={className} />,
      'alert-triangle': <AlertTriangle className={className} />,
      'lightbulb': <Lightbulb className={className} />,
      'target': <Target className={className} />,
      'link': <Activity className={className} />,
      'info': <Info className={className} />
    };
    return iconMap[iconName] || <Sparkles className={className} />;
  };

  // Section Header Component
  const SectionHeader = ({ title, icon, section, badge }) => (
    <button
      onClick={() => toggleSection(section)}
      className="w-full flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-white hover:from-blue-50 hover:to-white transition-all rounded-t-xl"
    >
      <div className="flex items-center gap-3">
        {icon}
        <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
        {badge && (
          <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-700 rounded-full">
            {badge}
          </span>
        )}
      </div>
      {expandedSections[section] ? (
        <ChevronUp className="w-5 h-5 text-gray-500" />
      ) : (
        <ChevronDown className="w-5 h-5 text-gray-500" />
      )}
    </button>
  );

  // KPI Card Component
  const KPICard = ({ kpi }) => {
    const colorMap = {
      blue: 'from-blue-500 to-blue-600',
      green: 'from-green-500 to-green-600',
      yellow: 'from-yellow-500 to-yellow-600',
      red: 'from-red-500 to-red-600',
      purple: 'from-purple-500 to-purple-600',
      indigo: 'from-indigo-500 to-indigo-600',
      pink: 'from-pink-500 to-pink-600',
      orange: 'from-orange-500 to-orange-600'
    };

    return (
      <div className="bg-white rounded-xl shadow-md hover:shadow-lg transition-all p-5 border border-gray-100">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-gray-500">{kpi.name}</p>
            <p className="text-3xl font-bold text-gray-900 mt-1">{kpi.value}</p>
            <p className="text-xs text-gray-400 mt-2">{kpi.description}</p>
          </div>
          <div className={`p-3 rounded-xl bg-gradient-to-br ${colorMap[kpi.color] || 'from-gray-500 to-gray-600'}`}>
            {getIcon(kpi.icon, "w-6 h-6 text-white")}
          </div>
        </div>
      </div>
    );
  };

  // Bar Chart Component
  const BarChart = ({ data, title, horizontal = false }) => {
    if (!data?.labels || !data?.values) return null;
    
    const maxValue = Math.max(...data.values);
    
    return (
      <div className="bg-white rounded-xl shadow-md p-5 border border-gray-100">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">{title}</h4>
        <div className="space-y-3">
          {data.labels.map((label, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="w-24 text-xs text-gray-600 truncate" title={label}>
                {label}
              </div>
              <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${(data.values[i] / maxValue) * 100}%`,
                    backgroundColor: colors.chartColors[i % colors.chartColors.length]
                  }}
                />
              </div>
              <div className="w-16 text-xs font-medium text-gray-700 text-right">
                {data.values[i].toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Pie Chart Component
  const PieChartComponent = ({ data, title }) => {
    if (!data?.labels || !data?.values) return null;
    
    const total = data.values.reduce((a, b) => a + b, 0);
    let cumulative = 0;
    
    return (
      <div className="bg-white rounded-xl shadow-md p-5 border border-gray-100">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">{title}</h4>
        <div className="flex items-center gap-6">
          {/* SVG Pie Chart */}
          <svg viewBox="0 0 100 100" className="w-32 h-32">
            {data.values.map((value, i) => {
              const percentage = (value / total) * 100;
              const startAngle = cumulative * 3.6;
              cumulative += percentage;
              const endAngle = cumulative * 3.6;
              
              const x1 = 50 + 40 * Math.cos((startAngle - 90) * Math.PI / 180);
              const y1 = 50 + 40 * Math.sin((startAngle - 90) * Math.PI / 180);
              const x2 = 50 + 40 * Math.cos((endAngle - 90) * Math.PI / 180);
              const y2 = 50 + 40 * Math.sin((endAngle - 90) * Math.PI / 180);
              const largeArc = percentage > 50 ? 1 : 0;
              
              return (
                <path
                  key={i}
                  d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArc} 1 ${x2} ${y2} Z`}
                  fill={colors.chartColors[i % colors.chartColors.length]}
                  className="hover:opacity-80 transition-opacity cursor-pointer"
                />
              );
            })}
          </svg>
          
          {/* Legend */}
          <div className="flex-1 space-y-2">
            {data.labels.slice(0, 6).map((label, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: colors.chartColors[i % colors.chartColors.length] }}
                />
                <span className="text-gray-600 truncate flex-1">{label}</span>
                <span className="font-medium text-gray-800">
                  {((data.values[i] / total) * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // Histogram Component
  const Histogram = ({ data, title }) => {
    if (!data?.counts) return null;
    
    const maxCount = Math.max(...data.counts);
    
    return (
      <div className="bg-white rounded-xl shadow-md p-5 border border-gray-100">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">{title}</h4>
        <div className="flex items-end gap-1 h-32">
          {data.counts.map((count, i) => (
            <div
              key={i}
              className="flex-1 bg-blue-500 hover:bg-blue-600 transition-colors rounded-t cursor-pointer"
              style={{ height: `${(count / maxCount) * 100}%` }}
              title={`${data.bins?.[i]?.toFixed(2) || i}: ${count}`}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>{data.bins?.[0]?.toFixed(2) || 'Min'}</span>
          <span>{data.bins?.[Math.floor(data.bins?.length / 2)]?.toFixed(2) || 'Mid'}</span>
          <span>{data.bins?.[data.bins?.length - 1]?.toFixed(2) || 'Max'}</span>
        </div>
      </div>
    );
  };

  // Correlation Heatmap Component
  const CorrelationHeatmap = ({ correlations }) => {
    if (!correlations?.columns || !correlations?.heatmap_data) return null;
    
    const getCorrelationColor = (value) => {
      if (value >= 0.7) return '#10B981';
      if (value >= 0.4) return '#A7F3D0';
      if (value >= 0) return '#FEF3C7';
      if (value >= -0.4) return '#FECACA';
      return '#EF4444';
    };
    
    return (
      <div className="bg-white rounded-xl shadow-md p-5 border border-gray-100">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">Correlation Heatmap</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr>
                <th></th>
                {correlations.columns.slice(0, 8).map((col, i) => (
                  <th key={i} className="p-1 text-center truncate max-w-16" title={col}>
                    {col.slice(0, 6)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {correlations.columns.slice(0, 8).map((row, i) => (
                <tr key={i}>
                  <td className="p-1 font-medium truncate max-w-16" title={row}>
                    {row.slice(0, 8)}
                  </td>
                  {correlations.heatmap_data[i]?.slice(0, 8).map((value, j) => (
                    <td
                      key={j}
                      className="p-2 text-center text-[10px] font-medium"
                      style={{
                        backgroundColor: getCorrelationColor(value),
                        color: Math.abs(value) > 0.5 ? 'white' : 'black'
                      }}
                      title={`${correlations.columns[i]} vs ${correlations.columns[j]}: ${value}`}
                    >
                      {value?.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {/* Legend */}
        <div className="flex items-center justify-center gap-2 mt-4 text-xs">
          <span>-1</span>
          <div className="flex">
            {['#EF4444', '#FECACA', '#FEF3C7', '#A7F3D0', '#10B981'].map((color, i) => (
              <div key={i} className="w-8 h-3" style={{ backgroundColor: color }} />
            ))}
          </div>
          <span>+1</span>
        </div>
      </div>
    );
  };

  // Top Correlations List
  const TopCorrelations = ({ correlations }) => {
    if (!correlations?.top_correlations) return null;
    
    return (
      <div className="bg-white rounded-xl shadow-md p-5 border border-gray-100">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">Top Correlated Features</h4>
        <div className="space-y-3">
          {correlations.top_correlations.slice(0, 5).map((corr, i) => (
            <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-gray-700">{corr.feature1}</span>
                <ArrowRight className="w-4 h-4 text-gray-400" />
                <span className="text-xs font-medium text-gray-700">{corr.feature2}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`px-2 py-1 text-xs font-medium rounded ${
                  corr.strength === 'Strong' ? 'bg-green-100 text-green-700' :
                  corr.strength === 'Moderate' ? 'bg-yellow-100 text-yellow-700' :
                  'bg-gray-100 text-gray-700'
                }`}>
                  {corr.correlation.toFixed(3)}
                </span>
                {corr.direction === 'Positive' ? (
                  <TrendingUp className="w-4 h-4 text-green-500" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-red-500" />
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Insight Card Component
  const InsightCard = ({ insight }) => {
    const typeStyles = {
      success: 'border-l-green-500 bg-green-50',
      info: 'border-l-blue-500 bg-blue-50',
      warning: 'border-l-yellow-500 bg-yellow-50',
      error: 'border-l-red-500 bg-red-50'
    };
    
    const iconColors = {
      success: 'text-green-600',
      info: 'text-blue-600',
      warning: 'text-yellow-600',
      error: 'text-red-600'
    };
    
    return (
      <div className={`p-4 rounded-lg border-l-4 ${typeStyles[insight.type] || typeStyles.info}`}>
        <div className="flex items-start gap-3">
          <div className={`mt-1 ${iconColors[insight.type] || iconColors.info}`}>
            {getIcon(insight.icon, "w-5 h-5")}
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium text-gray-500">{insight.category}</span>
            </div>
            <h5 className="font-semibold text-gray-800">{insight.title}</h5>
            <p className="text-sm text-gray-600 mt-1">{insight.description}</p>
          </div>
        </div>
      </div>
    );
  };

  // Recommendation Card Component
  const RecommendationCard = ({ recommendation }) => {
    const priorityStyles = {
      high: 'border-red-200 bg-red-50',
      medium: 'border-yellow-200 bg-yellow-50',
      low: 'border-green-200 bg-green-50'
    };
    
    const priorityBadge = {
      high: 'bg-red-500',
      medium: 'bg-yellow-500',
      low: 'bg-green-500'
    };
    
    return (
      <div className={`p-4 rounded-lg border ${priorityStyles[recommendation.priority] || priorityStyles.medium}`}>
        <div className="flex items-start justify-between mb-2">
          <span className="text-xs font-medium text-gray-500">{recommendation.category}</span>
          <span className={`px-2 py-0.5 text-xs font-medium text-white rounded-full ${priorityBadge[recommendation.priority] || priorityBadge.medium}`}>
            {recommendation.priority} priority
          </span>
        </div>
        <h5 className="font-semibold text-gray-800">{recommendation.title}</h5>
        <p className="text-sm text-gray-600 mt-1">{recommendation.description}</p>
        <div className="flex items-center gap-2 mt-3 text-sm text-blue-600">
          <Zap className="w-4 h-4" />
          <span>{recommendation.action}</span>
        </div>
      </div>
    );
  };

  // Data Quality Scorecard
  const DataQualityCard = ({ quality }) => {
    if (!quality) return null;
    
    const getRatingColor = (rating) => {
      switch(rating) {
        case 'Excellent': return 'text-green-600';
        case 'Good': return 'text-blue-600';
        case 'Fair': return 'text-yellow-600';
        default: return 'text-red-600';
      }
    };
    
    return (
      <div className="bg-gradient-to-br from-white to-gray-50 rounded-xl shadow-md p-6 border border-gray-100">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h4 className="text-lg font-semibold text-gray-800">Data Quality Score</h4>
            <p className="text-sm text-gray-500">Overall assessment of your data</p>
          </div>
          <div className="text-right">
            <div className="text-4xl font-bold text-gray-900">{quality.overall_score}%</div>
            <span className={`text-sm font-medium ${getRatingColor(quality.rating)}`}>
              {quality.rating}
            </span>
          </div>
        </div>
        
        {/* Progress Ring */}
        <div className="flex justify-center mb-6">
          <svg className="w-32 h-32 transform -rotate-90">
            <circle
              cx="64"
              cy="64"
              r="56"
              fill="none"
              stroke="#E5E7EB"
              strokeWidth="8"
            />
            <circle
              cx="64"
              cy="64"
              r="56"
              fill="none"
              stroke={quality.overall_score >= 95 ? '#10B981' : quality.overall_score >= 80 ? '#3B82F6' : quality.overall_score >= 60 ? '#F59E0B' : '#EF4444'}
              strokeWidth="8"
              strokeDasharray={`${quality.overall_score * 3.52} ${352 - quality.overall_score * 3.52}`}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute flex items-center justify-center">
            <Award className="w-8 h-8 text-gray-400" />
          </div>
        </div>
        
        {/* Column Quality Table */}
        <div className="max-h-48 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-gray-500 uppercase bg-gray-100 sticky top-0">
              <tr>
                <th className="p-2 text-left">Column</th>
                <th className="p-2 text-right">Missing</th>
                <th className="p-2 text-right">Unique</th>
                <th className="p-2 text-right">Score</th>
              </tr>
            </thead>
            <tbody>
              {quality.columns?.slice(0, 10).map((col, i) => (
                <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="p-2 font-medium text-gray-700 truncate max-w-32" title={col.column}>
                    {col.column}
                  </td>
                  <td className="p-2 text-right text-gray-600">{col.missing_pct}%</td>
                  <td className="p-2 text-right text-gray-600">{col.unique_values}</td>
                  <td className="p-2 text-right">
                    <span className={`font-medium ${
                      col.quality_score >= 95 ? 'text-green-600' :
                      col.quality_score >= 80 ? 'text-blue-600' :
                      col.quality_score >= 60 ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {col.quality_score}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Statistics Card Component
  const StatisticsCard = ({ column, stats }) => {
    if (!stats?.statistics) return null;
    
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-4">
        <h5 className="font-medium text-gray-800 mb-3 truncate" title={column}>{column}</h5>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-500">Mean:</span>
            <span className="font-medium">{stats.statistics.mean}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Median:</span>
            <span className="font-medium">{stats.statistics['50%']}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Std:</span>
            <span className="font-medium">{stats.statistics.std}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Range:</span>
            <span className="font-medium">{stats.statistics.min} - {stats.statistics.max}</span>
          </div>
        </div>
        
        {/* Mini box plot visualization */}
        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="flex items-center h-6 bg-gray-100 rounded relative">
            <div
              className="absolute h-full bg-blue-200 rounded"
              style={{
                left: `${((stats.statistics['25%'] - stats.statistics.min) / (stats.statistics.max - stats.statistics.min)) * 100}%`,
                width: `${((stats.statistics['75%'] - stats.statistics['25%']) / (stats.statistics.max - stats.statistics.min)) * 100}%`
              }}
            />
            <div
              className="absolute w-0.5 h-full bg-blue-600"
              style={{
                left: `${((stats.statistics['50%'] - stats.statistics.min) / (stats.statistics.max - stats.statistics.min)) * 100}%`
              }}
            />
          </div>
        </div>
        
        {/* Outlier info */}
        {stats.outliers && stats.outliers.count > 0 && (
          <div className="mt-2 flex items-center gap-1 text-xs text-yellow-600">
            <AlertTriangle className="w-3 h-3" />
            <span>{stats.outliers.count} outliers ({stats.outliers.percentage}%)</span>
          </div>
        )}
      </div>
    );
  };

  // Main Render
  if (!taskId) {
    return (
      <div className="p-6">
        <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-lg p-8">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-gray-800">Data Analytics Dashboard</h2>
            <p className="text-gray-500 mt-2">Select a processed dataset to analyze</p>
          </div>
          
          {availableDatasets.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <FileText className="w-12 h-12 mx-auto mb-4 text-gray-300" />
              <p>No processed datasets available.</p>
              <p className="text-sm mt-2">Complete a preprocessing task first.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {availableDatasets.map((dataset, i) => (
                <button
                  key={i}
                  onClick={() => setTaskId(dataset.task_id)}
                  className="w-full p-4 bg-gray-50 hover:bg-blue-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-all text-left group"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-800 group-hover:text-blue-700">
                        Task: {dataset.task_id}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        {dataset.rows?.toLocaleString()} rows · {dataset.columns} columns · {dataset.file_size_mb} MB
                      </p>
                    </div>
                    <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-blue-600 transition-colors" />
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Generating dashboard...</p>
          <p className="text-sm text-gray-400 mt-1">Analyzing your data</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="max-w-lg mx-auto bg-red-50 rounded-xl p-6 text-center">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-red-700">Failed to Load Dashboard</h3>
          <p className="text-red-600 mt-2">{error}</p>
          <button
            onClick={() => setTaskId('')}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Select Another Dataset
          </button>
        </div>
      </div>
    );
  }

  if (!dashboard) return null;

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            Data Analytics Dashboard
          </h1>
          <p className="text-gray-500 mt-1">Task: {taskId}</p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Target Column Selector */}
          <select
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {dashboard.dataset_info?.columns?.map((col, i) => (
              <option key={i} value={col}>{col}</option>
            ))}
          </select>
          
          <button
            onClick={fetchDashboard}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
          
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg hover:from-purple-700 hover:to-indigo-700 transition-all shadow-md"
          >
            <Brain className="w-4 h-4" />
            Train Model
          </button>
          
          <button
            onClick={() => setTaskId('')}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
          >
            Change Dataset
          </button>
        </div>
      </div>

      {/* Executive Summary */}
      <div className="bg-white rounded-xl shadow-md mb-6 overflow-hidden">
        <SectionHeader
          title="Executive Summary"
          icon={<BookOpen className="w-5 h-5 text-blue-600" />}
          section="summary"
        />
        {expandedSections.summary && dashboard.executive_summary && (
          <div className="p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-3">
              {dashboard.executive_summary.headline}
            </h2>
            <p className="text-gray-600 leading-relaxed mb-4">
              {dashboard.executive_summary.narrative}
            </p>
            
            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl">
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">
                  {dashboard.dataset_info?.total_rows?.toLocaleString()}
                </p>
                <p className="text-xs text-gray-500">Total Records</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-purple-600">
                  {dashboard.dataset_info?.total_columns}
                </p>
                <p className="text-xs text-gray-500">Features</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">
                  {dashboard.executive_summary.data_completeness}%
                </p>
                <p className="text-xs text-gray-500">Completeness</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-orange-600">
                  {dashboard.dataset_info?.memory_usage_mb} MB
                </p>
                <p className="text-xs text-gray-500">Memory Usage</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* KPI Cards */}
      <div className="bg-white rounded-xl shadow-md mb-6 overflow-hidden">
        <SectionHeader
          title="Key Metrics"
          icon={<Target className="w-5 h-5 text-green-600" />}
          section="kpis"
          badge={dashboard.kpis?.length}
        />
        {expandedSections.kpis && dashboard.kpis && (
          <div className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {dashboard.kpis.map((kpi, i) => (
                <KPICard key={i} kpi={kpi} />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Data Quality & Distributions Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Data Quality */}
        <div className="bg-white rounded-xl shadow-md overflow-hidden">
          <SectionHeader
            title="Data Quality"
            icon={<CheckCircle className="w-5 h-5 text-green-600" />}
            section="quality"
          />
          {expandedSections.quality && (
            <div className="p-6">
              <DataQualityCard quality={dashboard.data_quality} />
            </div>
          )}
        </div>

        {/* Target Distribution */}
        {dashboard.charts?.target_pie && (
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <SectionHeader
              title="Target Distribution"
              icon={<PieChart className="w-5 h-5 text-purple-600" />}
              section="distributions"
            />
            {expandedSections.distributions && (
              <div className="p-6">
                <PieChartComponent
                  data={dashboard.charts.target_pie}
                  title={`${targetColumn} Distribution`}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Correlations */}
      <div className="bg-white rounded-xl shadow-md mb-6 overflow-hidden">
        <SectionHeader
          title="Correlation Analysis"
          icon={<Activity className="w-5 h-5 text-indigo-600" />}
          section="correlations"
        />
        {expandedSections.correlations && dashboard.correlations && (
          <div className="p-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <CorrelationHeatmap correlations={dashboard.correlations} />
              <TopCorrelations correlations={dashboard.correlations} />
            </div>
          </div>
        )}
      </div>

      {/* Categorical Analysis */}
      {Object.keys(dashboard.categorical_analysis || {}).length > 0 && (
        <div className="bg-white rounded-xl shadow-md mb-6 overflow-hidden">
          <SectionHeader
            title="Categorical Features"
            icon={<Tag className="w-5 h-5 text-pink-600" />}
            section="categorical"
            badge={Object.keys(dashboard.categorical_analysis).length}
          />
          {expandedSections.categorical && (
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(dashboard.categorical_analysis).slice(0, 6).map(([col, data]) => (
                  <BarChart
                    key={col}
                    data={data.chart_data}
                    title={col}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Numerical Analysis - Histograms */}
      {dashboard.charts?.histograms && Object.keys(dashboard.charts.histograms).length > 0 && (
        <div className="bg-white rounded-xl shadow-md mb-6 overflow-hidden">
          <SectionHeader
            title="Numerical Distributions"
            icon={<BarChart3 className="w-5 h-5 text-blue-600" />}
            section="distributions"
            badge={Object.keys(dashboard.charts.histograms).length}
          />
          {expandedSections.distributions && (
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(dashboard.charts.histograms).slice(0, 6).map(([col, data]) => (
                  <Histogram
                    key={col}
                    data={data}
                    title={col}
                  />
                ))}
              </div>
              
              {/* Statistics Cards */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(dashboard.numerical_analysis || {}).slice(0, 8).map(([col, stats]) => (
                  <StatisticsCard key={col} column={col} stats={stats} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Insights */}
      {dashboard.insights && dashboard.insights.length > 0 && (
        <div className="bg-white rounded-xl shadow-md mb-6 overflow-hidden">
          <SectionHeader
            title="Key Insights"
            icon={<Lightbulb className="w-5 h-5 text-yellow-600" />}
            section="insights"
            badge={dashboard.insights.length}
          />
          {expandedSections.insights && (
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {dashboard.insights.map((insight, i) => (
                  <InsightCard key={i} insight={insight} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recommendations */}
      {dashboard.recommendations && dashboard.recommendations.length > 0 && (
        <div className="bg-white rounded-xl shadow-md mb-6 overflow-hidden">
          <SectionHeader
            title="Recommendations"
            icon={<Sparkles className="w-5 h-5 text-purple-600" />}
            section="recommendations"
            badge={dashboard.recommendations.length}
          />
          {expandedSections.recommendations && (
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {dashboard.recommendations.map((rec, i) => (
                  <RecommendationCard key={i} recommendation={rec} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div className="bg-white rounded-xl shadow-md mb-6 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Next Steps</h3>
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg hover:from-purple-700 hover:to-indigo-700 transition-all shadow-md font-medium"
          >
            <Brain className="w-5 h-5" />
            Train a Model
          </button>
          
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-lg hover:from-gray-700 hover:to-gray-800 transition-all shadow-md font-medium"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Home
          </button>
        </div>
        <p className="text-sm text-gray-500 mt-3">
          Ready to build a model? Click "Train a Model" to proceed with model training based on this data analysis.
        </p>
      </div>

      {/* Footer */}
      <div className="text-center text-sm text-gray-400 mt-8 pb-4">
        Dashboard generated at {new Date(dashboard.generated_at).toLocaleString()}
      </div>
    </div>
  );
};

export default DataDashboard;
