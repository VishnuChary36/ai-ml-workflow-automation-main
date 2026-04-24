import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Download, Copy, Play, Terminal, CheckCircle, Circle } from 'lucide-react';
import LogWebSocket from '../../api/websocket';
import { downloadLogsText } from '../../api/client';

// Create a unique key for log deduplication
const getLogKey = (log) => {
  return `${log.timestamp || ''}-${log.message || ''}-${log.source || ''}`;
};

const Console = ({ taskId, autoConnect = true, showPreview = false, previewLogs = [] }) => {
  const [logs, setLogs] = useState(showPreview ? previewLogs : []);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const consoleRef = useRef(null);
  const wsRef = useRef(null);
  const isMountedRef = useRef(false);
  const seenLogsRef = useRef(new Set());

  // Add log with deduplication
  const addLog = useCallback((logEntry) => {
    if (!isMountedRef.current) return;
    
    const logKey = getLogKey(logEntry);
    if (seenLogsRef.current.has(logKey)) {
      return; // Skip duplicate
    }
    seenLogsRef.current.add(logKey);
    setLogs((prev) => [...prev, logEntry]);
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    // Reset seen logs when taskId changes
    seenLogsRef.current = new Set();
    setLogs([]);
    
    if (!showPreview && autoConnect && taskId) {
      // Small delay to avoid React Strict Mode double-mount issues
      const timeoutId = setTimeout(() => {
        if (isMountedRef.current) {
          connectWebSocket();
        }
      }, 150);
      
      return () => {
        clearTimeout(timeoutId);
        isMountedRef.current = false;
        if (wsRef.current) {
          wsRef.current.disconnect();
          wsRef.current = null;
        }
      };
    }

    return () => {
      isMountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.disconnect();
        wsRef.current = null;
      }
    };
  }, [taskId, autoConnect, showPreview]);

  useEffect(() => {
    // Auto-scroll to bottom
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [logs]);

  const connectWebSocket = () => {
    if (!isMountedRef.current) return;
    
    if (wsRef.current) {
      wsRef.current.disconnect();
    }

    wsRef.current = new LogWebSocket(
      taskId,
      addLog,
      (err) => {
        if (isMountedRef.current) {
          setError('Connection interrupted. Attempting to reconnect...');
          setIsConnected(false);
        }
      },
      () => {
        if (isMountedRef.current) {
          setIsConnected(false);
        }
      }
    );

    wsRef.current.connect();
    setIsConnected(true);
    setError(null);
  };

  const formatLogLine = (log) => {
    const timestamp = log.timestamp ? new Date(log.timestamp).toISOString().replace('T', ' ').substring(11, 19) : '';
    const level = log.level || 'INFO';
    const source = log.source || '';
    const message = log.message || '';

    return { timestamp, level, source, message };
  };

  const getLevelColor = (level) => {
    switch (level?.toUpperCase()) {
      case 'ERROR':
        return 'text-error-400';
      case 'WARN':
        return 'text-warning-400';
      case 'INFO':
        return 'text-primary-400';
      case 'DEBUG':
        return 'text-on-surface-variant';
      default:
        return 'text-on-surface-variant';
    }
  };

  const getLevelBadgeColor = (level) => {
    switch (level?.toUpperCase()) {
      case 'ERROR':
        return 'bg-error-500/20 text-error-400';
      case 'WARN':
        return 'bg-warning-500/20 text-warning-400';
      case 'INFO':
        return 'bg-primary-500/20 text-primary-400';
      case 'DEBUG':
        return 'bg-surface-container-high/20 text-on-surface-variant';
      default:
        return 'bg-surface-container-high/20 text-on-surface-variant';
    }
  };

  const handleCopyLogs = () => {
    const logsText = logs.map(log => {
      const { timestamp, level, source, message } = formatLogLine(log);
      return `[${timestamp}] ${level} | ${source} | ${message}`;
    }).join('\n');
    navigator.clipboard.writeText(logsText);
  };

  const handleDownloadLogs = async () => {
    if (showPreview) {
      const logsText = logs.map(log => {
        const { timestamp, level, source, message } = formatLogLine(log);
        return `[${timestamp}] ${level} | ${source} | ${message}`;
      }).join('\n');
      const blob = new Blob([logsText], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'preview.log');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } else {
      await downloadLogsText(taskId);
    }
  };

  return (
    <div className="console-container overflow-hidden">
      {/* Header */}
      <div className="console-header">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Terminal size={18} className="text-on-surface-variant" />
            <span className="text-sm font-medium text-on-surface-variant">
              {showPreview ? 'Preview Console' : 'Execution Logs'}
            </span>
          </div>
          {!showPreview && (
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <div className="flex items-center space-x-1.5 px-2 py-1 bg-success-500/10 rounded-md">
                  <div className="w-1.5 h-1.5 bg-success-500 rounded-full animate-pulse-subtle"></div>
                  <span className="text-xs font-medium text-success-400">Connected</span>
                </div>
              ) : (
                <div className="flex items-center space-x-1.5 px-2 py-1 bg-surface-container-highest rounded-md">
                  <div className="w-1.5 h-1.5 bg-surface-container-low0 rounded-full"></div>
                  <span className="text-xs font-medium text-on-surface-variant">Disconnected</span>
                </div>
              )}
            </div>
          )}
          {error && (
            <span className="text-xs text-warning-400 font-medium">{error}</span>
          )}
        </div>
        
        <div className="flex items-center space-x-1">
          <button
            onClick={handleCopyLogs}
            className="p-2 hover:bg-surface-container-highest rounded-lg text-on-surface-variant hover:text-on-surface transition-colors"
            title="Copy logs"
          >
            <Copy size={16} />
          </button>
          <button
            onClick={handleDownloadLogs}
            className="p-2 hover:bg-surface-container-highest rounded-lg text-on-surface-variant hover:text-on-surface transition-colors"
            title="Download logs"
          >
            <Download size={16} />
          </button>
          {!showPreview && !isConnected && (
            <button
              onClick={connectWebSocket}
              className="p-2 hover:bg-surface-container-highest rounded-lg text-on-surface-variant hover:text-on-surface transition-colors"
              title="Reconnect"
            >
              <Play size={16} />
            </button>
          )}
        </div>
      </div>

      {/* Console Content */}
      <div
        ref={consoleRef}
        className="console-content bg-surface-container-lowest"
      >
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-on-surface-variant">
            <Terminal size={32} className="mb-3 opacity-50" />
            <p className="text-sm">
              {showPreview ? 'No preview available' : 'Waiting for logs...'}
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            {logs.map((log, index) => {
              const { timestamp, level, source, message } = formatLogLine(log);
              return (
                <div 
                  key={index} 
                  className="flex items-start space-x-3 py-1 px-2 rounded hover:bg-surface-container-high/50 transition-colors group"
                >
                  <span className="text-on-surface-variant text-xs font-mono flex-shrink-0 pt-0.5">
                    {timestamp}
                  </span>
                  <span className={`text-xs font-medium px-1.5 py-0.5 rounded flex-shrink-0 ${getLevelBadgeColor(level)}`}>
                    {level}
                  </span>
                  {source && (
                    <span className="text-on-surface-variant text-xs flex-shrink-0 pt-0.5">
                      {source}
                    </span>
                  )}
                  <span className={`${getLevelColor(level)} text-sm flex-1`}>
                    {message}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default Console;
