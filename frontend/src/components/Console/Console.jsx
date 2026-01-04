import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Download, Copy, XCircle, Play } from 'lucide-react';
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
          setError('WebSocket error. Trying to reconnect...');
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
    const timestamp = log.timestamp ? new Date(log.timestamp).toISOString().replace('T', ' ').substring(0, 19) : '';
    const level = log.level || 'INFO';
    const source = log.source || '';
    const message = log.message || '';

    return `${level} ${timestamp}Z | ${source} | ${message}`;
  };

  const getLevelColor = (level) => {
    switch (level?.toUpperCase()) {
      case 'ERROR':
        return 'text-red-400';
      case 'WARN':
        return 'text-yellow-400';
      case 'INFO':
        return 'text-blue-400';
      case 'DEBUG':
        return 'text-gray-400';
      default:
        return 'text-gray-300';
    }
  };

  const handleCopyLogs = () => {
    const logsText = logs.map(formatLogLine).join('\n');
    navigator.clipboard.writeText(logsText);
  };

  const handleDownloadLogs = async () => {
    if (showPreview) {
      const logsText = logs.map(formatLogLine).join('\n');
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
    <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* Header */}
      <div className="bg-gray-800 px-4 py-2 flex items-center justify-between border-b border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="text-sm font-mono text-gray-300">
            {showPreview ? 'Preview Console' : 'Live Console'}
          </div>
          {!showPreview && (
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <>
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-xs text-green-400">Connected</span>
                </>
              ) : (
                <>
                  <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                  <span className="text-xs text-red-400">Disconnected</span>
                </>
              )}
            </div>
          )}
          {error && <span className="text-xs text-yellow-400">{error}</span>}
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={handleCopyLogs}
            className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition"
            title="Copy logs"
          >
            <Copy size={16} />
          </button>
          <button
            onClick={handleDownloadLogs}
            className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition"
            title="Download logs"
          >
            <Download size={16} />
          </button>
          {!showPreview && !isConnected && (
            <button
              onClick={connectWebSocket}
              className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition"
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
        className="bg-gray-950 p-4 h-96 overflow-y-auto font-mono text-sm"
      >
        {logs.length === 0 ? (
          <div className="text-gray-500 text-center py-8">
            {showPreview ? 'No preview available' : 'Waiting for logs...'}
          </div>
        ) : (
          logs.map((log, index) => (
            <div key={index} className={`${getLevelColor(log.level)} py-0.5`}>
              {formatLogLine(log)}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Console;
