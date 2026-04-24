import React from 'react';

const STATUS_STYLES = {
  online: {
    dot: 'bg-success-500 animate-pulse-subtle',
    bg: 'bg-success-500/10',
    border: 'border-success-500/20',
    text: 'text-success-400',
    label: 'Online',
  },
  offline: {
    dot: 'bg-on-surface-variant',
    bg: 'bg-surface-container-highest',
    border: 'border-outline-variant',
    text: 'text-on-surface-variant',
    label: 'Offline',
  },
  error: {
    dot: 'bg-error-400',
    bg: 'bg-error-500/10',
    border: 'border-error-500/20',
    text: 'text-error-400',
    label: 'Error',
  },
  warning: {
    dot: 'bg-warning-400',
    bg: 'bg-warning-500/10',
    border: 'border-warning-500/20',
    text: 'text-warning-400',
    label: 'Warning',
  },
  active: {
    dot: 'bg-success-500 animate-pulse-subtle',
    bg: 'bg-success-500/10',
    border: 'border-success-500/20',
    text: 'text-success-400',
    label: 'System Active',
  },
  connected: {
    dot: 'bg-success-500 animate-pulse-subtle',
    bg: 'bg-success-500/10',
    border: 'border-success-500/20',
    text: 'text-success-400',
    label: 'Connected',
  },
  disconnected: {
    dot: 'bg-on-surface-variant',
    bg: 'bg-surface-container-highest',
    border: 'border-outline-variant',
    text: 'text-on-surface-variant',
    label: 'Disconnected',
  },
};

/**
 * Status indicator badge.
 * @param {'online'|'offline'|'error'|'warning'|'active'|'connected'|'disconnected'} status
 * @param {string} label - Override default label
 */
const StatusBadge = ({ status = 'offline', label }) => {
  const style = STATUS_STYLES[status] || STATUS_STYLES.offline;
  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium ${style.bg} border ${style.border}`}>
      <span className={`w-2 h-2 rounded-full ${style.dot}`} style={status === 'online' || status === 'active' || status === 'connected' ? { boxShadow: '0 0 6px rgba(16, 185, 129, 0.5)' } : {}} />
      <span className={style.text}>{label || style.label}</span>
    </div>
  );
};

export default StatusBadge;
