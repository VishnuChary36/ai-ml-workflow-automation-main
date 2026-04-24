import React from 'react';

/**
 * Unified loading spinner component.
 * @param {'sm'|'md'|'lg'} size
 * @param {string} message - Optional loading text
 * @param {boolean} fullPage - Center in viewport
 */
const SIZE_MAP = { sm: 'w-5 h-5 border-2', md: 'w-10 h-10 border-3', lg: 'w-14 h-14 border-4' };

const LoadingSpinner = ({ size = 'md', message = null, fullPage = false }) => {
  const spinner = (
    <div className="flex flex-col items-center justify-center gap-3">
      <div className={`${SIZE_MAP[size]} border-primary-500 border-t-transparent rounded-full animate-spin`} />
      {message && <p className="text-sm text-on-surface-variant font-medium animate-pulse">{message}</p>}
    </div>
  );

  if (fullPage) {
    return <div className="flex items-center justify-center min-h-[60vh]">{spinner}</div>;
  }
  return spinner;
};

export default LoadingSpinner;
