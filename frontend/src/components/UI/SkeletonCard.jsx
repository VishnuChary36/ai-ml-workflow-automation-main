import React from 'react';

/**
 * Skeleton placeholder card for loading states.
 * @param {number} lines - Number of shimmer lines to show
 * @param {boolean} showIcon - Show a square icon placeholder
 */
const SkeletonCard = ({ lines = 3, showIcon = true }) => {
  return (
    <div className="section-card-elevated p-6 space-y-4 animate-pulse">
      {showIcon && (
        <div className="flex items-center space-x-4">
          <div className="w-14 h-14 bg-surface-container-highest rounded-xl" />
          <div className="space-y-2 flex-1">
            <div className="h-5 bg-surface-container-highest rounded w-2/5" />
            <div className="h-3 bg-surface-container-high rounded w-1/3" />
          </div>
        </div>
      )}
      <div className="space-y-3">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="shimmer-line"
            style={{ width: `${85 - i * 12}%`, animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  );
};

export default SkeletonCard;
