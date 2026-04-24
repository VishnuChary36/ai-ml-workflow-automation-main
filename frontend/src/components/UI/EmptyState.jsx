import React from 'react';
import { Inbox } from 'lucide-react';

/**
 * Unified empty state component.
 * @param {React.ReactNode} icon - Lucide icon component
 * @param {string} title
 * @param {string} description
 * @param {React.ReactNode} action - Optional CTA button
 */
const EmptyState = ({ icon: Icon = Inbox, title = 'Nothing here yet', description, action }) => {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      <div className="w-16 h-16 bg-surface-container-high rounded-2xl flex items-center justify-center mb-5">
        <Icon size={28} className="text-on-surface-variant opacity-50" />
      </div>
      <h3 className="text-lg font-semibold text-on-surface mb-1">{title}</h3>
      {description && <p className="text-sm text-on-surface-variant max-w-sm">{description}</p>}
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
};

export default EmptyState;
