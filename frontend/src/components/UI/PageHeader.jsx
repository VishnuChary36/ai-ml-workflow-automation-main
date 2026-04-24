import React from 'react';

/**
 * Consistent page/section header with icon, title, subtitle, and actions slot.
 * @param {React.ReactNode} icon - Lucide icon element (already sized + colored)
 * @param {string} title
 * @param {string} subtitle
 * @param {string} gradientFrom - Tailwind color for icon bg gradient start
 * @param {string} gradientTo - Tailwind color for icon bg gradient end
 * @param {React.ReactNode} actions - Right-side action buttons
 */
const PageHeader = ({
  icon,
  title,
  subtitle,
  gradientFrom = 'from-primary-400',
  gradientTo = 'to-primary-500',
  actions,
  children,
}) => {
  return (
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
      <div className="flex items-center gap-3">
        {icon && (
          <div className={`w-12 h-12 bg-gradient-to-br ${gradientFrom} ${gradientTo} rounded-xl flex items-center justify-center shadow-lg`}>
            {icon}
          </div>
        )}
        <div>
          <h1 className="text-2xl font-bold text-on-surface">{title}</h1>
          {subtitle && <p className="text-sm text-on-surface-variant mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {(actions || children) && (
        <div className="flex items-center gap-3">
          {actions || children}
        </div>
      )}
    </div>
  );
};

export default PageHeader;
