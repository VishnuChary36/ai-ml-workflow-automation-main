import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  BarChart3,
  Database,
  Workflow,
  TrendingUp,
  Eye,
  Rocket,
  LayoutDashboard,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
} from 'lucide-react';
import StatusBadge from '../UI/StatusBadge';

const NAV_ITEMS = [
  {
    label: 'Dashboard',
    path: '/',
    icon: Database,
    description: 'Upload & Pipeline',
    exact: true,
  },
];

// These appear dynamically when a model is active
const MODEL_NAV_ITEMS = [
  {
    label: 'Visualizations',
    pathPrefix: '/visualizations',
    icon: BarChart3,
    description: 'Charts & Metrics',
  },
  {
    label: 'Explainability',
    pathPrefix: '/explainability',
    icon: Eye,
    description: 'SHAP & LIME',
  },
  {
    label: 'Deployment',
    pathPrefix: '/deploy',
    icon: Rocket,
    description: 'Deploy & Export',
  },
  {
    label: 'Grafana',
    pathPrefix: '/grafana',
    icon: LayoutDashboard,
    description: 'Live Dashboard',
  },
];

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  // Extract modelId from current path (e.g., /visualizations/abc123)
  const pathParts = location.pathname.split('/');
  const currentModelId = pathParts.length >= 3 ? pathParts[2] : null;
  const isOnModelPage = currentModelId && pathParts[1] !== '';

  const linkClass = (isActive) =>
    `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group ${
      isActive
        ? 'bg-primary-500/12 text-primary-500 shadow-sm'
        : 'text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface'
    }`;

  const sidebarContent = (
    <>
      {/* Logo */}
      <div className="flex items-center justify-between px-4 py-5 border-b border-outline-variant">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-primary-400 to-primary-500 rounded-xl flex items-center justify-center shadow-glow-sm flex-shrink-0">
            <Workflow className="text-white" size={18} />
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <h1 className="text-sm font-bold text-on-surface leading-tight">ML Workflow</h1>
              <p className="text-[10px] text-on-surface-variant">Platform</p>
            </div>
          )}
        </div>
        {/* Collapse toggle — desktop only */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="hidden lg:flex p-1.5 rounded-lg hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface transition-colors"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
        {/* Close button — mobile only */}
        <button
          onClick={() => setMobileOpen(false)}
          className="lg:hidden p-1.5 rounded-lg hover:bg-surface-container-high text-on-surface-variant"
          aria-label="Close menu"
        >
          <X size={18} />
        </button>
      </div>

      {/* Nav Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {/* Main Links */}
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.exact}
            className={({ isActive }) => linkClass(isActive)}
            onClick={() => setMobileOpen(false)}
          >
            <item.icon size={18} className="flex-shrink-0" />
            {!collapsed && (
              <div className="min-w-0">
                <span className="block truncate">{item.label}</span>
                <span className="block text-[10px] text-on-surface-variant truncate opacity-70 group-hover:opacity-100">
                  {item.description}
                </span>
              </div>
            )}
          </NavLink>
        ))}

        {/* Model pages — only visible when navigated to a model */}
        {isOnModelPage && (
          <>
            <div className="pt-4 pb-2">
              {!collapsed && (
                <span className="px-3 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">
                  Model Analysis
                </span>
              )}
              {collapsed && <div className="border-t border-outline-variant mx-2" />}
            </div>
            {MODEL_NAV_ITEMS.map((item) => {
              const path = `${item.pathPrefix}/${currentModelId}`;
              const isActive = location.pathname.startsWith(item.pathPrefix);
              return (
                <NavLink
                  key={path}
                  to={path}
                  className={() => linkClass(isActive)}
                  onClick={() => setMobileOpen(false)}
                >
                  <item.icon size={18} className="flex-shrink-0" />
                  {!collapsed && (
                    <div className="min-w-0">
                      <span className="block truncate">{item.label}</span>
                      <span className="block text-[10px] text-on-surface-variant truncate opacity-70 group-hover:opacity-100">
                        {item.description}
                      </span>
                    </div>
                  )}
                </NavLink>
              );
            })}
          </>
        )}
      </nav>

      {/* Footer Status */}
      <div className="px-4 py-4 border-t border-outline-variant">
        {!collapsed ? (
          <StatusBadge status="active" />
        ) : (
          <div className="flex justify-center">
            <span className="w-2.5 h-2.5 rounded-full bg-success-500 animate-pulse-subtle" style={{ boxShadow: '0 0 6px rgba(16, 185, 129, 0.5)' }} />
          </div>
        )}
      </div>
    </>
  );

  return (
    <>
      {/* Mobile hamburger */}
      <button
        onClick={() => setMobileOpen(true)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2.5 rounded-xl bg-surface-container border border-outline-variant shadow-elevated text-on-surface-variant"
        aria-label="Open menu"
      >
        <Menu size={20} />
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar panel */}
      <aside
        className={`
          fixed top-0 left-0 z-50 h-screen flex flex-col
          bg-surface-container border-r border-outline-variant
          transition-all duration-300 ease-in-out
          ${collapsed ? 'w-[72px]' : 'w-60'}
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {sidebarContent}
      </aside>
    </>
  );
};

export default Sidebar;
