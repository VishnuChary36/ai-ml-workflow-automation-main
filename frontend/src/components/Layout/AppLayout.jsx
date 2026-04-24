import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

/**
 * Application shell: sidebar + content area.
 * All pages render inside <Outlet/> which gets the right margin offset.
 */
const AppLayout = () => {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      {/* Main content area — offset by sidebar width */}
      <div className="flex-1 lg:ml-60 min-h-screen transition-all duration-300">
        <Outlet />
      </div>
    </div>
  );
};

export default AppLayout;
