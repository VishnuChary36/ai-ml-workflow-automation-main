import React from 'react';
import { Clock, CheckCircle, XCircle, Loader } from 'lucide-react';
import EmptyState from '../UI/EmptyState';

const TaskList = ({ tasks = [] }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-success-500" size={20} />;
      case 'failed':
        return <XCircle className="text-error-400" size={20} />;
      case 'running':
        return <Loader className="text-primary-500 animate-spin" size={20} />;
      default:
        return <Clock className="text-on-surface-variant" size={20} />;
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return 'badge badge-success';
      case 'failed':
        return 'badge badge-error';
      case 'running':
        return 'badge badge-primary';
      default:
        return 'badge badge-slate';
    }
  };

  return (
    <div className="section-card-elevated p-6">
      <h2 className="text-xl font-bold text-on-surface mb-4">Recent Tasks</h2>
      
      {tasks.length === 0 ? (
        <EmptyState
          icon={Clock}
          title="No tasks yet"
          description="Tasks will appear here once you start processing data or training models."
        />
      ) : (
        <div className="space-y-3">
          {tasks.map((task) => (
            <div
              key={task.task_id}
              className="border border-outline-variant rounded-xl p-4 hover:border-primary-200 hover:bg-surface-container-low/50 transition-all duration-200"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3 flex-1 min-w-0">
                  {getStatusIcon(task.status)}
                  <div className="min-w-0">
                    <p className="font-semibold text-on-surface truncate">{task.task_id}</p>
                    <p className="text-sm text-on-surface-variant">{task.task_type}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3 flex-shrink-0">
                  <span className={getStatusBadge(task.status)}>
                    {task.status}
                  </span>
                  <span className="text-sm text-on-surface-variant">
                    {new Date(task.created_at).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TaskList;
