import React, { useState } from 'react';
import { Upload, FileText, Loader, FileSpreadsheet, CheckCircle, X } from 'lucide-react';
import { uploadDataset } from '../../api/client';

const Uploader = ({ onUploadComplete }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const result = await uploadDataset(file);
      onUploadComplete(result);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const getFileIcon = () => {
    const ext = file?.name?.split('.').pop()?.toLowerCase();
    if (ext === 'csv') return <FileSpreadsheet className="text-success-500" size={48} />;
    if (ext === 'xlsx' || ext === 'xls') return <FileSpreadsheet className="text-success-500" size={48} />;
    if (ext === 'json') return <FileText className="text-primary-500" size={48} />;
    return <FileText className="text-on-surface-variant" size={48} />;
  };

  return (
    <div className="section-card-elevated p-8">
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl mb-4" style={{ background: 'rgba(77, 142, 255, 0.12)' }}>
          <Upload className="text-primary-500" size={24} />
        </div>
        <h2 className="text-2xl font-bold text-on-surface">Upload Dataset</h2>
        <p className="text-on-surface-variant mt-2">Select a data file to begin the analysis workflow</p>
      </div>
      
      <div
        className={`upload-zone relative ${
          dragActive
            ? 'upload-zone-active scale-[1.01]'
            : file 
              ? '' 
              : ''
        }`}
        style={file ? { borderColor: 'rgba(16, 185, 129, 0.3)', background: 'rgba(16, 185, 129, 0.05)' } : {}}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {file ? (
          <div className="space-y-4">
            <div className="w-20 h-20 mx-auto rounded-2xl flex items-center justify-center" style={{ background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.15)' }}>
              {getFileIcon()}
            </div>
            <div>
              <p className="font-semibold text-on-surface text-lg">{file.name}</p>
              <p className="text-sm text-on-surface-variant mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
            <div className="flex items-center justify-center space-x-2">
              <span className="inline-flex items-center px-3 py-1.5 rounded-lg text-sm font-medium" style={{ background: 'rgba(16, 185, 129, 0.12)', color: '#6ee7b7' }}>
                <CheckCircle size={14} className="mr-1.5" />
                Ready to upload
              </span>
            </div>
            <button
              onClick={() => setFile(null)}
              className="inline-flex items-center text-sm text-on-surface-variant hover:text-error-400 transition-colors font-medium"
            >
              <X size={16} className="mr-1" />
              Remove file
            </button>
          </div>
        ) : (
          <div className="space-y-5">
            <div className="w-20 h-20 mx-auto rounded-2xl flex items-center justify-center" style={{ background: 'var(--color-surface-high)', border: '1px solid var(--color-border)' }}>
              <Upload className={`transition-colors duration-200 ${dragActive ? 'text-primary-500' : 'text-on-surface-variant'}`} size={36} />
            </div>
            <div>
              <p className="text-lg font-semibold text-on-surface">Drag and drop your dataset here</p>
              <p className="text-sm text-on-surface-variant mt-2">or click the button below to browse files</p>
            </div>
            <label className="inline-block">
              <input type="file" className="hidden" accept=".csv,.xlsx,.xls,.json" onChange={handleFileChange} />
              <span className="btn-primary cursor-pointer inline-flex items-center">
                <Upload size={18} className="mr-2" />
                Browse Files
              </span>
            </label>
            <div className="flex items-center justify-center space-x-3 pt-2">
              <span className="inline-flex items-center px-3 py-1 rounded-md text-xs font-medium" style={{ background: 'var(--color-surface-high)', color: 'var(--color-text-secondary)' }}>CSV</span>
              <span className="inline-flex items-center px-3 py-1 rounded-md text-xs font-medium" style={{ background: 'var(--color-surface-high)', color: 'var(--color-text-secondary)' }}>Excel</span>
              <span className="inline-flex items-center px-3 py-1 rounded-md text-xs font-medium" style={{ background: 'var(--color-surface-high)', color: 'var(--color-text-secondary)' }}>JSON</span>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-5 p-4 rounded-xl flex items-start space-x-3" style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
          <div className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5" style={{ background: 'rgba(239, 68, 68, 0.15)' }}>
            <X size={12} className="text-error-400" />
          </div>
          <p className="text-error-400 text-sm">{error}</p>
        </div>
      )}

      {file && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="mt-6 w-full btn-primary flex items-center justify-center"
        >
          {uploading ? (
            <><Loader className="animate-spin mr-2" size={20} />Processing Dataset...</>
          ) : (
            <><Upload size={20} className="mr-2" />Upload and Analyze Dataset</>
          )}
        </button>
      )}
    </div>
  );
};

export default Uploader;
