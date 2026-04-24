import React, { useState } from 'react';
import { Copy, Check, Code } from 'lucide-react';

/**
 * Syntax code block with copy button.
 * @param {string} code - Code content
 * @param {string} language - Language label
 * @param {string} title - Block title
 */
const CodeBlock = ({ code, language = '', title = '' }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-xl overflow-hidden border border-outline-variant">
      <div className="flex items-center justify-between px-4 py-2.5 bg-surface-container-low border-b border-outline-variant">
        <div className="flex items-center gap-2">
          <Code size={14} className="text-primary-500" />
          <span className="text-xs font-semibold text-on-surface">{title || language}</span>
        </div>
        <button
          onClick={handleCopy}
          className="p-1.5 rounded-md hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface transition-colors"
          aria-label="Copy code"
        >
          {copied ? <Check size={14} className="text-success-500" /> : <Copy size={14} />}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto bg-surface-container-lowest text-sm font-mono text-on-surface-variant leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;
