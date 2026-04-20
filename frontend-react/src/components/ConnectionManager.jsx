import React, { useState } from 'react';
import { Settings, Wifi, WifiOff, Globe } from 'lucide-react';

const ConnectionManager = ({ backendUrl, setBackendUrl, connectionStatus }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="fixed top-4 right-4 z-50">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`p-2 rounded-full glass-card hover:bg-surface-high transition-colors ${
          connectionStatus === 'error' ? 'text-error shadow-[0_0_10px_rgba(255,180,171,0.5)]' : 'text-primary'
        }`}
      >
        <Settings size={20} />
      </button>

      {isOpen && (
        <div className="absolute top-12 right-0 w-72 glass-card p-4 animate-in fade-in slide-in-from-top-2">
          <div className="flex items-center gap-2 mb-4">
            <Globe size={16} className="text-on-surface-muted" />
            <h3 className="text-sm font-bold tracking-tight">Backend Configuration</h3>
          </div>
          
          <div className="space-y-3">
            <div>
              <label className="text-[10px] uppercase tracking-widest text-on-surface-muted block mb-1">API Endpoint</label>
              <input 
                type="text" 
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                placeholder="http://192.168.1.5:5000"
                className="w-full bg-bg border border-outline-variant rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-primary transition-colors"
              />
            </div>

            <div className="pt-2 flex items-center justify-between border-t border-outline-variant/30">
              <span className="text-[10px] uppercase tracking-widest text-on-surface-muted">Status</span>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-medium ${connectionStatus === 'ok' ? 'text-tertiary' : connectionStatus === 'error' ? 'text-error' : 'text-on-surface-muted'}`}>
                  {connectionStatus === 'ok' ? 'Linked' : connectionStatus === 'error' ? 'Disconnected' : 'Ready'}
                </span>
                {connectionStatus === 'ok' ? <Wifi size={14} className="text-tertiary" /> : <WifiOff size={14} className="text-error" />}
              </div>
            </div>

            <p className="text-[9px] text-on-surface-muted leading-relaxed">
              Tip: If testing on mobile, enter your laptop's Local IP (e.g., http://192.168.1.XX:5000) or use VS Code port forwarding.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConnectionManager;
