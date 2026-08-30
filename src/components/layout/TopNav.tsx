/**
 * CouncilMed — Top Navigation Bar
 * ===================================
 * Shows system status indicator, search, and quick actions.
 * Removed hardcoded "Dr. Sarah Jenkins" and fake alert count.
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Cpu, Scan, Settings, Wifi, WifiOff, Loader2 } from 'lucide-react';
import { api } from '../../services/api';

export default function TopNav() {
  const navigate = useNavigate();
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [device, setDevice] = useState<string>('');

  useEffect(() => {
    async function checkHealth() {
      try {
        const health = await api.health();
        setBackendOnline(true);
        setDevice(health.device || 'cpu');
      } catch {
        setBackendOnline(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-14 border-b border-white/5 bg-black/20 backdrop-blur-md flex items-center justify-between px-4 lg:px-8 shrink-0 z-20 sticky top-0">
      {/* Left: Search */}
      <div className="flex items-center gap-3">
        <div className="hidden md:flex relative group">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-600 group-focus-within:text-cyan-400 transition-colors" />
          <input
            type="text"
            placeholder="Search analyses, models..."
            className="w-56 bg-white/[0.03] border border-white/5 rounded-lg py-1.5 pl-9 pr-4 text-xs text-zinc-300 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-cyan-500/30 focus:bg-white/[0.05] transition-all"
          />
        </div>
      </div>

      {/* Right: Status + Actions */}
      <div className="flex items-center gap-3">
        {/* Backend status */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-white/[0.03] border border-white/5">
          {backendOnline === null ? (
            <Loader2 className="w-3 h-3 text-zinc-500 animate-spin" />
          ) : backendOnline ? (
            <Wifi className="w-3 h-3 text-emerald-400" />
          ) : (
            <WifiOff className="w-3 h-3 text-red-400" />
          )}
          <span className="text-[10px] text-zinc-500">
            {backendOnline === null ? 'Connecting...' : backendOnline ? 'Online' : 'Offline'}
          </span>
          {device && backendOnline && (
            <span className="text-[10px] text-zinc-600 flex items-center gap-0.5 ml-1 pl-1 border-l border-white/5">
              <Cpu className="w-2.5 h-2.5" /> {device}
            </span>
          )}
        </div>

        {/* New Analysis */}
        <button
          onClick={() => navigate('/analyze')}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 text-cyan-400 rounded-lg transition-colors text-xs font-medium"
        >
          <Scan className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">New Analysis</span>
        </button>

        {/* Settings */}
        <button
          onClick={() => navigate('/settings')}
          className="p-2 text-zinc-500 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
