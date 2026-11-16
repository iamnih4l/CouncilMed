/**
 * CouncilMed — Executive HUD Navigation Bar
 * =========================================
 * Provides seamless navigation across all 5 workstation views:
 * - Analysis Workstation (/analyze or /)
 * - Clinical Reports (/reports)
 * - Model Registry (/models)
 * - Analysis History (/history)
 * - System Settings (/settings)
 *
 * Displays live backend telemetry, compute device, and audit readiness status.
 */

import { useState, useEffect } from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Hexagon, Scan, FileText, Database, Clock, Settings,
  Cpu, Wifi, WifiOff, Loader2, ShieldCheck, Plus
} from 'lucide-react';
import { api } from '../../services/api';

const navItems = [
  { name: 'Workstation', path: '/analyze', alternatePaths: ['/'], icon: Scan },
  { name: 'Reports', path: '/reports', icon: FileText },
  { name: 'Model Registry', path: '/models', icon: Database },
  { name: 'History', path: '/history', icon: Clock },
  { name: 'Settings', path: '/settings', icon: Settings },
];

export default function TopNav() {
  const navigate = useNavigate();
  const location = useLocation();
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [device, setDevice] = useState<string>('');
  const [modelsLoaded, setModelsLoaded] = useState<number>(9);

  useEffect(() => {
    async function checkHealth() {
      try {
        const health = await api.health();
        setBackendOnline(true);
        setDevice(health.device || 'cpu');
        if (health.models_loaded) {
          setModelsLoaded(health.models_loaded);
        }
      } catch {
        setBackendOnline(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 20000);
    return () => clearInterval(interval);
  }, []);

  const isCurrentActive = (item: typeof navItems[0]) => {
    if (location.pathname === item.path) return true;
    if (item.alternatePaths && item.alternatePaths.includes(location.pathname)) return true;
    return false;
  };

  return (
    <header className="h-16 border-b border-white/10 bg-black/70 backdrop-blur-2xl flex items-center justify-between px-4 lg:px-6 shrink-0 z-50 sticky top-0 shadow-[0_4px_30px_rgba(0,0,0,0.8)]">
      {/* Left: Brand Identity */}
      <div className="flex items-center gap-3">
        <NavLink
          to="/"
          className="flex items-center gap-3 group btn-tactile focus:outline-none"
        >
          <div className="relative flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500/20 to-teal-500/10 border border-cyan-500/30 group-hover:border-cyan-400/60 transition-colors shadow-[0_0_15px_rgba(0,240,255,0.2)]">
            <Hexagon className="w-5 h-5 text-cyan-400 absolute animate-pulse opacity-60" />
            <Hexagon className="w-5 h-5 text-cyan-300 relative z-10" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-base font-light tracking-[0.12em] text-white">
                COUNCIL<span className="font-semibold text-cyan-400 text-glow">MED</span>
              </span>
              <span className="hidden sm:inline-block px-1.5 py-0.5 rounded text-[9px] font-mono tracking-widest uppercase bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                v0.1.0
              </span>
            </div>
            <p className="text-[10px] text-zinc-400 font-mono tracking-wider uppercase hidden sm:block">
              Clinical Multi-Model AI
            </p>
          </div>
        </NavLink>
      </div>

      {/* Center: HUD Navigation Pills */}
      <nav className="flex items-center gap-1 sm:gap-1.5 p-1 rounded-xl bg-white/[0.03] border border-white/10 backdrop-blur-md max-w-full overflow-x-auto">
        {navItems.map((item) => {
          const active = isCurrentActive(item);
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={`relative flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono tracking-wider uppercase transition-all duration-200 btn-tactile focus:outline-none whitespace-nowrap ${
                active
                  ? 'text-cyan-300 font-semibold shadow-[0_0_15px_rgba(0,240,255,0.2)]'
                  : 'text-zinc-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {active && (
                <motion.div
                  layoutId="activeNavIndicator"
                  className="absolute inset-0 rounded-lg bg-gradient-to-r from-cyan-500/20 to-teal-500/10 border border-cyan-500/40"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <Icon className={`w-3.5 h-3.5 relative z-10 ${active ? 'text-cyan-400' : 'text-zinc-400'}`} />
              <span className="relative z-10 hidden md:inline">{item.name}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Right: Live Telemetry & Actions */}
      <div className="flex items-center gap-2.5">
        {/* Backend & Compute Telemetry */}
        <div className="hidden sm:flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-white/[0.03] border border-white/10 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            {backendOnline === null ? (
              <Loader2 className="w-3 h-3 text-zinc-500 animate-spin" />
            ) : backendOnline ? (
              <div className="flex items-center gap-1">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <Wifi className="w-3 h-3 text-emerald-400" />
              </div>
            ) : (
              <WifiOff className="w-3 h-3 text-red-400" />
            )}
            <span className={`text-[11px] ${backendOnline ? 'text-emerald-400' : 'text-zinc-500'}`}>
              {backendOnline === null ? 'CHECKING' : backendOnline ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>

          {backendOnline && (
            <div className="flex items-center gap-1.5 pl-2 border-l border-white/10 text-[10px] text-zinc-400">
              <Cpu className="w-3 h-3 text-cyan-400" />
              <span>{device.toUpperCase()}</span>
              <span className="text-zinc-600">•</span>
              <span className="text-zinc-400">{modelsLoaded} MODELS</span>
            </div>
          )}
        </div>

        {/* Audit Readiness Pill */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[11px] font-mono tracking-wider uppercase">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>AUDIT READY</span>
        </div>

        {/* Quick New Analysis Action Button */}
        <button
          onClick={() => navigate('/analyze')}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-cyan-500/20 to-teal-500/20 hover:from-cyan-500/30 hover:to-teal-500/30 border border-cyan-500/40 text-cyan-300 rounded-lg text-xs font-mono tracking-wider uppercase btn-tactile shadow-[0_0_15px_rgba(0,240,255,0.15)]"
        >
          <Plus className="w-3.5 h-3.5 text-cyan-400" />
          <span className="hidden xl:inline">New Study</span>
        </button>
      </div>
    </header>
  );
}
