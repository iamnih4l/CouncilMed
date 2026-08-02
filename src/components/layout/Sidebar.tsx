import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Brain, ActivitySquare, Bone, Settings, FileText, Hexagon } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const navItems = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'MRI (Brain & Spine)', path: '/mri', icon: Brain },
  { name: 'CT (Abdominal)', path: '/ct', icon: ActivitySquare },
  { name: 'X-Ray (Skeletal)', path: '/xray', icon: Bone },
  { name: 'Reports', path: '/reports', icon: FileText },
];

export default function Sidebar() {
  return (
    <aside className="w-64 hidden lg:flex flex-col border-r border-white/5 bg-black/20 backdrop-blur-xl h-full z-10 transition-all duration-300 relative">
      <div className="p-6 flex items-center space-x-3 mb-6">
        <div className="relative flex items-center justify-center p-2 rounded-xl bg-gradient-to-br from-[var(--color-accent-cyan)] to-[var(--color-accent-teal)]">
          <Hexagon className="w-6 h-6 text-white absolute animate-pulse opacity-50" size={28} />
          <Hexagon className="w-6 h-6 text-white relative z-10" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center">
            Council<span className="text-[var(--color-accent-cyan)]">AI</span>
          </h1>
          <p className="text-[10px] text-zinc-400 font-medium tracking-widest uppercase mt-0.5">Medical Diagnostic</p>
        </div>
      </div>

      <nav className="flex-1 px-4 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              twMerge(
                clsx(
                  'flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 text-sm font-medium',
                  isActive
                    ? 'bg-white/10 text-[var(--color-accent-cyan)] border border-white/10 shadow-[0_0_15px_rgba(6,182,212,0.15)]'
                    : 'text-zinc-400 hover:bg-white/5 hover:text-white'
                )
              )
            }
          >
            <item.icon className="w-5 h-5" />
            <span>{item.name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 mt-auto mb-4">
        <NavLink
            to="/settings"
            className={({ isActive }) =>
              twMerge(
                clsx(
                  'flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 text-sm font-medium',
                  isActive
                    ? 'bg-white/10 text-[var(--color-accent-cyan)] border border-white/10 shadow-[0_0_15px_rgba(6,182,212,0.15)]'
                    : 'text-zinc-400 hover:bg-white/5 hover:text-white'
                )
              )
            }
          >
            <Settings className="w-5 h-5" />
            <span>System Settings</span>
        </NavLink>
      </div>
    </aside>
  );
}
