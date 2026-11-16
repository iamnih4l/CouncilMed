import React from 'react';
import { cn } from '../../utils/cn';
import { AlertTriangle, CheckCircle, Info } from 'lucide-react';

type StatusType = 'critical' | 'warning' | 'clear' | 'info';

interface StatusBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  status: StatusType;
  children?: React.ReactNode;
}

const statusConfig = {
  critical: {
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    glow: 'shadow-[0_0_12px_rgba(239,68,68,0.2)]',
    icon: AlertTriangle,
  },
  warning: {
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    glow: 'shadow-[0_0_12px_rgba(245,158,11,0.2)]',
    icon: AlertTriangle,
  },
  clear: {
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    glow: 'shadow-[0_0_12px_rgba(16,185,129,0.2)]',
    icon: CheckCircle,
  },
  info: {
    color: 'text-cyan-400',
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/30',
    glow: 'shadow-[0_0_12px_rgba(0,240,255,0.2)]',
    icon: Info,
  },
};

export function StatusBadge({ status, className, children, ...props }: StatusBadgeProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-semibold uppercase tracking-wider border backdrop-blur-md transition-all duration-300",
        config.color,
        config.bg,
        config.border,
        config.glow,
        className
      )}
      {...props}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{children || status}</span>
    </div>
  );
}
