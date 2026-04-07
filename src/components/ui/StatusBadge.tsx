import React from 'react';
import { cn } from './GlassCard';
import { AlertTriangle, CheckCircle, Info } from 'lucide-react';

type StatusType = 'critical' | 'warning' | 'clear' | 'info';

interface StatusBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  status: StatusType;
  children?: React.ReactNode;
}

const statusConfig = {
  critical: {
    color: 'text-[var(--color-accent-ruby)]',
    bg: 'bg-[var(--color-accent-ruby)]/10',
    border: 'border-[var(--color-accent-ruby)]/20',
    icon: AlertTriangle,
  },
  warning: {
    color: 'text-[var(--color-accent-amber)]',
    bg: 'bg-[var(--color-accent-amber)]/10',
    border: 'border-[var(--color-accent-amber)]/20',
    icon: AlertTriangle,
  },
  clear: {
    color: 'text-[var(--color-accent-emerald)]',
    bg: 'bg-[var(--color-accent-emerald)]/10',
    border: 'border-[var(--color-accent-emerald)]/20',
    icon: CheckCircle,
  },
  info: {
    color: 'text-[var(--color-accent-cyan)]',
    bg: 'bg-[var(--color-accent-cyan)]/10',
    border: 'border-[var(--color-accent-cyan)]/20',
    icon: Info,
  },
};

export function StatusBadge({ status, className, children, ...props }: StatusBadgeProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold uppercase tracking-wider border backdrop-blur-sm",
        config.color,
        config.bg,
        config.border,
        className
      )}
      {...props}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{children || status}</span>
    </div>
  );
}
