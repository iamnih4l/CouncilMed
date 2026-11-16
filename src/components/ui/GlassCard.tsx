import React from 'react';
import { cn } from '../../utils/cn';

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  focused?: boolean;
}

export function GlassCard({ children, className, focused, ...props }: GlassCardProps) {
  return (
    <div
      className={cn(
        "bg-black/40 backdrop-blur-xl border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.6)] rounded-2xl p-6 transition-all duration-300 hover:border-white/20",
        focused && "border-cyan-500/40 shadow-[0_0_25px_rgba(0,240,255,0.2)] bg-black/60",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
