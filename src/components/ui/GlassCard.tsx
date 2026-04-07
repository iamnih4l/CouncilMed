import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  focused?: boolean;
}

export function GlassCard({ children, className, focused, ...props }: GlassCardProps) {
  return (
    <div
      className={cn(
        "bg-white/5 backdrop-blur-md border border-white/10 shadow-[0_4px_30px_rgba(0,0,0,0.1)] rounded-2xl p-6 transition-all duration-300",
        focused && "border-white/20 shadow-[0_0_20px_rgba(6,182,212,0.15)] bg-white/10",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
