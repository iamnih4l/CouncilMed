import { cn } from './GlassCard';

interface ConfidenceBarProps {
  score: number; // 0 to 100
  className?: string;
}

export function ConfidenceBar({ score, className }: ConfidenceBarProps) {
  // Determine color based on score
  let colorClass = 'bg-[var(--color-accent-emerald)] shadow-[0_0_10px_rgba(16,185,129,0.5)]';
  if (score < 60) colorClass = 'bg-[var(--color-accent-ruby)] shadow-[0_0_10px_rgba(239,68,68,0.5)]';
  else if (score < 85) colorClass = 'bg-[var(--color-accent-amber)] shadow-[0_0_10px_rgba(245,158,11,0.5)]';

  return (
    <div className={cn("w-full flex items-center space-x-3", className)}>
      <div className="flex-1 h-2 bg-black/40 rounded-full overflow-hidden border border-white/5 relative">
        <div 
          className={cn("h-full rounded-full transition-all duration-1000 ease-out", colorClass)}
          style={{ width: `${score}%` }}
        />
      </div>
      <span className={cn(
        "text-sm font-semibold tracking-wide w-12 text-right",
        score >= 85 ? "text-[var(--color-accent-emerald)]" : score >= 60 ? "text-[var(--color-accent-amber)]" : "text-[var(--color-accent-ruby)]"
      )}>
        {score.toFixed(1)}%
      </span>
    </div>
  );
}
