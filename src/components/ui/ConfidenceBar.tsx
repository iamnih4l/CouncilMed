import { cn } from '../../utils/cn';

interface ConfidenceBarProps {
  score: number; // 0 to 100
  className?: string;
}

export function ConfidenceBar({ score, className }: ConfidenceBarProps) {
  // Determine color based on score
  let colorClass = 'bg-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.6)]';
  if (score < 60) colorClass = 'bg-red-400 shadow-[0_0_12px_rgba(239,68,68,0.6)]';
  else if (score < 85) colorClass = 'bg-amber-400 shadow-[0_0_12px_rgba(245,158,11,0.6)]';

  return (
    <div className={cn("w-full flex items-center space-x-3 font-mono", className)}>
      <div className="flex-1 h-2 bg-black/60 rounded-full overflow-hidden border border-white/10 relative">
        <div 
          className={cn("h-full rounded-full transition-all duration-700 ease-out", colorClass)}
          style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
        />
      </div>
      <span className={cn(
        "text-xs font-semibold tabular-nums w-12 text-right",
        score >= 85 ? "text-emerald-400" : score >= 60 ? "text-amber-400" : "text-red-400"
      )}>
        {score.toFixed(1)}%
      </span>
    </div>
  );
}
