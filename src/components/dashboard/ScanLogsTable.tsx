import { GlassCard } from '../ui/GlassCard';
import { StatusBadge } from '../ui/StatusBadge';
import { ConfidenceBar } from '../ui/ConfidenceBar';
import { Brain, ActivitySquare, Bone } from 'lucide-react';
import { useDiagnosticEngine } from '../../hooks/useDiagnosticEngine';
import { useEffect, useState } from 'react';

const mockLogs = [
  { id: 'PT-8801', modality: 'MRI', time: '10:42 AM', finding: 'Glioblastoma Suspicion', models: 'viT-Med-Alpha, ResNet-Diagnostic-v4', conf: 92.4, status: 'critical' as const },
  { id: 'PT-8802', modality: 'CT', time: '10:25 AM', finding: 'Clear / No Anomalies', models: 'DenseNet-Abd, ResNet-Diagnostic-v4', conf: 98.1, status: 'clear' as const },
  { id: 'PT-8803', modality: 'X-Ray', time: '09:50 AM', finding: 'Hairline Fracture - Tibia', models: 'viT-Med-Alpha', conf: 76.5, status: 'warning' as const },
  { id: 'PT-8804', modality: 'MRI', time: '09:12 AM', finding: 'Micro-aneurysm detected', models: 'Ensemble-V2', conf: 88.9, status: 'critical' as const },
  { id: 'PT-8805', modality: 'CT', time: '08:45 AM', finding: 'Clear / No Anomalies', models: 'DenseNet-Abd', conf: 95.0, status: 'clear' as const },
];

const modalityIcon = {
  'MRI': Brain,
  'CT': ActivitySquare,
  'X-Ray': Bone,
};

export function ScanLogsTable() {
  const { result } = useDiagnosticEngine();
  const [logs, setLogs] = useState(mockLogs);

  useEffect(() => {
    if (result) {
      const newLog = {
        id: `PT-${Math.floor(Math.random() * 9000) + 1000}`,
        modality: 'MRI',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        finding: result.primaryDiagnosis,
        models: 'DenseNet-121, Attention-Net, Swin-UNETR',
        conf: parseFloat(result.overallConfidence.toFixed(1)),
        status: result.severity as any,
      };
      
      setLogs((prev) => {
        // Prevent duplicate appending if the result is already there
        if (prev[0]?.finding === newLog.finding && prev[0]?.conf === newLog.conf) {
          return prev;
        }
        return [newLog, ...prev].slice(0, 10);
      });
    }
  }, [result]);

  return (
    <GlassCard className="col-span-1 lg:col-span-3">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-bold text-white tracking-tight">Recent Scans Analysis</h2>
          <p className="text-sm text-zinc-400 mt-1">Real-time log of multi-modal AI processing</p>
        </div>
        <button className="text-sm text-[var(--color-accent-cyan)] hover:text-white transition-colors font-medium">View All Logs &rarr;</button>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse min-w-[800px]">
          <thead>
            <tr className="border-b border-white/10 text-xs uppercase tracking-wider text-zinc-500">
              <th className="pb-3 pl-2 font-medium">Patient ID</th>
              <th className="pb-3 px-4 font-medium">Modality</th>
              <th className="pb-3 px-4 font-medium">AI Finding</th>
              <th className="pb-3 px-4 font-medium hidden sm:table-cell">Model Consensus</th>
              <th className="pb-3 px-4 font-medium">Confidence</th>
              <th className="pb-3 pr-2 font-medium text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {logs.map((log) => {
              const Icon = modalityIcon[log.modality as keyof typeof modalityIcon] || Brain;
              return (
                <tr key={log.id} className="hover:bg-white/5 transition-colors group cursor-pointer">
                  <td className="py-4 pl-2">
                    <span className="font-mono text-sm font-medium text-zinc-200 group-hover:text-white transition-colors">{log.id}</span>
                    <div className="text-xs text-zinc-500 mt-1">{log.time}</div>
                  </td>
                  <td className="py-4 px-4">
                    <div className="flex items-center space-x-2 text-zinc-300">
                      <div className="p-1.5 bg-white/5 rounded-lg border border-white/10 group-hover:border-[var(--color-accent-cyan)]/30 transition-colors">
                        <Icon className="w-4 h-4 text-[var(--color-accent-cyan)]" />
                      </div>
                      <span className="text-sm font-medium">{log.modality}</span>
                    </div>
                  </td>
                  <td className="py-4 px-4">
                    <span className="text-sm text-white font-medium">{log.finding}</span>
                  </td>
                  <td className="py-4 px-4 hidden sm:table-cell">
                    <span className="text-xs text-zinc-400 truncate max-w-[150px] block">{log.models}</span>
                  </td>
                  <td className="py-4 px-4 min-w-[140px]">
                    <ConfidenceBar score={log.conf} />
                  </td>
                  <td className="py-4 pr-2 text-right">
                    <StatusBadge status={log.status} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </GlassCard>
  );
}
