import { GlassCard } from '../components/ui/GlassCard';
import { motion } from 'framer-motion';
import { FileText, Download, Share2, Search, Filter, ArrowUpRight, History } from 'lucide-react';
import { StatusBadge } from '../components/ui/StatusBadge';
import { toast } from 'sonner';
import { usePersistentReports } from '../hooks/usePersistentReports';

export default function Reports() {
  const { reports, deleteReport } = usePersistentReports();

  const handleDownload = async (reportId: string) => {
    // For a real app, we'd need to re-render the report or have it in a hidden div
    // For now, we'll notify that we're generating from history
    toast.info(`Generating archived report for ${reportId}...`);
    // Note: Re-generating exact PDF from history requires the DOM element to exist.
    // In a prod app, we'd store the PDF blob or its JSON state.
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-12">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <FileText className="w-8 h-8 text-[var(--color-accent-cyan)]" />
            Diagnostic Reports
          </h1>
          <p className="text-zinc-400 mt-2">Centralized access to all AI-generated diagnostic records</p>
        </div>
        <div className="flex gap-3">
          <div className="relative group">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500 group-focus-within:text-[var(--color-accent-cyan)] transition-colors" />
            <input 
              type="text" 
              placeholder="Search reports..."
              className="bg-white/5 border border-white/10 rounded-xl py-2 pl-10 pr-4 text-sm text-zinc-300 focus:outline-none focus:ring-2 focus:ring-[var(--color-accent-cyan)] transition-all w-64"
            />
          </div>
          <button className="p-2 bg-white/5 border border-white/10 rounded-xl text-zinc-300 hover:text-white transition-all">
            <Filter className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <GlassCard className="p-6 md:col-span-3">
          <div className="flex items-center justify-between mb-6 border-b border-white/5 pb-4">
             <h3 className="text-lg font-bold text-white flex items-center gap-2">
               <History className="w-5 h-5 text-zinc-400" />
               Recent Activity
             </h3>
             <span className="text-xs font-medium text-zinc-500 uppercase tracking-widest">Total {reports.length} Reports</span>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-white/5 text-zinc-500 text-xs uppercase tracking-wider">
                  <th className="px-4 py-3 font-semibold">Report ID</th>
                  <th className="px-4 py-3 font-semibold">Diagnostic Target</th>
                  <th className="px-4 py-3 font-semibold">Capture Date</th>
                  <th className="px-4 py-3 font-semibold">AI Status</th>
                  <th className="px-4 py-3 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {reports.map((report, index) => (
                  <motion.tr 
                    key={report.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="hover:bg-white/5 transition-colors group"
                  >
                    <td className="px-4 py-4 text-sm font-medium text-white">{report.id}</td>
                    <td className="px-4 py-4 text-sm text-zinc-300 font-medium">{report.modality} Analysis</td>
                    <td className="px-4 py-4 text-sm text-zinc-500">{report.date}</td>
                    <td className="px-4 py-4 text-sm">
                      <StatusBadge status={report.result.severity as any}>
                        {report.result.primaryDiagnosis}
                      </StatusBadge>
                    </td>
                    <td className="px-4 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button 
                          onClick={() => toast.info(`Viewing ${report.id}`)}
                          className="p-1.5 bg-white/5 rounded-lg text-zinc-400 hover:text-white border border-white/10 transition-all opacity-0 group-hover:opacity-100"
                        >
                          <ArrowUpRight className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={() => deleteReport(report.id)}
                          className="p-1.5 bg-red-500/10 rounded-lg text-red-500 hover:bg-red-500/20 border border-red-500/20 transition-all opacity-0 group-hover:opacity-100"
                        >
                          <Download className="w-4 h-4 rotate-180" />
                        </button>
                        <button 
                          onClick={() => handleDownload(report.id)}
                          className="p-1.5 bg-gradient-to-br from-[var(--color-accent-cyan)] to-[var(--color-accent-teal)] rounded-lg text-[#0d1324] font-bold border border-white/10 transition-all opacity-0 group-hover:opacity-100"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassCard>

        <div className="md:col-span-1 space-y-6">
          <GlassCard className="p-6">
            <h4 className="text-sm font-bold text-white uppercase tracking-widest mb-4">Storage Insights</h4>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-zinc-500">DICOM Vault Usage</span>
                  <span className="text-white">64.2%</span>
                </div>
                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-[var(--color-accent-cyan)] rounded-full w-[64.2%] transition-all"></div>
                </div>
              </div>
              <div className="pt-4 border-t border-white/10">
                <p className="text-xs text-zinc-400 mb-3">Syncing with Central Medical Archive...</p>
                <div className="flex -space-x-2">
                   {[1,2,3,4].map(i => (
                     <div key={i} className="w-8 h-8 rounded-full border-2 border-[#0d1324] bg-white/10 flex items-center justify-center text-[10px] text-zinc-500 font-bold uppercase overflow-hidden">
                        <img src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${i * 123}`} alt="User" />
                     </div>
                   ))}
                   <div className="w-8 h-8 rounded-full border-2 border-[#0d1324] bg-white/10 flex items-center justify-center text-[10px] text-zinc-500 font-bold">+12</div>
                </div>
              </div>
            </div>
          </GlassCard>
          
          <button className="w-full py-4 bg-white/5 border border-white/10 hover:bg-white/10 rounded-2xl flex flex-col items-center justify-center gap-2 transition-all group">
            <Share2 className="w-6 h-6 text-zinc-400 group-hover:text-[var(--color-accent-cyan)] transition-colors" />
            <span className="text-sm font-medium text-white">Share Selected</span>
          </button>
        </div>
      </div>
    </div>
  );
}
