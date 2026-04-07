import { Canvas } from '@react-three/fiber';
import { GlassCard } from '../components/ui/GlassCard';
import ParticleField from '../components/3d/ParticleField';
import { ScanLogsTable } from '../components/dashboard/ScanLogsTable';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Activity, Brain, Users, CheckCircle, Loader2 } from 'lucide-react';
import { useDiagnosticEngine } from '../hooks/useDiagnosticEngine';
import { motion } from 'framer-motion';
import type { Variants } from 'framer-motion';

const pieData = [
  { name: 'Clear', value: 65, color: '#10b981' },
  { name: 'Anomalies', value: 25, color: '#f59e0b' },
  { name: 'Critical', value: 10, color: '#ef4444' },
];



const containerVariants: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

export default function Dashboard() {
  const { state, result, isProcessing } = useDiagnosticEngine();

  // Dynamically update stats based on live engine state
  const scansProcessed = 1284 + (result ? 1 : 0);
  const anomaliesDetected = 347 + (result && result.primaryDiagnosis !== 'Normal' ? 1 : 0);
  const currentModelStr = isProcessing 
    ? (state.densenetStatus === 'running' ? 'DenseNet-121' 
      : state.attentionStatus === 'running' ? 'Attention-Net' 
      : state.swinStatus === 'running' ? 'Swin-UNETR' 
      : 'Council Fusion')
    : 'ResNet-Diag-v4';

  return (
    <motion.div 
      className="max-w-7xl mx-auto space-y-6"
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-400">System Analytics Overview</h1>
          <p className="text-zinc-400 mt-1">Live metrics from the CouncilAI diagnostic engine.</p>
        </div>
        <div className="flex items-center space-x-2 text-sm bg-white/5 border border-white/10 rounded-full px-4 py-1.5 backdrop-blur-md">
          <span className={`w-2.5 h-2.5 rounded-full animate-pulse shadow-[0_0_10px_currentColor] ${isProcessing ? 'bg-[var(--color-accent-amber)] text-[var(--color-accent-amber)]' : 'bg-[var(--color-accent-emerald)] text-[var(--color-accent-emerald)]'}`}></span>
          <span className="text-zinc-200">
            {isProcessing ? `Council Pipeline Active — ${state.progress.toFixed(0)}%` : 'System Online - Processing Live Queue'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-[var(--color-accent-cyan)]/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-[var(--color-accent-cyan)]/20 transition-all duration-500"></div>
            <div className="flex justify-between items-start">
              <div>
                <p className="text-zinc-400 text-sm font-medium">Scans Processed (24h)</p>
                <h3 className="text-3xl font-bold text-white mt-1">{scansProcessed.toLocaleString()}</h3>
                <p className="text-xs text-[var(--color-accent-emerald)] mt-2 flex items-center font-medium">
                  ↑ 12.5% vs yesterday
                </p>
              </div>
              <div className="p-3 rounded-xl bg-[var(--color-accent-cyan)]/10 text-[var(--color-accent-cyan)] border border-[var(--color-accent-cyan)]/20">
                <Activity className="w-5 h-5" />
              </div>
            </div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-[var(--color-accent-emerald)]/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-[var(--color-accent-emerald)]/20 transition-all duration-500"></div>
            <div className="flex justify-between items-start">
              <div>
                <p className="text-zinc-400 text-sm font-medium">Avg Diagnostic Time</p>
                <h3 className="text-3xl font-bold text-white mt-1">0.8<span className="text-lg text-zinc-500 ml-1">sec</span></h3>
                <p className="text-xs text-[var(--color-accent-emerald)] mt-2 flex items-center font-medium">
                  ↓ 0.2s optimization
                </p>
              </div>
              <div className="p-3 rounded-xl bg-[var(--color-accent-emerald)]/10 text-[var(--color-accent-emerald)] border border-[var(--color-accent-emerald)]/20">
                <CheckCircle className="w-5 h-5" />
              </div>
            </div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-[var(--color-accent-ruby)]/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-[var(--color-accent-ruby)]/20 transition-all duration-500"></div>
            <div className="flex justify-between items-start">
              <div>
                <p className="text-zinc-400 text-sm font-medium">Anomalies Detected</p>
                <h3 className="text-3xl font-bold text-white mt-1">{anomaliesDetected.toLocaleString()}</h3>
                <p className="text-xs text-[var(--color-accent-ruby)] mt-2 flex items-center font-medium">
                  ↑ 4 critical alerts
                </p>
              </div>
              <div className="p-3 rounded-xl bg-[var(--color-accent-ruby)]/10 text-[var(--color-accent-ruby)] border border-[var(--color-accent-ruby)]/20">
                <Brain className="w-5 h-5" />
              </div>
            </div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-[var(--color-accent-amber)]/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-[var(--color-accent-amber)]/20 transition-all duration-500"></div>
            <div className="flex justify-between items-start">
              <div>
                <p className="text-zinc-400 text-sm font-medium">Active Radiologists</p>
                <h3 className="text-3xl font-bold text-white mt-1">42</h3>
                <p className="text-xs text-[var(--color-accent-teal)] mt-2 flex items-center font-medium">
                  System load: Normal
                </p>
              </div>
              <div className="p-3 rounded-xl bg-[var(--color-accent-amber)]/10 text-[var(--color-accent-amber)] border border-[var(--color-accent-amber)]/20">
                <Users className="w-5 h-5" />
              </div>
            </div>
          </GlassCard>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div variants={itemVariants} className="col-span-1 lg:col-span-2">
          <GlassCard className="h-96 flex flex-col relative overflow-hidden">
            <h2 className="text-lg font-bold text-white mb-6">Live AI Pipeline Visualization</h2>
            
            {/* Embedded 3D Canvas */}
            <div className="absolute inset-0 z-0 opacity-40 mix-blend-screen pointer-events-none mt-16">
               <Canvas camera={{ position: [0, 0, 5] }}>
                 <ambientLight intensity={0.5} />
                 <ParticleField />
               </Canvas>
            </div>
            
            <div className="relative z-10 flex-1 flex flex-col justify-end">
               <div className="bg-black/40 backdrop-blur-md rounded-xl p-4 border border-white/5 flex items-center justify-between">
                 <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-lg bg-[var(--color-accent-cyan)]/20 flex items-center justify-center border border-[var(--color-accent-cyan)]/30">
                      {isProcessing ? (
                        <Loader2 className="w-5 h-5 text-[var(--color-accent-cyan)] animate-spin" />
                      ) : (
                        <Activity className="w-5 h-5 text-[var(--color-accent-cyan)]" />
                      )}
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-white">
                        {isProcessing ? 'Council Inference Active' : 'Neural Net Active'}
                      </p>
                      <p className="text-xs text-[var(--color-accent-cyan)]">
                        {isProcessing ? 'Analyzing DICOM streams...' : 'Processing 12 streams/sec'}
                      </p>
                    </div>
                 </div>
                 <div className="text-right">
                    <p className="text-xs text-zinc-400">Current Model</p>
                    <p className="text-sm font-mono text-zinc-200">{currentModelStr}</p>
                 </div>
               </div>
            </div>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants} className="col-span-1">
          <GlassCard className="h-96 flex flex-col">
            <h2 className="text-lg font-bold text-white mb-2">Finding Distribution</h2>
            <div className="flex-1 -ml-4">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={5}
                    dataKey="value"
                    stroke="rgba(255,255,255,0.05)"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0d1324', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '12px' }}
                    itemStyle={{ color: '#fff' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center space-x-4 mt-2 mb-2">
              {pieData.map((item) => (
                <div key={item.name} className="flex items-center text-xs text-zinc-400">
                  <div className="w-2 h-2 rounded-full mr-2" style={{ backgroundColor: item.color }} />
                  {item.name}
                </div>
              ))}
            </div>
          </GlassCard>
        </motion.div>
      </div>

      <motion.div variants={itemVariants}>
        <ScanLogsTable />
      </motion.div>
    </motion.div>
  );
}
