import { GlassCard } from '../components/ui/GlassCard';
import { motion } from 'framer-motion';
import { Settings as SettingsIcon, Bell, Shield, User, Globe, Moon, CreditCard, Save, RefreshCcw } from 'lucide-react';
import { toast } from 'sonner';
import { useWorkstationSettings } from '../hooks/useWorkstationSettings';

export default function Settings() {
  const { settings, updateSettings, resetSettings } = useWorkstationSettings();

  const handleSave = () => {
    toast.success('Clinical settings synchronized');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <SettingsIcon className="w-8 h-8 text-[var(--color-accent-cyan)]" />
            System Settings
          </h1>
          <p className="text-zinc-400 mt-2">Manage your clinical workstation and account preferences</p>
        </div>
        <button 
          onClick={handleSave}
          className="px-6 py-2.5 bg-[var(--color-accent-cyan)] hover:bg-[var(--color-accent-teal)] text-[#0d1324] font-bold rounded-xl transition-all shadow-lg flex items-center gap-2"
        >
          <Save className="w-4 h-4" />
          Save Changes
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-1 space-y-2">
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-white/10 text-white border border-white/10 text-left font-medium transition-all">
            <User className="w-4 h-4 text-[var(--color-accent-cyan)]" />
            Profile
          </button>
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-zinc-400 hover:bg-white/5 hover:text-white text-left font-medium transition-all">
            <Bell className="w-4 h-4" />
            Notifications
          </button>
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-zinc-400 hover:bg-white/5 hover:text-white text-left font-medium transition-all">
            <Shield className="w-4 h-4" />
            Security
          </button>
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-zinc-400 hover:bg-white/5 hover:text-white text-left font-medium transition-all">
            <Globe className="w-4 h-4" />
            Language
          </button>
        </div>

        <div className="md:col-span-2 space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <GlassCard className="p-6 space-y-6">
              <h3 className="text-lg font-bold text-white border-b border-white/5 pb-4">Clinical Profile</h3>
              
              <div className="grid grid-cols-1 gap-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-zinc-400">Full Name</label>
                  <input 
                    type="text" 
                    value={settings.doctorName}
                    onChange={(e) => updateSettings({ doctorName: e.target.value })}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-[var(--color-accent-cyan)] transition-all"
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                   <div className="space-y-2">
                    <label className="text-sm font-medium text-zinc-400">Specialization</label>
                    <input 
                      type="text" 
                      value={settings.specialization}
                      onChange={(e) => updateSettings({ specialization: e.target.value })}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-[var(--color-accent-cyan)] transition-all"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-zinc-400">Clinical ID</label>
                    <input 
                      type="text" 
                      value={settings.clinicalId}
                      onChange={(e) => updateSettings({ clinicalId: e.target.value })}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-[var(--color-accent-cyan)] transition-all"
                    />
                  </div>
                </div>
              </div>
            </GlassCard>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <GlassCard className="p-6 space-y-6">
              <h3 className="text-lg font-bold text-white border-b border-white/5 pb-4">Workstation Display</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-white/5 border border-white/5 rounded-xl">
                  <div className="flex items-center gap-3">
                    <Moon className="w-5 h-5 text-[var(--color-accent-cyan)]" />
                    <div>
                      <p className="text-sm font-medium text-white">Dark Mode (High Contrast)</p>
                      <p className="text-xs text-zinc-500 text-left">Optimized for diagnostic accuracy</p>
                    </div>
                  </div>
                   <div className="w-12 h-6 bg-white/5 border border-white/10 rounded-full relative cursor-pointer" onClick={() => updateSettings({ darkMode: !settings.darkMode })}>
                    <div className={`absolute top-1 w-4 h-4 rounded-full transition-all ${settings.darkMode ? 'right-1 bg-[var(--color-accent-cyan)] shadow-[0_0_8px_var(--color-accent-cyan)]' : 'left-1 bg-zinc-600'}`}></div>
                  </div>
                </div>

                <div className="flex flex-col gap-3 p-4 bg-white/5 border border-white/5 rounded-xl">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <Shield className="w-5 h-5 text-[var(--color-accent-cyan)]" />
                      <p className="text-sm font-medium text-white">AI Diagnostic Sensitivity</p>
                    </div>
                    <span className="text-xs font-mono text-[var(--color-accent-cyan)]">{(settings.aiSensitivity * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range"
                    min="0.5"
                    max="0.95"
                    step="0.05"
                    value={settings.aiSensitivity}
                    onChange={(e) => updateSettings({ aiSensitivity: parseFloat(e.target.value) })}
                    className="w-full accent-[var(--color-accent-cyan)] h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-[10px] text-zinc-500">Higher sensitivity increases anomaly recall but may produce more false positive alerts.</p>
                </div>

                <div className="flex items-center justify-between p-4 bg-white/5 border border-white/5 rounded-xl">
                   <div className="flex items-center gap-3">
                    <CreditCard className="w-5 h-5 text-zinc-400" />
                    <div>
                      <p className="text-sm font-medium text-white">Clinical Vault Persistence</p>
                      <p className="text-xs text-zinc-500 text-left">Internal localStorage active</p>
                    </div>
                  </div>
                  <button 
                    onClick={() => { resetSettings(); toast.info('System factory reset applied'); }}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-500 text-xs font-bold rounded-lg border border-red-500/20 transition-all"
                  >
                    <RefreshCcw className="w-3 h-3" />
                    Factory Reset
                  </button>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
