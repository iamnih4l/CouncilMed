import { Bell, Search, User, Menu, Plus } from 'lucide-react';
import { toast } from 'sonner';

export default function TopNav() {
  return (
    <header className="h-16 border-b border-white/5 bg-black/10 backdrop-blur-md flex items-center justify-between px-4 lg:px-8 shrink-0 z-20 sticky top-0">
      <div className="flex items-center space-x-4">
        <button 
          onClick={() => toast('Menu opened')}
          className="lg:hidden p-2 text-zinc-400 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
        >
          <Menu className="w-6 h-6" />
        </button>
        <div className="hidden md:flex relative group">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500 group-focus-within:text-[var(--color-accent-cyan)] transition-colors" />
          <input
            type="text"
            placeholder="Search patient ID, name, or scan type..."
            className="w-72 bg-white/5 border border-white/10 rounded-full py-1.5 pl-10 pr-4 text-sm text-zinc-300 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-[var(--color-accent-cyan)] focus:bg-white/10 transition-all shadow-inner"
          />
        </div>
      </div>

      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 border-r border-white/10 pr-4">
          <button 
             onClick={() => {
                const event = new CustomEvent('open-upload');
                window.dispatchEvent(event);
             }}
             className="hidden md:flex items-center space-x-2 px-3 py-1.5 bg-[var(--color-accent-cyan)]/10 hover:bg-[var(--color-accent-cyan)]/20 border border-[var(--color-accent-cyan)]/20 text-[var(--color-accent-cyan)] rounded-full transition-colors text-sm font-medium"
          >
            <Plus className="w-4 h-4" />
            <span>New Scan</span>
          </button>
          
          <button 
            onClick={() => toast('You have 3 new alerts from the diagnostic engine')}
            className="p-2 relative text-zinc-400 hover:text-white rounded-full hover:bg-white/5 transition-colors"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-[var(--color-accent-ruby)] rounded-full border border-[#0d1324] animate-pulse"></span>
          </button>
        </div>
        
        <div className="flex items-center space-x-3 cursor-pointer group" onClick={() => toast('Profile settings opened')}>
          <div className="text-right hidden sm:block">
            <p className="text-sm font-medium text-white group-hover:text-[var(--color-accent-cyan)] transition-colors">Dr. Sarah Jenkins</p>
            <p className="text-xs text-zinc-500">Chief Radiologist</p>
          </div>
          <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-[var(--color-accent-teal)] to-[var(--color-accent-cyan)] p-0.5 shdaow-lg relative">
            <div className="w-full h-full bg-[#0d1324] rounded-full flex items-center justify-center border border-white/10">
              <User className="w-5 h-5 text-white" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
