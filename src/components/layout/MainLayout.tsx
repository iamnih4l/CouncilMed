import { Outlet } from 'react-router-dom';
import TopNav from './TopNav';

export default function MainLayout() {
  return (
    <div className="flex flex-col h-screen w-full overflow-hidden bg-black text-white relative">
      {/* Cinematic Top Navigation HUD */}
      <TopNav />

      {/* Dynamic Workspace Layer */}
      <main className="flex-1 overflow-y-auto p-3 sm:p-4 md:p-6 lg:p-8 relative z-10 w-full h-[calc(100vh-4rem)] pointer-events-auto">
        <Outlet />
      </main>
    </div>
  );
}
