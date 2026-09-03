import { Outlet } from 'react-router-dom';

export default function MainLayout() {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-black text-white relative">
      {/* Dynamic Workspace Layer */}
      <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 relative z-10 w-full h-full pointer-events-auto">
        <Outlet />
      </main>
    </div>
  );
}
