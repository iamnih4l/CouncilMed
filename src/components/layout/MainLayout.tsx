import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopNav from './TopNav';

export default function MainLayout() {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-[var(--color-medical-dark)] text-white">
      {/* Sidebar Navigation */}
      <Sidebar />

      <div className="flex flex-col flex-1 min-w-0">
        {/* Top Navigation Bar */}
        <TopNav />

        {/* Dynamic Workspace */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 relative z-0">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
