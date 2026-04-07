import { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopNav from './TopNav';
import { UploadModal } from '../ui/UploadModal';

export default function MainLayout() {
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [uploadModality, setUploadModality] = useState('mri');

  useEffect(() => {
    const handleOpenOption = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail?.modality) {
        setUploadModality(customEvent.detail.modality);
      } else {
        setUploadModality('mri');
      }
      setIsUploadOpen(true);
    };
    window.addEventListener('open-upload', handleOpenOption);
    return () => window.removeEventListener('open-upload', handleOpenOption);
  }, []);
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

      <UploadModal 
        isOpen={isUploadOpen} 
        onClose={() => setIsUploadOpen(false)} 
        defaultModality={uploadModality}
      />
    </div>
  );
}
