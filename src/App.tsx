import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import { Toaster } from 'sonner';

// Lazy load pages
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const WorkstationAnalysis = React.lazy(() => import('./pages/WorkstationAnalysis'));
const ModelRegistry = React.lazy(() => import('./pages/ModelRegistry'));
const History = React.lazy(() => import('./pages/History'));
const Reports = React.lazy(() => import('./pages/Reports'));
const Settings = React.lazy(() => import('./pages/Settings'));

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={
        <div className="h-screen w-full flex items-center justify-center bg-[var(--color-medical-dark)] text-white">
          <div className="flex flex-col items-center">
            <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-zinc-500 mt-4">Loading CouncilMed...</p>
          </div>
        </div>
      }>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="analyze" element={<WorkstationAnalysis />} />
            <Route path="models" element={<ModelRegistry />} />
            <Route path="history" element={<History />} />
            <Route path="reports" element={<Reports />} />
            <Route path="settings" element={<Settings />} />
            {/* Legacy routes redirect to new workstation */}
            <Route path="mri" element={<Navigate to="/analyze" replace />} />
            <Route path="ct" element={<Navigate to="/analyze" replace />} />
            <Route path="xray" element={<Navigate to="/analyze" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
      <Toaster 
        theme="dark" 
        position="bottom-right"
        toastOptions={{
          className: 'bg-[#0d1324] border border-white/10 text-white backdrop-blur-xl',
          style: {
            background: 'rgba(13, 19, 36, 0.8)',
            backdropFilter: 'blur(16px)',
            borderColor: 'rgba(255,255,255,0.1)'
          }
        }} 
      />
    </BrowserRouter>
  );
}

export default App;
