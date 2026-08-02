import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import { Toaster } from 'sonner';

const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const MRIAnalysis = React.lazy(() => import('./pages/MRIAnalysis'));
const CTAnalysis = React.lazy(() => import('./pages/CTAnalysis'));
const XRayAnalysis = React.lazy(() => import('./pages/XRayAnalysis'));
const Reports = React.lazy(() => import('./pages/Reports'));
const Settings = React.lazy(() => import('./pages/Settings'));

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<div className="h-screen w-full flex items-center justify-center bg-[var(--color-medical-dark)] text-white">Loading...</div>}>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="mri" element={<MRIAnalysis />} />
            <Route path="ct" element={<CTAnalysis />} />
            <Route path="xray" element={<XRayAnalysis />} />
            <Route path="reports" element={<Reports />} />
            <Route path="settings" element={<Settings />} />
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
