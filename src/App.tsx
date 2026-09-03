import React, { Suspense, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import { CinematicBootSequence } from './components/layout/CinematicBootSequence';
import { Toaster } from 'sonner';
import { AnimatePresence } from 'framer-motion';

// Lazy load pages
// const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const WorkstationAnalysis = React.lazy(() => import('./pages/WorkstationAnalysis'));
const ModelRegistry = React.lazy(() => import('./pages/ModelRegistry'));
const History = React.lazy(() => import('./pages/History'));
const Reports = React.lazy(() => import('./pages/Reports'));
const Settings = React.lazy(() => import('./pages/Settings'));

function App() {
  const [booted, setBooted] = useState(false);

  return (
    <BrowserRouter>
      <AnimatePresence mode="wait">
        {!booted ? (
          <CinematicBootSequence key="boot" onComplete={() => setBooted(true)} />
        ) : (
          <Suspense key="app" fallback={
            <div className="h-screen w-full flex items-center justify-center bg-black text-white">
              <div className="flex flex-col items-center">
                <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-xs tracking-widest text-zinc-500 mt-4 uppercase">Loading Councilman...</p>
              </div>
            </div>
          }>
            <Routes>
              <Route path="/" element={<MainLayout />}>
                {/* Default to the spatial workstation analysis */}
                <Route index element={<WorkstationAnalysis />} />
                <Route path="analyze" element={<WorkstationAnalysis />} />
                <Route path="models" element={<ModelRegistry />} />
                <Route path="history" element={<History />} />
                <Route path="reports" element={<Reports />} />
                <Route path="settings" element={<Settings />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </Suspense>
        )}
      </AnimatePresence>

      <Toaster 
        theme="dark" 
        position="bottom-right"
        toastOptions={{
          className: 'bg-black/60 border border-white/10 text-white backdrop-blur-xl',
          style: {
            background: 'rgba(0, 0, 0, 0.6)',
            backdropFilter: 'blur(16px)',
            borderColor: 'rgba(255,255,255,0.1)'
          }
        }} 
      />
    </BrowserRouter>
  );
}

export default App;
