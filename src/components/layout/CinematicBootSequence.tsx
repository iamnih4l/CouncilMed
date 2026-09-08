import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  onComplete: () => void;
}

export function CinematicBootSequence({ onComplete }: Props) {
  const [stage, setStage] = useState(0);

  const stages = [
    'INITIALIZING INTELLIGENCE CORE',
    'LOADING ANALYSIS ENGINE',
    'ESTABLISHING DATA CONTEXT',
    'SYSTEM READY'
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setStage(s => {
        if (s >= stages.length - 1) {
          clearInterval(timer);
          setTimeout(onComplete, 400); // Wait a beat before transition
          return s;
        }
        return s + 1;
      });
    }, 300);

    return () => clearInterval(timer);
  }, [onComplete, stages.length]);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.8, ease: "easeInOut" } }}
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black"
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,240,255,0.05)_0%,transparent_50%)]" />
      
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 1 }}
        className="text-center z-10"
      >
        <h1 className="text-4xl md:text-6xl font-light tracking-[0.3em] text-white mb-8 text-glow">
          COUNCILMED
        </h1>
        
        <div className="h-6 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.p
              key={stage}
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -20, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="text-xs md:text-sm font-mono tracking-widest text-cyan-500 uppercase"
            >
              {stages[stage]}
            </motion.p>
          </AnimatePresence>
        </div>
      </motion.div>
      
      <button 
        onClick={onComplete}
        className="absolute bottom-8 text-[10px] uppercase tracking-widest text-white/30 hover:text-white/60 transition-colors"
      >
        [ SKIP SEQUENCE ]
      </button>
    </motion.div>
  );
}
