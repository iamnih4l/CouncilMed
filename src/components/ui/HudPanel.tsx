import type { ReactNode } from 'react';
import { useState, useRef } from 'react';
import { motion } from 'framer-motion';

interface Props {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  onDrop?: (e: React.DragEvent) => void;
  onDragOver?: (e: React.DragEvent) => void;
}

export function HudPanel({ children, className = '', onClick, onDrop, onDragOver }: Props) {
  const divRef = useRef<HTMLDivElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!divRef.current || isFocused) return;

    const div = divRef.current;
    const rect = div.getBoundingClientRect();

    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const handleFocus = () => {
    setIsFocused(true);
    setOpacity(1);
  };

  const handleBlur = () => {
    setIsFocused(false);
    setOpacity(0);
  };

  const handleMouseEnter = () => {
    setOpacity(1);
  };

  const handleMouseLeave = () => {
    setOpacity(0);
  };

  return (
    <motion.div
      ref={divRef}
      onMouseMove={handleMouseMove}
      onFocus={handleFocus}
      onBlur={handleBlur}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      onDrop={onDrop}
      onDragOver={onDragOver}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={`relative overflow-hidden rounded-xl bg-black/50 backdrop-blur-2xl border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.7)] p-6 transition-all duration-300 hover:border-cyan-500/40 hover:shadow-[0_8px_32px_rgba(0,240,255,0.08)] ${className}`}
    >
      <div
        className="pointer-events-none absolute -inset-px opacity-0 transition duration-300 rounded-xl"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, rgba(0,240,255,.08), transparent 40%)`,
        }}
      />
      
      {/* Corner accents with holographic glow */}
      <div className="absolute top-0 left-0 w-2.5 h-2.5 border-t border-l border-cyan-400/80 rounded-tl-xl pointer-events-none shadow-[0_0_8px_rgba(0,240,255,0.3)]" />
      <div className="absolute top-0 right-0 w-2.5 h-2.5 border-t border-r border-cyan-400/80 rounded-tr-xl pointer-events-none shadow-[0_0_8px_rgba(0,240,255,0.3)]" />
      <div className="absolute bottom-0 left-0 w-2.5 h-2.5 border-b border-l border-cyan-400/80 rounded-bl-xl pointer-events-none shadow-[0_0_8px_rgba(0,240,255,0.3)]" />
      <div className="absolute bottom-0 right-0 w-2.5 h-2.5 border-b border-r border-cyan-400/80 rounded-br-xl pointer-events-none shadow-[0_0_8px_rgba(0,240,255,0.3)]" />

      <div className="relative z-10 h-full">{children}</div>
    </motion.div>
  );
}
