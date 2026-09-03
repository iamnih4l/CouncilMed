import { Canvas } from '@react-three/fiber';
import { Stars } from '@react-three/drei';
import { IntelligenceCore } from './IntelligenceCore';
import { Suspense } from 'react';

interface Props {
  coreState: 'idle' | 'ingesting' | 'processing' | 'complete';
}

export function WorkspaceEnvironment({ coreState }: Props) {
  return (
    <div className="absolute inset-0 z-0 pointer-events-none">
      <Canvas camera={{ position: [0, 0, 12], fov: 45 }}>
        <color attach="background" args={['#010205']} />
        
        <ambientLight intensity={0.2} />
        <directionalLight position={[10, 10, 5]} intensity={1} color="#00f0ff" />
        <directionalLight position={[-10, -10, -5]} intensity={0.5} color="#ffaa00" />
        
        <Suspense fallback={null}>
          <IntelligenceCore state={coreState} />
          
          <Stars 
            radius={50} 
            depth={50} 
            count={3000} 
            factor={3} 
            saturation={0} 
            fade 
            speed={coreState === 'idle' ? 0.5 : 1.5} 
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
