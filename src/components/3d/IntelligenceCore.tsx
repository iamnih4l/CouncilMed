import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere } from '@react-three/drei';
import * as THREE from 'three';

interface Props {
  state: 'idle' | 'ingesting' | 'processing' | 'complete';
}

export function IntelligenceCore({ state }: Props) {
  const groupRef = useRef<THREE.Group>(null);
  const particlesRef = useRef<THREE.Points>(null);

  // Generate random particles
  const particlesCount = 200;
  const particlesPosition = useMemo(() => {
    const positions = new Float32Array(particlesCount * 3);
    for (let i = 0; i < particlesCount; i++) {
      const radius = 3 + Math.random() * 5;
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos(2 * Math.random() - 1);
      
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
    }
    return positions;
  }, []);

  useFrame((_state, delta) => {
    if (groupRef.current) {
      // Rotation speed depends on state
      let speed = 0.2;
      if (state === 'ingesting') speed = 0.8;
      if (state === 'processing') speed = 1.5;
      if (state === 'complete') speed = 0.1;
      
      groupRef.current.rotation.y += delta * speed;
      groupRef.current.rotation.x += delta * (speed * 0.5);
    }

    if (particlesRef.current) {
      let speed = 0.05;
      if (state === 'processing') speed = 0.2;
      particlesRef.current.rotation.y -= delta * speed;
    }
  });

  const coreColor = 
    state === 'processing' ? '#00f0ff' : 
    state === 'complete' ? '#10b981' : 
    '#08f7c3';

  return (
    <group ref={groupRef}>
      {/* Central Core */}
      <Sphere args={[1.5, 32, 32]}>
        <meshBasicMaterial color="#000" />
      </Sphere>
      
      {/* Wireframe outer shell */}
      <Sphere args={[1.6, 16, 16]}>
        <meshBasicMaterial color={coreColor} wireframe transparent opacity={0.3} />
      </Sphere>

      {/* Outer abstract ring */}
      <Sphere args={[2.5, 8, 8]}>
        <meshBasicMaterial color={coreColor} wireframe transparent opacity={state === 'processing' ? 0.4 : 0.1} />
      </Sphere>

      {/* Orbiting particles */}
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[particlesPosition, 3]}
          />
        </bufferGeometry>
        <pointsMaterial size={0.05} color={coreColor} transparent opacity={0.6} />
      </points>
    </group>
  );
}
