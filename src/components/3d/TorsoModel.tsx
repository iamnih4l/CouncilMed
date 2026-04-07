import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function TorsoModel() {
  const pointsRef = useRef<THREE.Points>(null);
  const scanLineRef = useRef<THREE.Mesh>(null);

  // Generate a point cloud that vaguely resembles an abdominal/torso region
  const particles = useMemo(() => {
    const p = [];
    for (let i = 0; i < 8000; i++) {
      const height = (Math.random() * 6) - 3; // From y=-3 to 3
      const angle = Math.random() * Math.PI * 2;
      
      // Taper the torso: wider at bottom (hips) and top (chest), narrower in middle (waist)
      const radiusBase = 1.8 + Math.abs(height) * 0.2; 
      const r = radiusBase * Math.sqrt(Math.random());

      let x = r * Math.cos(angle) * 1.5; // Wider in X
      let y = height;
      let z = r * Math.sin(angle) * 0.8; // Narrower in Z (depth)

      p.push(x, y, z);
    }
    return new Float32Array(p);
  }, []);

  useFrame((state) => {
    const time = state.clock.getElapsedTime();
    if (pointsRef.current) {
      pointsRef.current.rotation.y = time * 0.1;
    }
    if (scanLineRef.current) {
      scanLineRef.current.position.y = Math.sin(time * 0.8) * 3;
    }
  });

  return (
    <group scale={0.7}>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[particles, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.03}
          color="#10b981" // Emerald for CT
          transparent
          opacity={0.25}
          blending={THREE.AdditiveBlending}
        />
      </points>
      
      {/* Scanning laser effect */}
      <mesh ref={scanLineRef} position={[0, 0, 0]}>
        <boxGeometry args={[7, 0.05, 5]} />
        <meshBasicMaterial color="#06b6d4" transparent opacity={0.4} />
      </mesh>
    </group>
  );
}
