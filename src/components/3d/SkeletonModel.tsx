import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function SkeletonModel() {
  const pointsRef = useRef<THREE.Points>(null);
  const anomalyRef = useRef<THREE.Points>(null);
  const scanLineRef = useRef<THREE.Mesh>(null);

  // Generate a point cloud resembling a Tibia bone
  const particles = useMemo(() => {
    const p = [];
    for (let i = 0; i < 4000; i++) {
      const height = (Math.random() * 6) - 3; // Long bone
      const angle = Math.random() * Math.PI * 2;
      
      // Bones are wider at the ends (epiphysis) and narrower in middle (diaphysis)
      const radiusBase = 0.4 + Math.pow(Math.abs(height) / 3, 2) * 0.6; 
      const r = radiusBase * Math.sqrt(Math.random());

      let x = r * Math.cos(angle);
      let y = height;
      let z = r * Math.sin(angle);

      p.push(x, y, z);
    }
    return new Float32Array(p);
  }, []);

  // Hairline fracture anomaly on the distal 1/3
  const anomalyParticles = useMemo(() => {
    const p = [];
    // Distal 1/3 is near the bottom
    const centerY = -1.5;
    
    for (let i = 0; i < 400; i++) {
      const theta = Math.random() * Math.PI * 2;
      const height = (Math.random() * 0.4) - 0.2;
      const r = 0.5; // on the surface

      const x = r * Math.cos(theta);
      const y = centerY + height + (x * 0.5); // Slanted fracture line
      const z = r * Math.sin(theta);

      // Only keep points near the fracture line
      if (Math.abs(y - (centerY + (x * 0.5))) < 0.1) {
          p.push(x, y, z);
      }
    }
    return new Float32Array(p);
  }, []);

  useFrame((state) => {
    const time = state.clock.getElapsedTime();
    if (pointsRef.current) {
      pointsRef.current.rotation.y = time * 0.2;
      pointsRef.current.rotation.z = Math.sin(time * 0.1) * 0.1;
    }
    if (anomalyRef.current) {
      anomalyRef.current.rotation.copy(pointsRef.current!.rotation);
      // Flash the fracture
      (anomalyRef.current.material as THREE.PointsMaterial).opacity = 0.5 + Math.sin(time * 8) * 0.5;
    }
    if (scanLineRef.current) {
      scanLineRef.current.position.y = -1.5 + Math.sin(time * 3) * 0.5; // Scan over the fracture
    }
  });

  return (
    <group scale={1.2}>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[particles, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.03}
          color="#e4e4e7" // Zinc-200 for bone
          transparent
          opacity={0.4}
          blending={THREE.AdditiveBlending}
        />
      </points>
      
      {/* Fracture Highlight */}
      <points ref={anomalyRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[anomalyParticles, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.06}
          color="#f59e0b" // Amber for fraction
          transparent
          opacity={0.9}
          blending={THREE.AdditiveBlending}
        />
      </points>
      
      {/* Scanning laser effect */}
      <mesh ref={scanLineRef} position={[0, -1.5, 0]}>
        <boxGeometry args={[3, 0.02, 3]} />
        <meshBasicMaterial color="#f59e0b" transparent opacity={0.6} />
      </mesh>
    </group>
  );
}
