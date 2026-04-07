import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { CouncilConsensusResult } from '../../services/inference/types';

interface BrainModelProps {
  consensusResult?: CouncilConsensusResult | null;
}

export default function BrainModel({ consensusResult }: BrainModelProps) {
  const pointsRef = useRef<THREE.Points>(null);
  const anomalyRef = useRef<THREE.Points>(null);
  const heatmapRef = useRef<THREE.Points>(null);
  const scanLineRef = useRef<THREE.Mesh>(null);

  // Generate a point cloud that resembles a brain with two hemispheres
  const particles = useMemo(() => {
    const p = [];
    for (let i = 0; i < 6000; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      const r = 2.5 + Math.random() * 0.4;

      let x = r * Math.sin(phi) * Math.cos(theta);
      let y = r * Math.sin(phi) * Math.sin(theta);
      let z = r * Math.cos(phi);

      // Squish along Z slightly to look more oblong (like a brain)
      z *= 1.3;

      // Indent the middle to separate hemispheres
      if (Math.abs(x) < 0.4) {
        y *= 0.6;
        z *= 0.8;
      }

      p.push(x, y, z);
    }
    return new Float32Array(p);
  }, []);

  // Generate anomaly cluster positioned by the Council Consensus result
  const anomalyData = useMemo(() => {
    const pos = consensusResult?.anomalyPosition || [-1.4, -0.5, 0.8];
    const radius = consensusResult?.anomalyRadius || 0.5;
    const centerX = pos[0];
    const centerY = pos[1];
    const centerZ = pos[2];

    const p = [];
    const count = consensusResult ? 1200 : 800;

    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      const r = Math.random() * radius;

      const x = centerX + (r * Math.sin(phi) * Math.cos(theta));
      const y = centerY + (r * Math.sin(phi) * Math.sin(theta));
      const z = centerZ + (r * Math.cos(phi));

      p.push(x, y, z);
    }
    return new Float32Array(p);
  }, [consensusResult]);

  // Generate attention heatmap gradient around the focus area
  const heatmapData = useMemo(() => {
    if (!consensusResult) return null;

    const focusCenter = consensusResult.attentionResult.focusCenter;
    const focusRadius = consensusResult.attentionResult.focusRadius;

    const positions: number[] = [];
    const colors: number[] = [];

    // Create points around the focus area with gradient coloring
    for (let i = 0; i < 2000; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      const r = Math.random() * (focusRadius * 2.5);

      const x = focusCenter[0] + (r * Math.sin(phi) * Math.cos(theta));
      const y = focusCenter[1] + (r * Math.sin(phi) * Math.sin(theta));
      const z = focusCenter[2] + (r * Math.cos(phi));

      positions.push(x, y, z);

      // Color gradient: hot pink (center) → cyan (edge)
      const intensity = Math.max(0, 1 - r / (focusRadius * 2.5));
      colors.push(
        0.94 * intensity + 0.02 * (1 - intensity), // R
        0.27 * intensity + 0.71 * (1 - intensity), // G
        0.44 * intensity + 0.83 * (1 - intensity), // B
      );
    }

    return {
      positions: new Float32Array(positions),
      colors: new Float32Array(colors),
    };
  }, [consensusResult]);

  // Determine anomaly color based on severity
  const anomalyColor = useMemo(() => {
    if (!consensusResult) return '#ef4444';
    switch (consensusResult.severity) {
      case 'critical': return '#ef4444';
      case 'warning': return '#f59e0b';
      case 'info': return '#06b6d4';
      case 'clear': return '#10b981';
      default: return '#ef4444';
    }
  }, [consensusResult]);

  useFrame((state) => {
    const time = state.clock.getElapsedTime();
    if (pointsRef.current) {
      pointsRef.current.rotation.y = time * 0.15;
      pointsRef.current.rotation.x = Math.sin(time * 0.1) * 0.05;
    }
    if (anomalyRef.current) {
      anomalyRef.current.rotation.copy(pointsRef.current!.rotation);
      // Pulse the anomaly
      (anomalyRef.current.material as THREE.PointsMaterial).size = 0.04 + Math.sin(time * 3) * 0.02;
    }
    if (heatmapRef.current) {
      heatmapRef.current.rotation.copy(pointsRef.current!.rotation);
      (heatmapRef.current.material as THREE.PointsMaterial).opacity = 0.3 + Math.sin(time * 2) * 0.15;
    }
    if (scanLineRef.current) {
      scanLineRef.current.position.y = Math.sin(time * 1.5) * 3;
    }
  });

  return (
    <group scale={0.8}>
      {/* Brain point cloud */}
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[particles, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.02}
          color="#06b6d4"
          transparent
          opacity={0.3}
          blending={THREE.AdditiveBlending}
        />
      </points>

      {/* Anomaly Highlight — positioned by Council Consensus */}
      <points ref={anomalyRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[anomalyData, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.05}
          color={anomalyColor}
          transparent
          opacity={0.8}
          blending={THREE.AdditiveBlending}
        />
      </points>

      {/* Attention Heatmap Gradient — from Attention Net */}
      {heatmapData && (
        <points ref={heatmapRef}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[heatmapData.positions, 3]}
            />
            <bufferAttribute
              attach="attributes-color"
              args={[heatmapData.colors, 3]}
            />
          </bufferGeometry>
          <pointsMaterial
            size={0.03}
            vertexColors
            transparent
            opacity={0.45}
            blending={THREE.AdditiveBlending}
          />
        </points>
      )}

      {/* Scanning laser effect */}
      <mesh ref={scanLineRef} position={[0, 0, 0]}>
        <boxGeometry args={[6, 0.02, 6]} />
        <meshBasicMaterial color="#10b981" transparent opacity={0.3} />
      </mesh>
    </group>
  );
}
