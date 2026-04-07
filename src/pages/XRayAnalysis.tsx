import { GlassCard } from '../components/ui/GlassCard';
import { StatusBadge } from '../components/ui/StatusBadge';
import { ConfidenceBar } from '../components/ui/ConfidenceBar';
import { motion } from 'framer-motion';
import { Share2, Download, Upload, Layers, Play, Maximize } from 'lucide-react';
import { toast } from 'sonner';
import { generatePDFReport } from '../utils/reportGenerator';
import { Canvas } from '@react-three/fiber';
import SkeletonModel from '../components/3d/SkeletonModel';
import { useWorkstationSettings } from '../hooks/useWorkstationSettings';
import type { CouncilConsensusResult } from '../services/inference/types';
import { PathologyClass, SeverityLevel } from '../services/inference/types';

export default function XRayAnalysis() {
  const { settings } = useWorkstationSettings();

  const mockResult: CouncilConsensusResult = {
    primaryDiagnosis: PathologyClass.GLIOMA_LOW, // Mapping to a pathology for now
    severity: SeverityLevel.WARNING,
    overallConfidence: 76.5,
    anomalyPosition: [0, 0, 0],
    anomalyRadius: 0,
    affectedRegion: 'PARIETAL_LEFT',
    densenetResult: {
      modelName: 'DenseNet-121',
      classifications: [{ pathology: PathologyClass.GLIOMA_LOW, probability: 0.76 }],
      primaryDiagnosis: PathologyClass.GLIOMA_LOW,
      confidence: 76.5,
      inferenceTimeMs: 140,
      featureVector: []
    },
    attentionResult: {
      modelName: 'Attention-Net',
      attentionMap: [],
      focusRegion: 'PARIETAL_LEFT',
      focusCenter: [0, 0, 0],
      focusRadius: 0,
      confidence: 72.0,
      inferenceTimeMs: 90
    },
    swinResult: {
      modelName: 'Swin-UNETR',
      segmentationMask: [],
      tumorVolumeMm3: 120,
      tumorCenter: [0, 0, 0],
      tumorRadius: 2,
      segmentedRegions: [],
      confidence: 74.5,
      inferenceTimeMs: 160
    },
    modelAgreement: 0.85,
    councilNotes: ['Hairline Fracture - Tibia suspected.', 'Minimal displacement, cortices compromised.'],
    totalInferenceTimeMs: 390
  };

  return (
    <div id="xray-report" className="max-w-7xl mx-auto h-full space-y-6">
      <div className="flex items-center justify-between shrink-0">
        <div>
           <div className="flex items-center space-x-3">
             <h1 className="text-2xl font-bold text-white tracking-tight">X-Ray Analysis (Skeletal)</h1>
             <StatusBadge status="warning">Fracture Detected</StatusBadge>
           </div>
           <p className="text-sm text-zinc-400 mt-1">Patient: PT-8803 • Scan Date: Oct 24, 2026</p>
        </div>
        <div className="flex space-x-2">
          <button 
             onClick={() => {
                const event = new CustomEvent('open-upload', { detail: { modality: 'xray' } });
                window.dispatchEvent(event);
             }}
             className="px-3 py-2 bg-white/10 hover:bg-white/20 text-white border border-white/20 font-medium rounded-lg transition-colors flex items-center space-x-2"
          >
            <Upload className="w-4 h-4" />
            <span className="hidden sm:inline">Upload X-Ray</span>
          </button>

          <button onClick={() => toast('Secure link copied to clipboard')} className="p-2 bg-white/5 hover:bg-white/10 rounded-lg text-zinc-300 transition-colors border border-white/10"><Share2 className="w-4 h-4" /></button>
          <button onClick={() => toast.success('Downloading raw DICOM files...')} className="p-2 bg-white/5 hover:bg-white/10 rounded-lg text-zinc-300 transition-colors border border-white/10"><Download className="w-4 h-4" /></button>
          <button 
             onClick={() => generatePDFReport('xray-report', 'CouncilMed_PT8803_XRay_Report.pdf', {
               diagnosis: mockResult,
               doctorName: settings.doctorName,
               clinicalId: settings.clinicalId,
               scanType: 'X-Ray'
             })} 
             className="px-4 py-2 bg-[var(--color-accent-cyan)] hover:bg-[var(--color-accent-teal)] text-[#0d1324] font-semibold rounded-lg transition-colors flex items-center space-x-2"
          >
            <span>Generate PDF Report</span>
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GlassCard className="h-[500px] flex flex-col p-0 relative overflow-hidden">
          <div className="absolute top-4 left-4 right-4 z-10 flex justify-between items-center pointer-events-none">
            <div className="bg-black/40 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 pointer-events-auto flex items-center space-x-2">
              <Layers className="w-4 h-4 text-zinc-400" />
              <span className="text-xs font-medium text-white">3D Skeletal Reconstruction</span>
            </div>
            <div className="flex space-x-2 pointer-events-auto">
               <button onClick={() => toast('Playback controls engaged')} className="p-1.5 bg-black/40 hover:bg-black/60 backdrop-blur-md rounded-lg border border-white/10 text-white transition-colors"><Play className="w-4 h-4" /></button>
               <button onClick={() => toast('Entering fullscreen view')} className="p-1.5 bg-black/40 hover:bg-black/60 backdrop-blur-md rounded-lg border border-white/10 text-white transition-colors"><Maximize className="w-4 h-4" /></button>
            </div>
          </div>
          
          <div className="flex-1 w-full bg-gradient-to-b from-[#0d1324] to-[#04060b] relative flex items-center justify-center">
            <Canvas camera={{ position: [0, 0, 8] }} gl={{ preserveDrawingBuffer: true }}>
              <ambientLight intensity={1} />
              <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} />
              <SkeletonModel />
            </Canvas>
          </div>
        </GlassCard>
        
        <motion.div 
          className="space-y-6"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <GlassCard>
             <h3 className="text-lg font-bold text-white mb-4">Diagnostic Result</h3>
             <ul className="space-y-4 mb-6">
                <li className="flex flex-col">
                  <span className="text-white font-medium">Hairline Fracture - Tibia</span>
                  <span className="text-sm text-zinc-400">Minimal displacement, cortices compromised.</span>
                  <div className="mt-2 text-[var(--color-accent-amber)]"><ConfidenceBar score={76.5} /></div>
                </li>
             </ul>
          </GlassCard>
        </motion.div>
      </div>
    </div>
  );
}
