import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, File, CheckCircle2 } from 'lucide-react';
import { GlassCard } from './GlassCard';
import { toast } from 'sonner';
import { useNavigate } from 'react-router-dom';
import { diagnosticEngine } from '../../services/inference/DiagnosticEngine';
interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultModality?: string;
}

export function UploadModal({ isOpen, onClose, defaultModality = 'mri' }: UploadModalProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [modality, setModality] = useState(defaultModality);
  const navigate = useNavigate();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = (selectedFile: File) => {
    // 1. Size Validation (Max 50MB)
    const MAX_SIZE = 50 * 1024 * 1024;
    if (selectedFile.size > MAX_SIZE) {
      toast.error('File size too large', { description: 'Maximum allowed size is 50MB for security reasons.' });
      return;
    }

    // 2. Type/Extension Validation
    const validExtensions = ['.dcm', '.nii', '.nrrd', '.png', '.jpg', '.jpeg'];
    const validMimes = ['image/png', 'image/jpeg', 'application/dicom'];
    const fileExtension = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
    
    if (!validExtensions.includes(fileExtension) && !validMimes.includes(selectedFile.type)) {
      toast.error('Invalid file type', { description: 'Please upload a valid DICOM, NIfTI, or medical imaging file.' });
      return;
    }

    setFile(selectedFile);
  };

  const simulateUpload = () => {
    if (!file) return;
    setUploading(true);
    setProgress(0);
    
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setUploading(false);
          
          const nextRoute = modality === 'mri' ? '/mri' : modality === 'ct' ? '/ct' : '/xray';
          toast.success('Scan Uploaded: Executing Council AI Analysis', {
            description: `${file.name} sent to the diagnostic pipeline.`,
          });
          
          diagnosticEngine.processImage(file).catch(console.error);
          navigate(nextRoute);

          setTimeout(() => {
            onClose();
            setFile(null);
            setProgress(0);
          }, 1000);
          return 100;
        }
        return prev + 10;
      });
    }, 200);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
          />
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
            <motion.div
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
              className="w-full max-w-lg pointer-events-auto"
            >
              <GlassCard className="p-0 overflow-hidden bg-[#0d1324]/90">
                <div className="flex items-center justify-between p-6 border-b border-white/10">
                  <h2 className="text-xl font-bold text-white">Upload New Scan</h2>
                  <button 
                    onClick={onClose}
                    className="p-2 text-zinc-400 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="px-6 pt-4">
                  <label className="block text-sm font-medium text-zinc-400 mb-2">Scan Modality</label>
                  <div className="flex space-x-2">
                    {['mri', 'ct', 'xray'].map((type) => (
                      <button
                        key={type}
                        onClick={() => setModality(type)}
                        className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                          modality === type 
                            ? 'bg-[var(--color-accent-cyan)]/20 text-[var(--color-accent-cyan)] border border-[var(--color-accent-cyan)]/50' 
                            : 'bg-white/5 text-zinc-400 border border-white/10 hover:bg-white/10 hover:text-zinc-300'
                        }`}
                      >
                        {type === 'mri' ? 'MRI' : type === 'ct' ? 'CT Scan' : 'X-Ray'}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="p-6">
                  {!file ? (
                    <div
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      className={`
                        border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300
                        ${isDragging 
                          ? 'border-[var(--color-accent-cyan)] bg-[var(--color-accent-cyan)]/10' 
                          : 'border-white/20 hover:border-white/40 hover:bg-white/5'
                        }
                      `}
                    >
                      <div className="w-16 h-16 mx-auto bg-white/5 rounded-full flex items-center justify-center mb-4">
                        <Upload className={`w-8 h-8 ${isDragging ? 'text-[var(--color-accent-cyan)]' : 'text-zinc-400'}`} />
                      </div>
                      <h3 className="text-lg font-medium text-white mb-2">Drag & Drop DICOM files</h3>
                      <p className="text-sm text-zinc-400 mb-6">or select files from your computer</p>
                      
                      <label className="cursor-pointer">
                        <span className="px-6 py-2.5 bg-white/10 hover:bg-white/20 text-white font-medium rounded-xl transition-colors">
                          Browse Files
                        </span>
                        <input 
                          type="file" 
                          className="hidden" 
                          accept=".dcm,.nii,.nrrd,image/*" 
                          onChange={(e) => e.target.files && handleFile(e.target.files[0])}
                        />
                      </label>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="flex items-center space-x-4 p-4 rounded-xl bg-white/5 border border-white/10">
                        <div className="p-3 bg-[var(--color-accent-cyan)]/20 rounded-lg text-[var(--color-accent-cyan)]">
                          <File className="w-6 h-6" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-white truncate">{file.name}</p>
                          <p className="text-xs text-zinc-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                        </div>
                        {!uploading && progress === 0 && (
                          <button onClick={() => setFile(null)} className="p-2 text-zinc-400 hover:text-[var(--color-accent-ruby)] transition-colors">
                            <X className="w-5 h-5" />
                          </button>
                        )}
                      </div>

                      {uploading && (
                        <div className="space-y-2">
                          <div className="flex justify-between text-xs font-medium">
                            <span className="text-zinc-400">Uploading and encrypting...</span>
                            <span className="text-[var(--color-accent-cyan)]">{progress}%</span>
                          </div>
                          <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                            <motion.div 
                              className="h-full bg-[var(--color-accent-cyan)]"
                              initial={{ width: 0 }}
                              animate={{ width: `${progress}%` }}
                            />
                          </div>
                        </div>
                      )}

                      <div className="flex space-x-3 pt-4 border-t border-white/10">
                        <button 
                          onClick={onClose}
                          disabled={uploading}
                          className="flex-1 px-4 py-2.5 bg-white/5 hover:bg-white/10 text-white font-medium rounded-xl transition-colors disabled:opacity-50"
                        >
                          Cancel
                        </button>
                        <button 
                          onClick={simulateUpload}
                          disabled={uploading || progress === 100}
                          className="flex-1 px-4 py-2.5 bg-[var(--color-accent-cyan)] hover:bg-[var(--color-accent-teal)] text-[#0d1324] font-bold rounded-xl transition-colors disabled:opacity-50 flex items-center justify-center space-x-2"
                        >
                          {progress === 100 ? (
                            <>
                              <CheckCircle2 className="w-5 h-5" />
                              <span>Complete</span>
                            </>
                          ) : (
                            <>
                              <Upload className="w-5 h-5" />
                              <span>Process Scan</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </GlassCard>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
