import AnalysisLayout from '../components/layout/AnalysisLayout';
import BrainModel from '../components/3d/BrainModel';

export default function MRIAnalysis() {
  return (
    <AnalysisLayout 
      title="MRI Analysis"
      modality="MRI"
      reportElementId="mri-report"
      pdfFilename="CouncilMed_MRI_Report.pdf"
      emptyStateText="Upload an MRI scan to begin multi-model Council Analysis"
      Viewer3D={BrainModel}
    />
  );
}
