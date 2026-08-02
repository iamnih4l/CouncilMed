import AnalysisLayout from '../components/layout/AnalysisLayout';
import TorsoModel from '../components/3d/TorsoModel';

export default function CTAnalysis() {
  return (
    <AnalysisLayout 
      title="CT Analysis (Abdominal)"
      modality="CT"
      reportElementId="ct-report"
      pdfFilename="CouncilMed_CT_Report.pdf"
      emptyStateText="Upload a CT scan to begin multi-model Council Analysis"
      Viewer3D={TorsoModel}
    />
  );
}
