import AnalysisLayout from '../components/layout/AnalysisLayout';
import SkeletonModel from '../components/3d/SkeletonModel';

export default function XRayAnalysis() {
  return (
    <AnalysisLayout 
      title="X-Ray Analysis (Skeletal)"
      modality="X-Ray (Skeletal)"
      reportElementId="xray-report"
      pdfFilename="CouncilMed_XRay_Report.pdf"
      emptyStateText="Upload an X-Ray scan to begin multi-model Council Analysis"
      Viewer3D={SkeletonModel}
    />
  );
}
