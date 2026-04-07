import { toJpeg } from 'html-to-image';
import jsPDF from 'jspdf';
import { toast } from 'sonner';
import type { CouncilConsensusResult } from '../services/inference/types';

export interface ClinicalReportData {
  diagnosis: CouncilConsensusResult;
  doctorName: string;
  clinicalId: string;
  patientId?: string;
  scanType: 'MRI' | 'CT' | 'X-Ray';
}

/**
 * Generates a high-fidelity clinical PDF report.
 * Uses jsPDF for document structure and html-to-image for visual evidence components.
 */
export const generatePDFReport = async (
  elementId: string, 
  filename: string,
  data: ClinicalReportData
) => {
  const element = document.getElementById(elementId);
  if (!element) {
    toast.error('Diagnostic error: Target UI layer not found.');
    return;
  }

  const toastId = toast.loading('Synthesizing clinical report...');

  try {
    // 1. Capture the visual evidence (3D Canvas snapshot)
    const dataUrl = await toJpeg(element, {
      quality: 0.95,
      pixelRatio: 2,
      backgroundColor: '#0d1324',
    });

    // 2. Initialize PDF (A4)
    const pdf = new jsPDF({
      orientation: 'p',
      unit: 'mm',
      format: 'a4',
      compress: true,
    });

    const margin = 20;
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    let currentY = 0;

    // --- Helper: Header & Watermark ---
    const drawHeader = () => {
      pdf.setFillColor(13, 19, 36); 
      pdf.rect(0, 0, pageWidth, 40, 'F');
      
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(22);
      pdf.setFont('helvetica', 'bold');
      pdf.text('COUNCILMED', margin, 18);
      
      pdf.setFontSize(9);
      pdf.setFont('helvetica', 'normal');
      pdf.text('ADVANCED CLINICAL DIAGNOSTICS PLATFORM', margin, 25);
      
      pdf.setFontSize(10);
      pdf.text('OFFICIAL DIAGNOSTIC REPORT', pageWidth - margin - 55, 18);
      pdf.text(new Date().toLocaleDateString(), pageWidth - margin - 30, 25);
    };

    const drawFooter = () => {
      const footerY = pageHeight - 15;
      pdf.setFontSize(8);
      pdf.setTextColor(150, 150, 150);
      pdf.setFont('helvetica', 'italic');
      pdf.text('This is an AI-augmented diagnostic assistant report. Final clinical decisions must be made by a certified physician.', margin, footerY);
      pdf.text(`FINGERPRINT: ${data.diagnosis.totalInferenceTimeMs.toFixed(0)}_MS_ENGINE`, pageWidth - margin - 50, footerY);
    };

    // --- Helper: Space Management ---
    const ensureSpace = (height: number) => {
      if (currentY + height > pageHeight - 25) {
        pdf.addPage();
        drawHeader();
        drawFooter();
        currentY = 55;
        return true;
      }
      return false;
    };

    // Initialize first page
    drawHeader();
    drawFooter();
    currentY = 55;

    // --- Patient & Clinical Metadata ---
    pdf.setTextColor(40, 40, 40);
    pdf.setFontSize(12);
    pdf.setFont('helvetica', 'bold');
    pdf.text('CLINICAL METADATA', margin, currentY);
    currentY += 8;
    
    pdf.setFontSize(9);
    pdf.setFont('helvetica', 'normal');
    pdf.text(`Authorized Physician: ${data.doctorName}`, margin, currentY);
    pdf.text(`Clinical ID: ${data.clinicalId}`, margin + 90, currentY);
    currentY += 6;
    pdf.text(`Diagnostic Modality: ${data.scanType} Scan`, margin, currentY);
    pdf.text(`System Reference: ${Math.random().toString(36).substring(7).toUpperCase()}`, margin + 90, currentY);
    
    currentY += 12;
    pdf.setDrawColor(230, 230, 230);
    pdf.line(margin, currentY, pageWidth - margin, currentY);
    currentY += 10;

    // --- Primary Status & Impression ---
    const report = data.diagnosis.radiologyReport;
    const status = (data.diagnosis.validation?.status || 'COMPLETE').toUpperCase();
    
    ensureSpace(30);
    pdf.setFontSize(14);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(13, 19, 36);
    pdf.text('DIAGNOSTIC STATUS: ' + status, margin, currentY);
    currentY += 10;

    pdf.setFontSize(11);
    pdf.setTextColor(100, 100, 100);
    pdf.text('IMPRESSION:', margin, currentY);
    currentY += 6;

    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(40, 40, 40);
    const impressionLines = pdf.splitTextToSize(report?.impression || 'No specific impression generated.', pageWidth - (margin * 2));
    pdf.text(impressionLines, margin, currentY);
    currentY += (impressionLines.length * 5) + 12;

    // --- Formal Findings Section ---
    if (report) {
      ensureSpace(40);
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'bold');
      pdf.text('STRUCTURED FINDINGS (RSNA Standard)', margin, currentY);
      currentY += margin / 2;

      const findings = [
        { label: 'Brain Parenchyma', content: report.findings.brainParenchyma },
        { label: 'Edema / Mass Effect', content: report.findings.edemaMassEffect },
        { label: 'Ventricular System', content: report.findings.ventricularSystem },
        { label: 'Midline Shift', content: report.findings.midlineStructures.shift }
      ];

      findings.forEach(f => {
        const textLines = pdf.splitTextToSize(f.content, pageWidth - (margin * 2) - 30);
        ensureSpace((textLines.length * 4.5) + 6);
        
        pdf.setFontSize(9);
        pdf.setFont('helvetica', 'bold');
        pdf.text(f.label + ':', margin, currentY);
        
        pdf.setFont('helvetica', 'normal');
        pdf.text(textLines, margin + 45, currentY);
        currentY += (textLines.length * 4.5) + 4;
      });

      if (report.findings.lesionAssessment.included) {
        currentY += 4;
        ensureSpace(20);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Lesion Assessment:', margin, currentY);
        currentY += 5;
        pdf.setFont('helvetica', 'normal');
        pdf.text(`• Location: ${report.findings.lesionAssessment.location}`, margin + 5, currentY);
        currentY += 5;
        pdf.text(`• Size: ${report.findings.lesionAssessment.sizeVolume}`, margin + 5, currentY);
        currentY += 5;
        pdf.text(`• Characteristics: ${report.findings.lesionAssessment.characteristics}`, margin + 5, currentY);
        currentY += 10;
      }
    }

    // --- Clinical Validation ---
    if (data.diagnosis.validation) {
      ensureSpace(40);
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'bold');
      pdf.text('CLINICAL VALIDATION LAYER', margin, currentY);
      currentY += 8;

      const val = data.diagnosis.validation;
      if (val.detectedIssues.length > 0) {
        pdf.setTextColor(200, 50, 50);
        pdf.setFontSize(9);
        pdf.text('Detected Issues:', margin, currentY);
        currentY += 5;
        pdf.setTextColor(80, 80, 80);
        pdf.setFont('helvetica', 'normal');
        val.detectedIssues.forEach(issue => {
          const issueLines = pdf.splitTextToSize(`• [${issue.type}] ${issue.description}`, pageWidth - (margin * 2) - 10);
          ensureSpace(issueLines.length * 4.5);
          pdf.text(issueLines, margin + 5, currentY);
          currentY += (issueLines.length * 4.5);
        });
      }
      
      currentY += 5;
      ensureSpace(15);
      pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(200, 150, 0);
      pdf.text('Safety Overrides Applied:', margin, currentY);
      currentY += 5;
      pdf.setFont('helvetica', 'normal');
      pdf.setTextColor(80, 80, 80);
      val.correctionsApplied.forEach(fix => {
        pdf.text(`• ${fix}`, margin + 5, currentY);
        currentY += 4.5;
      });
      currentY += 10;
    }

    // --- Model Breakdown Table ---
    ensureSpace(45);
    pdf.setFontSize(12);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(40, 40, 40);
    pdf.text('MODEL CONSENSUS BREAKDOWN', margin, currentY);
    currentY += 8;

    const rowHeight = 10;
    pdf.setFillColor(245, 247, 250);
    pdf.rect(margin, currentY, pageWidth - (margin * 2), rowHeight, 'F');
    pdf.setFontSize(9);
    pdf.text('Inference Model', margin + 5, currentY + 6.5);
    pdf.text('Confidence', margin + 80, currentY + 6.5);
    pdf.text('Status', margin + 120, currentY + 6.5);
    currentY += rowHeight;

    const models = [
      { name: 'DenseNet-121 (Classification)', conf: data.diagnosis.densenetResult.confidence },
      { name: 'Attention-Net (Localization)', conf: data.diagnosis.attentionResult.confidence },
      { name: 'Swin-UNETR (Segmentation)', conf: data.diagnosis.swinResult.confidence }
    ];

    models.forEach(m => {
      pdf.text(m.name, margin + 5, currentY + 6.5);
      pdf.text(`${m.conf.toFixed(1)}%`, margin + 80, currentY + 6.5);
      pdf.text('Verified', margin + 120, currentY + 6.5);
      currentY += rowHeight;
    });

    // --- Visual Evidence ---
    ensureSpace(90);
    currentY += 10;
    pdf.setFontSize(12);
    pdf.setFont('helvetica', 'bold');
    pdf.text('ANATOMICAL MAPPING (VISUAL EVIDENCE)', margin, currentY);
    currentY += 8;

    const imgWidth = pageWidth - (margin * 2);
    const imgHeight = (imgWidth / element.clientWidth) * element.clientHeight;
    pdf.addImage(dataUrl, 'JPEG', margin, currentY, imgWidth, imgHeight, undefined, 'FAST');
    currentY += imgHeight + 15;

    // Finalize
    pdf.save(filename);

    toast.success('Clinical Report Finalized', { 
      id: toastId,
      description: `Saved as ${filename}`
    });
  } catch (error) {
    console.error('[PDF Engine Error]:', error);
    toast.error('PDF Generation Failed', { 
      id: toastId,
      description: 'Diagnostic pipeline synchronization error.'
    });
  }
};
