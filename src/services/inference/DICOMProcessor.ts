import * as dicomParser from 'dicom-parser';
import * as tf from '@tensorflow/tfjs';

export interface DICOMResult {
  tensor: tf.Tensor3D;
  metadata: {
    patientName?: string;
    modality?: string;
    seriesDescription?: string;
    studyDate?: string;
    rows: number;
    columns: number;
    bitsStored: number;
    windowCenter?: number;
    windowWidth?: number;
  };
  originalPixelData: Uint16Array | Uint8Array;
}

/**
 * DICOM Processor Service
 * Handles parsing of medical imaging files and conversion to AI-ready tensors.
 */
export class DICOMProcessor {
  /**
   * Processes a raw DICOM file into a normalized TensorFlow tensor.
   */
  static async process(file: File): Promise<DICOMResult> {
    const arrayBuffer = await file.arrayBuffer();
    const byteArray = new Uint8Array(arrayBuffer);

    // 1. Parse DICOM Dataset
    const dataSet = dicomParser.parseDicom(byteArray);

    // 2. Extract Metadata
    const rows = dataSet.uint16('x00280010');
    const columns = dataSet.uint16('x00280011');

    if (rows === undefined || columns === undefined) {
      throw new Error('DICOM file is missing required dimensions (Rows/Columns).');
    }

    const metadata = {
      patientName: dataSet.string('x00100010'),
      modality: dataSet.string('x00080060'),
      seriesDescription: dataSet.string('x0008103e'),
      studyDate: dataSet.string('x00080020'),
      rows: rows,
      columns: columns,
      bitsStored: dataSet.uint16('x00280101') || 8,
      windowCenter: parseFloat(dataSet.string('x00281050') || '0'),
      windowWidth: parseFloat(dataSet.string('x00281051') || '0'),
    };

    // 3. Extract Pixel Data (Assuming uncompressed Little Endian for now)
    const pixelDataElement = dataSet.elements['x7fe00010'];
    if (!pixelDataElement) {
      throw new Error('No pixel data found in DICOM file.');
    }

    // Get the pixel data as a TypedArray
    let pixelData: Uint16Array | Uint8Array;
    if (metadata.bitsStored > 8) {
      pixelData = new Uint16Array(
        byteArray.buffer,
        pixelDataElement.dataOffset,
        pixelDataElement.length / 2
      );
    } else {
      pixelData = new Uint8Array(
        byteArray.buffer,
        pixelDataElement.dataOffset,
        pixelDataElement.length
      );
    }

    // 4. Normalization & Tensor Conversion
    // We normalize to [0, 1] based on the window or max value
    const numPixels = metadata.rows * metadata.columns;
    const floatData = new Float32Array(numPixels);
    
    let min = Infinity;
    let max = -Infinity;

    // Use Windowing if available, otherwise use min/max of the image
    if (metadata.windowWidth > 0) {
      const low = metadata.windowCenter - metadata.windowWidth / 2;
      const high = metadata.windowCenter + metadata.windowWidth / 2;
      
      for (let i = 0; i < numPixels; i++) {
        const val = pixelData[i];
        floatData[i] = Math.min(Math.max((val - low) / (high - low), 0), 1);
      }
    } else {
      for (let i = 0; i < numPixels; i++) {
        const val = pixelData[i];
        if (val < min) min = val;
        if (val > max) max = val;
      }
      const range = max - min || 1;
      for (let i = 0; i < numPixels; i++) {
        floatData[i] = (pixelData[i] - min) / range;
      }
    }

    // 5. Create 3D Tensor [Rows, Cols, 1]
    const tensor = tf.tensor3d(floatData, [metadata.rows, metadata.columns, 1]);

    return {
      tensor,
      metadata,
      originalPixelData: pixelData
    };
  }
}
