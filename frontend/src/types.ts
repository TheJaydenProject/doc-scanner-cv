export interface ScanResult {
  text: string;
  char_count: number;
  word_count: number;
  processing_time_ms: number;
  warped_image_b64: string;
  detection_count: number;
  doc_type: string;
  doc_type_confidence: number;
}
