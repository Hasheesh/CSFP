import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import time
from pathlib import Path

# PaddleOCR imports for explicit model loading
from paddleocr import PaddleOCR
import logging

class ImageReader:
    """
    Advanced Image Reader using PaddleOCR with explicit model loading.
    Designed for university project demonstrating deep understanding of OCR components.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 lang: str = "en",
                 ocr_version: str = "PP-OCRv5",
                 device: str = "cpu",
                 cpu_threads: int = 10):
        """
        Initialize ImageReader with explicit PaddleOCR configuration.
        
        Args:
            model_path: Custom model path (if None, uses default PaddleOCR models)
            lang: Language for OCR ('en', 'ch', 'french', 'german', 'korean', 'japan')
            ocr_version: OCR version ('PP-OCRv4', 'PP-OCRv3', 'PP-OCRv2')
            device: Device to use ('cpu', 'gpu')
            use_gpu: Whether to use GPU acceleration
            cpu_threads: Number of CPU threads
        """
        self.model_path = model_path
        self.lang = lang
        self.ocr_version = ocr_version
        self.device = device
        self.cpu_threads = cpu_threads
        
        # Performance tracking
        self.processing_times = []
        self.total_images_processed = 0
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Load OCR model with explicit configuration
        self._load_ocr_model()
        
    def _load_ocr_model(self):
        """Explicitly load PaddleOCR model with custom configuration."""
        try:
            self.logger.info("Loading PaddleOCR model with explicit configuration...")
            
            # Configure model parameters
            model_config = {
                'lang': self.lang,
            }
            
          
            #     model_config['use_gpu'] = False
            #     model_config['enable_mkldnn'] = self.enable_mkldnn
            #     model_config['cpu_threads'] = self.cpu_threads
            #     self.logger.info(f"Using CPU with {self.cpu_threads} threads")
            
            # Load custom model if path provided
            # if self.model_path and os.path.exists(self.model_path):
            #     model_config['det_model_dir'] = os.path.join(self.model_path, 'det')
            #     model_config['rec_model_dir'] = os.path.join(self.model_path, 'rec')
            #     model_config['cls_model_dir'] = os.path.join(self.model_path, 'cls')
            #     self.logger.info(f"Loading custom model from: {self.model_path}")
            
            # Initialize PaddleOCR with explicit configuration
            self.ocr = PaddleOCR(**model_config)
            
            self.logger.info("PaddleOCR model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading PaddleOCR model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR performance.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to RGB (PaddleOCR expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Basic preprocessing
            # 1. Resize if too large (for memory efficiency)
            height, width = image_rgb.shape[:2]
            max_dimension = 2048
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            
            # Merge channels and convert back to RGB
            
            
            
            return image_rgb
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def process_input(self, image_path: str, 
                     save_output: bool = True,
                     output_dir: str = "ocr_output") -> Dict:
        """
        Process image input and extract text using PaddleOCR.
        
        Args:
            image_path: Path to the input image
            save_output: Whether to save output images and JSON
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            self.logger.info(f"Processing image: {image_path}")
            
            # Preprocess image
            preprocessed_image = self.preprocess_image(image_path)
            
            # Perform OCR
            ocr_start_time = time.time()
            result = self.ocr.predict(preprocessed_image)
            ocr_time = time.time() - ocr_start_time
            
            # Process results
            extracted_text = []
            confidence_scores = []
            bounding_boxes = []
            
            if result is not None and len(result) > 0:
                for line in result[0]:
                    if line is not None:
                        bbox, (text, confidence) = line
                        extracted_text.append(text)
                        confidence_scores.append(confidence)
                        bounding_boxes.append(bbox)
            
            # Calculate processing time
            total_time = time.time() - start_time
            self.processing_times.append(total_time)
            self.total_images_processed += 1
            
            # Prepare output
            output_data = {
                'image_path': image_path,
                'extracted_text': extracted_text,
                'confidence_scores': confidence_scores,
                'bounding_boxes': bounding_boxes,
                'total_text': ' '.join(extracted_text),
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'text_count': len(extracted_text),
                'processing_time': total_time,
                'ocr_time': ocr_time,
                'preprocessing_time': total_time - ocr_time
            }
            
            # Save outputs if requested
            if save_output:
                self._save_outputs(output_data, preprocessed_image, output_dir)
            
            self.logger.info(f"Successfully processed image. Extracted {len(extracted_text)} text segments.")
            return output_data
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise
    
    def _save_outputs(self, output_data: Dict, image: np.ndarray, output_dir: str):
        """Save OCR outputs to files."""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save JSON results
            json_path = os.path.join(output_dir, f"ocr_result_{Path(output_data['image_path']).stem}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Save annotated image
            if output_data['bounding_boxes']:
                annotated_image = self._draw_ocr_results(image, output_data)
                img_path = os.path.join(output_dir, f"annotated_{Path(output_data['image_path']).stem}.jpg")
                cv2.imwrite(img_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            self.logger.info(f"Outputs saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving outputs: {str(e)}")
    
    
    
    # def batch_process(self, image_paths: List[str], 
    #                  save_output: bool = True,
    #                  output_dir: str = "ocr_batch_output") -> List[Dict]:
    #     """
    #     Process multiple images in batch.
        
    #     Args:
    #         image_paths: List of image paths
    #         save_output: Whether to save outputs
    #         output_dir: Output directory
            
    #     Returns:
    #         List of processing results
    #     """
    #     results = []
        
    #     for i, image_path in enumerate(image_paths):
    #         try:
    #             self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
    #             result = self.process_input(image_path, save_output, output_dir)
    #             results.append(result)
    #         except Exception as e:
    #             self.logger.error(f"Error processing {image_path}: {str(e)}")
    #             results.append({'error': str(e), 'image_path': image_path})
        
    #     return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.processing_times:
            return {'message': 'No images processed yet'}
        
        return {
            'total_images_processed': self.total_images_processed,
            'average_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'total_processing_time': np.sum(self.processing_times)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize ImageReader with custom configuration
    image_reader = ImageReader(
        lang="en",
        ocr_version="PP-OCRv5",
        device="cpu",
        cpu_threads=4,
    )
    
    # Process a single image
    try:
        result = image_reader.process_input("./img.png")
        print("Extracted text:", result['total_text'])
        print("Average confidence:", result['average_confidence'])
        print("Processing time:", result['processing_time'])
    except Exception as e:
        print(f"Error: {e}")
    
    # Get performance stats
    stats = image_reader.get_performance_stats()
    print("Performance stats:", stats)