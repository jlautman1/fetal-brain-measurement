#!/usr/bin/env python3
"""
Dummy OpenRecon i2i handler for testing deployment and comparison
This creates a minimal working OpenRecon handler that can be deployed to verify the process
"""

import logging
import time
import os
import json
import ismrmrd
import numpy as np


def process_image(group, config, metadata):
    """
    Dummy i2i processing function for OpenRecon testing
    
    Args:
        group: ISMRMRD group containing image data
        config: Configuration dictionary
        metadata: Image metadata
        
    Returns:
        ISMRMRD group with processed images
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DummyOpenRecon")
    
    logger.info("🎭 DUMMY OpenRecon Handler - Starting Processing")
    logger.info(f"📊 Config: {config}")
    logger.info(f"📋 Metadata keys: {list(metadata.keys()) if metadata else 'None'}")
    
    # Create output directory for dummy files
    output_dir = "/tmp/share/dummy_measurements"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image in the group
    for idx in range(group.data.data.shape[0]):
        logger.info(f"🖼️  Processing dummy image {idx + 1}/{group.data.data.shape[0]}")
        
        # Get original image data
        image_data = group.data.data[idx, :, :, :]
        logger.info(f"📐 Image shape: {image_data.shape}")
        logger.info(f"🔢 Image dtype: {image_data.dtype}")
        logger.info(f"📊 Image range: [{np.min(image_data):.2f}, {np.max(image_data):.2f}]")
        
        # Dummy processing: Add a simple border to show we processed it
        processed_image = image_data.copy()
        
        # Add bright border (10% of max value)
        border_value = np.max(image_data) * 0.1
        if len(processed_image.shape) >= 2:
            processed_image[0:5, :] = border_value      # Top border
            processed_image[-5:, :] = border_value      # Bottom border  
            processed_image[:, 0:5] = border_value      # Left border
            processed_image[:, -5:] = border_value      # Right border
        
        # Update the image data
        group.data.data[idx, :, :, :] = processed_image
        
        # Create dummy measurement data
        dummy_measurements = {
            "processing_time": time.time(),
            "image_index": idx,
            "dummy_measurement_1": 42.5 + idx,
            "dummy_measurement_2": 37.2 - idx * 0.5,
            "dummy_measurement_3": 15.8 + idx * 1.2,
            "processing_status": "success",
            "handler_type": "dummy_openrecon",
            "note": "This is a dummy measurement for testing purposes"
        }
        
        # Save dummy results
        result_file = os.path.join(output_dir, f"dummy_results_{idx}.json")
        with open(result_file, 'w') as f:
            json.dump(dummy_measurements, f, indent=2)
        
        logger.info(f"💾 Saved dummy results to: {result_file}")
    
    # Add dummy metadata to each image header
    for idx in range(group.data.headers.shape[0]):
        # Add custom fields to ISMRMRD header
        header = group.data.headers[idx]
        
        # Set some dummy user parameters (these will show up in DICOM)
        header.user_int[0] = 12345  # Dummy ID
        header.user_int[1] = idx    # Image index
        header.user_float[0] = 42.5 + idx  # Dummy measurement 1
        header.user_float[1] = 37.2 - idx * 0.5  # Dummy measurement 2
        
        # Update acquisition timestamp
        header.acquisition_time_stamp = int(time.time() * 1000000)  # microseconds
    
    # Create summary file
    summary_data = {
        "handler_name": "Dummy OpenRecon Handler",
        "version": "1.0.0",
        "processing_time": time.time(),
        "total_images_processed": group.data.data.shape[0],
        "output_directory": output_dir,
        "status": "completed_successfully",
        "dummy_measurements": {
            "average_measurement_1": np.mean([42.5 + i for i in range(group.data.data.shape[0])]),
            "average_measurement_2": np.mean([37.2 - i * 0.5 for i in range(group.data.data.shape[0])]),
            "total_processing_time_seconds": 0.1  # Dummy fast processing
        }
    }
    
    summary_file = os.path.join(output_dir, "dummy_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"📄 Created summary file: {summary_file}")
    logger.info("✅ DUMMY OpenRecon Handler - Processing Complete!")
    
    return group


# For testing when run directly
if __name__ == "__main__":
    print("🎭 Dummy OpenRecon Handler - Test Mode")
    print("This is a dummy handler for testing OpenRecon deployment")
    print("It adds borders to images and creates dummy measurement files")
    print("✅ Handler is ready for deployment testing")
