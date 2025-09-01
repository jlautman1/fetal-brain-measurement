# OpenRecon Fetal Brain Measurement Integration

This directory contains the integration of the Fetal Brain Measurement Pipeline with Siemens OpenRecon MRI systems using the i2i (Image-to-Image) processing framework.

## 🏗️ Architecture

The integration consists of:

1. **i2i Handler**: Python OpenRecon i2i handler that processes images in real-time
2. **Fetal Brain Pipeline**: AI-powered measurement pipeline integrated within the handler
3. **Docker Container**: Unified environment containing all dependencies and models
4. **OpenRecon Interface**: Full compatibility with Siemens OpenRecon workflow

## 📁 Files Overview

### Core Integration Files
- `Dockerfile.openrecon.integrated` - Main Docker image for OpenRecon integration
- `openrecon.py` - Main i2i handler for fetal brain processing (correct OpenRecon structure)
- `openrecon.json` - Configuration file for the OpenRecon module

### Build and Deployment Scripts  
- `build-openrecon-image.bat` - Windows script to build the Docker image
- `run-openrecon-server.bat` - Windows script to run the OpenRecon server
- `test-client.py` - Test client for connectivity validation
- `validate-setup.py` - Pre-deployment validation script

### Documentation
- `README.openrecon.md` - This file

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
# Make scripts executable
chmod +x build-openrecon-image.sh run-openrecon-server.sh

# Build the integrated Docker image
./build-openrecon-image.sh
```

### 2. Run the Server

```bash
# Start the OpenRecon fetal brain measurement server
./run-openrecon-server.sh
```

The server will be available on port `9002` by default.

### 3. Test the Connection

```bash
# Basic connection test
python test-client.py

# Create and send test data
python test-client.py --create-test-data test_data.h5
python test-client.py --send-test-data test_data.h5
```

## 🔧 Configuration

### i2i Handler Configuration

The OpenRecon i2i handler can be configured by modifying `openrecon.json`:

```json
{
    "version": "2.0.0",
    "description": "Fetal brain measurement configuration for OpenRecon i2i handler",
    "parameters": {
        "processRawData": "true",
        "processImageData": "true",
        "enableMeasurements": "true",
        "outputDirectory": "/tmp/share/fetal_measurements"
    },
    "measurement_settings": {
        "CBD_enabled": "true",
        "BBD_enabled": "true", 
        "TCD_enabled": "true"
    }
}
```

### Environment Variables

- `PYTHONPATH` - Includes paths to fetal measurement modules
- `CUDA_VISIBLE_DEVICES` - Control GPU usage
- `MRD_SERVER_PORT` - Server port (default: 9002)

## 📊 Data Flow & Conversion Process

### 🏗️ **High-Level Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Siemens MRI   │    │  OpenRecon i2i  │    │ Fetal Brain     │    │ AI Processing   │    │ Enhanced DICOM  │
│    Scanner      │───▶│   Framework     │───▶│   Handler       │───▶│   Pipeline      │───▶│    Output       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼                       ▼
  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │ T2W Fetal   │    │ process_image() │    │ ISMRMRD →       │    │ • Brain Seg     │    │ Original Image  │
  │ Brain Scans │    │ Function Call   │    │ NIfTI Convert   │    │ • Measurements  │    │ + DICOM Tags    │
  │ (Real-time) │    │                 │    │                 │    │ • Validation    │    │ + Metadata      │
  └─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🔄 **Detailed Conversion Process**
```
📡 SCANNER INPUT                🔄 i2i HANDLER PROCESSING              📤 DICOM OUTPUT
     │                                    │                                  │
     ▼                                    ▼                                  ▼
┌──────────────┐            ┌──────────────────────────────┐       ┌──────────────────┐
│ ISMRMRD      │   Step 1   │ FetalBrainI2IHandler.process │       │ Enhanced ISMRMRD │
│ Image Data   │──────────▶ │ • input_image               │       │ Image with       │
│ • 3D/4D      │            │ • output_image              │       │ • CBD: 79.8mm    │
│ • T2W Fetal  │            │ • metadata                  │       │ • BBD: 84.9mm    │
│ • Real-time  │            └──────────────────────────────┘       │ • TCD: 45.2mm    │
└──────────────┘                           │                      │ • GA: 36.9 weeks │
                                           ▼                      │ • Brain: 250k mm³│
                             ┌──────────────────────────────┐       └──────────────────┘
                    Step 2   │ ISMRMRD → NIfTI Conversion   │
                   ────────▶ │ • Extract image.data         │
                             │ • Create Nifti1Image         │
                             │ • Save temp file             │
                             └──────────────────────────────┘
                                           │
                                           ▼
                             ┌──────────────────────────────┐
                    Step 3   │ AI Pipeline Execution       │
                   ────────▶ │ • fetal_measure.execute()    │
                             │ • Brain segmentation         │
                             │ • Structure detection        │
                             │ • Measurement calculation    │
                             └──────────────────────────────┘
                                           │
                                           ▼
                             ┌──────────────────────────────┐
                    Step 4   │ Results Extraction           │
                   ────────▶ │ • data.json → measurements   │
                             │ • *.png → base64 plots       │
                             │ • *.pdf → base64 reports     │
                             │ • Clinical validation        │
                             └──────────────────────────────┘
                                           │
                                           ▼
                             ┌──────────────────────────────┐
                    Step 5   │ DICOM Metadata Embedding    │
                   ────────▶ │ • output_image.meta[tags]    │
                             │ • Clinical measurements      │
                             │ • Gestational age preds     │
                             │ • Visual data references    │
                             └──────────────────────────────┘
```

### 📋 **Key Technical Details**

#### **Input Processing:**
- **Format**: ISMRMRD Image objects from OpenRecon
- **Data Extraction**: `image.data` array (3D/4D NumPy)
- **Shape Handling**: Automatic 3D/4D detection and conversion
- **Temporary Storage**: `/tmp/fetal_openrecon_xxxxx/input.nii.gz`

#### **AI Pipeline Integration:**
- **Module**: `fetal_measure.FetalMeasure()`
- **Execution**: Direct Python import and function call
- **Processing**: Complete brain segmentation and measurement
- **Output**: JSON data + PNG plots + PDF reports

#### **DICOM Compliance:**
- **Metadata Embedding**: 15+ DICOM tags in `output_image.meta`
- **Visual Data**: Base64-encoded plots and reports
- **Clinical Tags**: CBD_mm, BBD_mm, TCD_mm, GA_*_weeks, Brain_Volume_mm3
- **Comments**: Human-readable summary in ImageComments tag

#### **Real-time Performance:**
- **Processing Time**: ~30-60 seconds per scan
- **Memory Usage**: Temporary files cleaned automatically
- **Error Handling**: Fallback to original image on failure
- **Debugging**: 30+ debug messages for full traceability

### Input Data Types
- **T2W Fetal Brain Scans**: Primary input for measurements
- **ISMRMRD Images**: Standard OpenRecon image format
- **3D/4D Volumes**: Single or multi-volume datasets

### Output Data
- **Enhanced ISMRMRD Images**: Original images with embedded DICOM metadata
- **Embedded Measurements**: CBD, BBD, TCD values in DICOM tags
- **Gestational Age**: Predictions based on measurements  
- **Clinical Metadata**: Brain volume, validity flags, comments
- **Visual Outputs**: Base64-encoded plots and reports (referenced in metadata)

## 🏥 OpenRecon Integration

### Installation on Scanner

1. **Copy Docker Image**:
   ```bash
   # Save image to file
   docker save openrecon-fetal-brain:latest > fetal-brain-openrecon.tar
   
   # Transfer to scanner and load
   docker load < fetal-brain-openrecon.tar
   ```

2. **Configure OpenRecon**:
   - Update `fire.ini` configuration
   - Set container startup parameters
   - Configure network settings

3. **Start Service**:
   ```bash
   # On the scanner system
   docker run -d --name fetal-brain-server \
     --gpus all \
     -p 9002:9002 \
     -v /scanner/data:/tmp/share \
     openrecon-fetal-brain:latest
   ```

### Scanner Configuration

Add to OpenRecon configuration:

```ini
[FIRE]
chroot_command = /path/to/start-fetal-server.sh
port = 9002
config = fetalbrainmeasure
timeout = 300
```

## 🔍 Measurements Provided

The system automatically computes:

### Primary Measurements
- **CBD (Cerebral Biparietal Diameter)**: Distance across cerebral hemispheres
- **BBD (Bone Biparietal Diameter)**: Skull-to-skull width  
- **TCD (Transcerebellar Diameter)**: Maximum width of cerebellum

### Additional Data
- Brain volume calculations
- Normative percentile comparisons
- Gestational age estimations
- Measurement quality scores

### Output Format

Results are embedded in DICOM metadata:
```
CBD_mm: 45.2
BBD_mm: 52.1
TCD_mm: 18.4
FetalMeasurements: {JSON with full results}
ImageComments: "Fetal Brain Measurements: CBD: 45.2mm, BBD: 52.1mm, TCD: 18.4mm"
```

## 🐛 Troubleshooting

### Common Issues

1. **Container Won't Start**:
   ```bash
   # Check logs
   docker logs openrecon-fetal-server
   
   # Check GPU access
   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

2. **Connection Refused**:
   ```bash
   # Test port
   telnet localhost 9002
   
   # Check firewall
   sudo ufw status
   ```

3. **Measurement Failures**:
   ```bash
   # Check model files
   docker exec -it openrecon-fetal-server ls -la /workspace/Models/
   
   # Check Python paths
   docker exec -it openrecon-fetal-server python -c "import fetal_measure; print('OK')"
   ```

### Debug Mode

Enable detailed logging:
```bash
docker run -e LOG_LEVEL=DEBUG \
  -v $(pwd)/logs:/var/log \
  openrecon-fetal-brain:latest
```

### Data Inspection

Check intermediate results:
```bash
# Mount debug volume
docker run -v $(pwd)/debug:/tmp/share/debug openrecon-fetal-brain:latest

# Inspect saved data
ls -la debug/
```

## 📈 Performance

### System Requirements
- **GPU**: NVIDIA GPU with CUDA 11.0+ support
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for models and temporary data
- **Network**: Gigabit Ethernet for real-time processing

### Processing Times
- **Reconstruction**: ~5-10 seconds
- **Segmentation**: ~15-30 seconds  
- **Measurements**: ~5-10 seconds
- **Total**: ~30-60 seconds per scan

### Optimization Tips
- Use GPU acceleration for neural networks
- Optimize Docker resource limits
- Use SSD storage for temporary files
- Monitor memory usage during processing

## 🔒 Security Notes

- Container runs with restricted permissions
- No external network access required
- Data processed locally within container
- Temporary files automatically cleaned up
- HIPAA compliance considerations included

## 📞 Support

For technical support:
1. Check logs: `docker logs openrecon-fetal-server`
2. Run diagnostics: `python test-client.py`
3. Review configuration: `fetalbrainmeasure.json`
4. Contact: Ichlov Sagol Lab team

## 📝 License

This integration maintains the same license as the original fetal brain measurement pipeline. For research and development use only - not intended for clinical decision-making without regulatory approval.
