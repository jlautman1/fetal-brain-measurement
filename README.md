# Fetal Brain Measurement Pipeline

## Overview

This repository provides a **fully automated pipeline** for computing fetal brain measurements (CBD, BBD, and TCD) directly from raw fetal MRI scans in `.nii` or `.nii.gz` format. The pipeline segments the fetal brain, identifies key reference slices, performs measurements, and outputs a **quantitative PDF report** similar to [this example](report.pdf).

This tool was developed at the **Sagol Brain Institute** as part of an effort to reduce manual measurement variability and enable real-time, consistent fetal brain assessments during MRI.

---

## 📥 Input

* A **NIfTI** (.nii.gz) file containing a coronal T2-weighted fetal MRI scan, where the fetus is visible in utero.
* Place your input file(s) here:

  ```
  /workspace/fetal-brain-measurement/Inputs/Fixed/
  ```

---

## 📤 Output

* PDF report: `report.pdf`
* Cropped images: `cropped.nii.gz`, `seg.cropped.nii.gz`
* Measurement plots: `cbd.png`, `bbd.png`, `tcd.png`
* JSON metadata: `data.json`
* Output folder:

  ```
  /workspace/fetal-brain-measurement/output/
  ```

---

## 🔧 How to Run the Pipeline

### 1. Clone this repository

```bash
git clone https://github.com/jlautman1/fetal-brain-measurement.git
```

### 2. Pull the prebuilt Docker image

```bash
docker pull jlautman1/fetal-pipeline-gpu-rebuilt:latest
```

### 3. Run the Docker container

Adapt this path to your local system:

```bash
docker run --gpus all -it \
  -v /your/local/path/fetal-brain-measurement:/workspace/fetal-brain-measurement \
  jlautman1/fetal-pipeline-gpu-rebuilt:latest bash
```

### 4. Inside the container, run the pipeline

```bash
python3 /workspace/fetal-brain-measurement/Code/FetalMeasurements-master/execute.py \
  -i /workspace/fetal-brain-measurement/Inputs/Fixed \
  -o /workspace/fetal-brain-measurement/output
```

---

## 📈 What the Pipeline Does

The pipeline is composed of five main stages:

### 1. **Fetal Brain Localization**

* Uses a 3D anisotropic U-Net to identify the region containing the fetal brain.
* Downscales the full volume and outputs a tight ROI bounding box.

### 2. **Reference Slice Selection**

* Modified ResNet50 model classifies slices for CBD/BBD and TCD.
* Identifies the best slices for consistent measurement across scans.

### 3. **Structure Segmentation**

* 2D U-Net with ResNet34 encoder segments each selected slice into:

  * Left Hemisphere
  * Right Hemisphere
  * Cerebellum

### 4. **Mid-Sagittal Line & Orientation Computation**

* Classical geometric methods detect bilateral symmetry and brain orientation.

### 5. **CBD, BBD, and TCD Measurement Computation**

* Cerebral Biparietal Diameter (CBD)
* Bone Biparietal Diameter (BBD)
* Transcerebellar Diameter (TCD)
* Measurements are drawn perpendicularly to the MSL using convex hulls, symmetry, and edge detection.

---

## 📄 Example Report

See [report.pdf](report.pdf) for a sample output.

Includes:

* Annotated slice views with CBD, BBD, and TCD lines
* Normative percentile graphs based on gestational age
* Brain volume and voxel statistics

---

## 🧠 Clinical Context

### Why Fetal MRI?

Fetal MRI is often used in cases where ultrasound is inconclusive. It is commonly performed:

* Around gestational weeks 22–36
* To investigate suspected anomalies
* When higher-resolution brain assessment is needed

### Why Automate?

* Manual MRI measurements are time-consuming and subjective
* Our system standardizes the process and enables real-time evaluation

### What Are the Measurements?

* **CBD (Cerebral Biparietal Diameter):** Distance across the cerebral hemispheres
* **BBD (Bone Biparietal Diameter):** Skull-to-skull width
* **TCD (Transcerebellar Diameter):** Maximum width of the cerebellum

Each measurement is validated against a normative dataset to highlight deviations from typical fetal development.

---

## 📚 Deep Dive (See Presentation)

For more technical details, refer to [`WorkshopPresentationFetalMeasurement.pptx`](WorkshopPresentationFetalMeasurement.pptx), which explains:

* The neural networks used (3D U-Net, ResNet50, U-Net+ResNet34)
* Accuracy and validation results
* Integration with Siemens OpenRecon MRI workflows
* Alternative approaches with landmark-based networks

---

## Questions?

For support or collaboration, contact [Jonathan Lautman](https://github.com/jlautman1) or the [Sagol Brain Institute](https://sagol.tau.ac.il).

---

## License

This code is provided for research and development use only. It is not intended for clinical decision-making without regulatory approval.
