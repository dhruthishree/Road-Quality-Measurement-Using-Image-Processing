# AutoRoad Vision AI: Continuous Road Quality Measurement using Image Processing

An enterprise-grade, edge-deployable computer vision and spatial-temporal image processing pipeline designed to continuously measure road surface distress, quantify defect dimensions in physical metric units ($m^2$), estimate dynamic road inclination/grade (%), and aggregate measurements over continuous distance chainage into a standardized Road Quality Index (RQI / PCI 0-100) with **Good / Fair / Bad** classification.

---

## 1. Problem Statement & Project Definition

Municipalities, highways departments, and transport authorities require scalable, objective, and continuous measurement of pavement condition. Traditional methods rely on either:
1. **Manual Visual Surveys**: Subjective, hazardous, slow, and non-repeatable.
2. **Dedicated Survey Fleets (LCMS/LiDAR)**: Prohibitively expensive ($150,000 to $500,000+ per vehicle), requiring specialized operators.
3. **Smartphone Accelerometers**: Prone to suspension noise, speed bias, tire pressure differences, and lacking visual audit trails.

### Project Goals
- **Surface Defect Quantification**: Detect potholes, alligator cracks, longitudinal/transverse cracks, and asphalt patches. Quantify physical surface area ($m^2$), length, width, and severity (1–5).
- **Road Inclination & Slope Estimation**: Track vanishing point horizon dynamics and perspective geometry to calculate vehicle pitch and road grade ($\%$).
- **Continuous Distance Aggregation**: Aggregate telemetry and verified defects over continuous road chainage (50m discrete segments) to avoid duplicate defect counts and compute true defect density ($\%$).
- **Civil Engineering Standard Alignment**: Output a calibrated Road Quality Index (RQI) aligned with ASTM D6433 (Pavement Condition Index) and IRC:SP:16 guidelines, classifying segments into **Good**, **Fair**, and **Bad**.
- **Interactive UI & Audit Reporting**: Provide an interactive web dashboard with live video streaming, top-down Bird's Eye View (IPM), telemetry HUD, continuous quality profile charts, and downloadable HTML/CSV audit reports.

---

## 2. Market Research: 5 Critical Industry Gaps Solved

| Existing System Gap | Flaw in Conventional Solutions | AutoRoad Vision AI Solution |
| :--- | :--- | :--- |
| **Gap 1: Cost & Hardware Prohibitions** | Multi-million dollar 3D laser profilers (LCMS / ARAN) cannot be widely deployed for daily municipal monitoring. | **Monocular Edge Computer Vision**: Replaces laser profilers with high-resolution camera feeds and calibrated Inverse Perspective Mapping (IPM) to compute true metric dimensions. |
| **Gap 2: Shadow & Marking False Alarms** | Tree canopy shadows, bridge shadows, and painted lane lines constantly trigger false pothole/crack alarms. | **Illumination-Invariant Preprocessing**: Multi-Scale Retinex, CIELAB shadow attenuation, HSV paint line masks, and bilateral edge-preserving filtering suppress false positives. |
| **Gap 3: Perspective Distortion & Slope Neglect** | 2D bounding boxes ignore camera perspective (defects 5m away appear 10x larger than at 20m) and road slope/grade is ignored. | **Dual Geometric Engine**: Vanishing point tracking measures dynamic pitch/grade ($\%$) while IPM homography maps pixels into real-world meters ahead of the bumper ($m^2$). |
| **Gap 4: Frame-Isolated Predictions (Duplicate Counting)** | Detecting defects frame-by-frame causes a single pothole to be counted 15–30 times as the car drives past it. | **Spatial-Temporal Multi-Object Tracker**: Ground-plane centroid tracking with persistent IDs (`T-001`) and odometry integration registers each physical defect exactly once. |
| **Gap 5: Black-Box Confidence vs. Engineering Metrics** | Existing AI models output arbitrary confidence scores (`pothole 0.88`) that offer zero utility to road repair engineers. | **ASTM D6433 Aligned Engine**: Computes defect density ($\%$), deduct values, 0–100 RQI score, and discrete **Good / Fair / Bad** ratings with actionable maintenance plans. |

---

## 3. Mathematical Formulations & Methodology

### 3.1 Camera Perspective & Inverse Perspective Mapping (IPM)
Given camera focal length $f_y$, mounting height $h_{\text{cam}}$, and nominal pitch angle $\theta_0$:
$$u = f_x \frac{X_{\text{cam}}}{Z_{\text{cam}}} + c_x, \quad v = f_y \frac{Y_{\text{cam}}}{Z_{\text{cam}}} + c_y$$

The IPM Homography matrix $H_{\text{ipm}}$ maps camera perspective coordinates $[u, v, 1]^T$ onto the top-down ground plane $[u_{\text{ipm}}, v_{\text{ipm}}, 1]^T$:
$$\mathbf{p}_{\text{ipm}} = H_{\text{ipm}} \cdot \mathbf{p}_{\text{img}}$$

For any defect contour $C$, the physical surface area $A_{\text{metric}}$ in $m^2$ is computed as:
$$A_{\text{metric}} = \text{Area}_{\text{px}}(H_{\text{ipm}}(C)) \times (\delta_x \times \delta_y)$$
where $\delta_x$ and $\delta_y$ are ground meters per IPM pixel.

### 3.2 Dynamic Horizon & Road Inclination
Vehicle pitch shifts the horizon line $y_{\text{horizon}}$ relative to optical center $c_y$:
$$\Delta y = y_{\text{nominal}} - y_{\text{horizon}}$$
$$\theta_{\text{pitch}} = \arctan\left(\frac{\Delta y}{f_y}\right)$$
$$\text{Road Grade } (\%) = \tan(\theta_{\text{pitch}}) \times 100\%$$

### 3.3 Continuous Segment Chaining & Defect Density
Continuous road distance is partitioned into discrete $L_{\text{seg}} = 50\text{ m}$ intervals:
$$\text{Segment Area } A_{\text{seg}} = L_{\text{seg}} \times W_{\text{road}} = 50\text{ m} \times 7.0\text{ m} = 350\text{ m}^2$$
$$\text{Defect Density } D (\%) = \frac{\sum_{i \in \text{seg}} A_{\text{defect}, i}}{A_{\text{seg}}} \times 100\%$$

### 3.4 Road Quality Index (RQI / PCI) Deduct Model
$$\text{Deduct}_{\text{pot}} = \min(65.0, A_{\text{pot}} \times 22.0 + N_{\text{pot}} \times 8.0)$$
$$\text{Deduct}_{\text{crack}} = \min(35.0, A_{\text{crack}} \times 14.0 + N_{\text{crack}} \times 2.5)$$
$$\text{Deduct}_{\text{patch}} = \min(18.0, A_{\text{patch}} \times 6.0 + N_{\text{patch}} \times 2.0)$$
$$\text{Deduct}_{\text{slope}} = \max(0.0, (\sigma_{\text{grade}} - 1.2) \times 2.5) \times 3.5$$
$$\text{RQI} = \max(0.0, \min(100.0, 100.0 - \sum \text{Deduct}))$$

- **Good ($80 \le \text{RQI} \le 100$)**: Routine maintenance and sweeping.
- **Fair ($60 \le \text{RQI} < 80$)**: Crack sealing, localized patching, micro-surfacing.
- **Bad ($\text{RQI} < 60$)**: Structural milling and dense asphalt overlay.

---

## 4. The Original Benchmark Dataset

The repository includes an original, deterministic photorealistic dataset synthesized with procedural asphalt micro-texture, 3D perspective projection, dynamic lane markings, moving vehicle odometry, illumination variations, and ground-truth telemetry:

1. `route_01_highway_smooth_good`: Pristine high-speed expressway; zero potholes; smooth grade; RQI $> 90$ (**Good**).
2. `route_02_urban_arterial_cracked_fair`: Arterial road with longitudinal/alligator cracks and patches; RQI $\sim 70-75$ (**Fair**).
3. `route_03_rural_distressed_bad`: Severely deteriorated secondary road with multiple deep potholes and dense alligator cracks; RQI $< 55$ (**Bad**).
4. `route_04_hill_incline_gradient`: Mountainous flyover with steep uphill ($+8\%$ grade) transitioning to downhill ($-7\%$ grade).
5. `route_05_canopy_shadows_complex`: Tree-lined boulevard with intense dappled shadows testing illumination-invariant shadow rejection.
6. `route_06_wet_patchy_mixed`: Rain-dampened road with specular sheen and bituminous repair patches.

---

## 5. Deep Learning Model: RoadDefectCNN

### 5.1 Architecture
The **RoadDefectCNN** is a deep residual convolutional neural network with dual output heads:

| Component | Details |
|:---|:---|
| **Input** | 224×224 RGB road surface images |
| **Backbone** | 5 conv blocks with BatchNorm, ReLU, residual skip connections |
| **Block 3 & 5** | Residual bottleneck with 1×1 projection shortcuts |
| **Pooling** | Global Adaptive Average Pooling |
| **Classification Head** | FC(256→128→2) with 40% Dropout — Binary defect/no-defect |
| **Localization Head** | FC(256→128→4) with Sigmoid — YOLO-format bbox regression (cx, cy, w, h) |
| **Optimizer** | AdamW (lr=0.001, weight_decay=1e-4) with Cosine Annealing LR |
| **Loss** | CrossEntropy (classification) + 2.5× SmoothL1 (bbox regression) |

### 5.2 Training on 100% Real Road Dataset

| Training Parameter | Value |
|:---|:---|
| **Dataset** | 100% Real Road Surface Images — Brazilian NDTI / Global Road Damage Dataset (RDD) |
| **Total Real Images** | **1,983** annotated real road photographs |
| **Training Split** | 1,562 images (79%) |
| **Validation Split** | 421 images (21%) |
| **Synthetic Data Used** | **NONE** — 0% synthetic, 100% real road photographs |
| **Training Epochs** | 15 epochs (extended) / 4 epochs (quick) |
| **Batch Size** | 64 |
| **Total Training Duration** | **~3.5 hours** on CPU |
| **Data Augmentation** | Random color jitter (±20 brightness), ImageNet normalization |
| **Final Checkpoint** | `models/road_defect_model_real_weights.pt` |

### 5.3 Evaluation Metrics
Run `python evaluate_model.py` after training to compute:
- **Accuracy**, **Precision**, **Recall (Sensitivity)**, **F1 Score**
- Confusion matrix (TP/FP/TN/FN)
- Per-image inference throughput (FPS)
- Visual prediction grid saved to `output/evaluation_grid.jpg`

---

## 6. System Architecture

```
Road Quality Measurement using image processing/
├── dataset/
│   ├── generator/
│   │   └── dataset_generator.py      # Procedural multi-scenario synthesis
│   ├── real_road_dataset/            # 1,983 real annotated road images (YOLO format)
│   │   └── dataset/
│   │       ├── train/                # 1,562 training images + labels
│   │       └── test/                 # 421 validation images + labels
│   ├── routes/                       # 6 original video routes (.mp4 & telemetry.csv)
│   └── test_images/                  # 12 high-resolution benchmark test frames
├── src/
│   ├── config.py                     # Camera calibration & threshold configuration
│   ├── model_network.py              # RoadDefectCNN deep residual network architecture
│   ├── preprocessing.py              # Retinex, CLAHE, shadow & marking suppression
│   ├── geometry.py                   # IPM homography, vanishing point & inclination
│   ├── defect_detector.py            # Defect extraction with metric scaling (m²)
│   ├── tracker.py                    # Spatial-temporal tracker (eliminates duplicates)
│   ├── aggregator.py                 # Segment-level chaining (50m) & RQI scoring
│   ├── report_generator.py           # HTML, CSV, and JSON audit report generator
│   └── pipeline.py                   # Unified end-to-end processing pipeline
├── models/
│   ├── road_defect_model_real_weights.pt  # Trained model checkpoint
│   └── training_metadata.json             # Training duration, accuracy, hyperparams
├── web/
│   ├── app.py                        # Full-featured Flask application & streaming API
│   ├── templates/index.html          # Interactive dark-mode glassmorphic dashboard
│   └── static/
│       ├── css/style.css             # Responsive styling & micro-animations
│       └── js/dashboard.js           # Live telemetry, Chart.js graphs & table updates
├── tests/
│   └── test_pipeline.py              # Automated unit and integration test suite
├── train_model.py                    # Training script (100% real data, ~3.5 hrs)
├── evaluate_model.py                 # Evaluation & inference script
├── main.py                           # CLI entrypoint for batch evaluation & web launch
└── requirements.txt
```

---

## 7. Installation & Execution

### 7.1 Prerequisites
Python 3.10+ with standard scientific stack:
```bash
pip install -r requirements.txt
```

### 7.2 Train the Deep Learning Model
```bash
python train_model.py
```
Trains the RoadDefectCNN on 1,983 real road images for ~3.5 hours. Saves checkpoint to `models/`.

### 7.3 Evaluate the Trained Model
```bash
# Full test set evaluation with metrics
python evaluate_model.py

# Single image inference
python evaluate_model.py --image "path/to/road_photo.jpg"

# Batch inference on a folder
python evaluate_model.py --folder "path/to/images/"
```

### 7.4 Launch Interactive Web Dashboard
```bash
python main.py --web
```
Open your browser at **`http://127.0.0.1:5000`** to:
- Select any benchmark route or upload your own dashcam video.
- View live video processing with annotated perspective HUD or top-down Bird's Eye View (IPM).
- Observe real-time continuous RQI and Road Grade graphs.
- Inspect the 50m Segment Matrix and Tracked Defects Catalog.
- Download the executive HTML report and CSV data exports.

### 7.5 Run Automated Benchmark Validation Suite
```bash
python main.py --benchmark
```
Executes batch analysis across all 6 original routes and prints the summary benchmark table.

### 7.6 Process a Custom Video via CLI
```bash
python main.py --video "path/to/dashcam.mp4" --output "output/my_survey"
```

### 7.7 Run Automated Test Suite
```bash
python -m unittest tests/test_pipeline.py
```
Validates all 7 subsystems: preprocessing, geometry, inclination, defect detection, tracking, continuous aggregation, and report generation.
