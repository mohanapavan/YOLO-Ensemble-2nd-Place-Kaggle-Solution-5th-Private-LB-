# 🏆 YOLO Ensemble: 2nd Place (Public LB) & 5th Place (Private LB) - Kaggle Competition

![Leaderboard Proof](https://github.com/mohanapavan/YOLO-Ensemble-2nd-Place-Kaggle-Solution-5th-Private-LB-/blob/main/results/leaderBoard.png?raw=true)

## 🔍 Detection Results

![results/ensemble/detection_example.jpg](https://github.com/mohanapavan/YOLO-Ensemble-2nd-Place-Kaggle-Solution-5th-Private-LB-/blob/main/results/result2.png?raw=true)

**What You're Seeing**:
- Model A detected 2 objects (red boxes)
- Model B detected 3 objects (blue boxes)
- Our ensemble combines both:
  - Keeps all unique detections
  - keeps unique ones from both models
  - Result: All objects captured

**Why It Works**:
- Model A catches clear objects
- Model B finds harder-to-see ones
- Together: Best of both worlds

## 🚀 Overview
A powerful ensemble of two YOLO models (Model A + Model B) that achieved:
- **2nd place** in public leaderboard
- **5th place** in private leaderboard  
for [Multi-Instance Object Detection Challenge](https://www.kaggle.com/competitions/multi-instance-object-detection-challenge).

## 🔧 Key Features
- **Hybrid Architecture**: Combines strengths of Model A (strength) and Model B (strength)
- **Enhanced Accuracy**: X% improvement over single-model baseline
- **Competition-Proven**: Top-tier results in [Kaggle Competition Name]

## 🔧 Key Features

### 🔍 IoU-Based Consensus
- Uses **mAP@50** to validate overlapping predictions  
- Two boxes count as "same detection" if IoU ≥ 0.5  
- Eliminates duplicate predictions while preserving unique detections  

### 🧩 Smart Prediction Fusion
1. **Common Objects** (IoU ≥ 0.5):  
   - Keeps the higher-confidence box  
2. **Unique Objects**:  
   - Preserves all Model A's high-precision detections  
   - Retains Model B's partial-object catches
  
     
## 🧠 Model A: High-Precision Detector (YOLOv11x)
Model A is built on YOLOv11x, customized to prioritize precision over recall, reduce false positives, and generalize well with limited data. It was trained on a single-class detection task using carefully selected augmentations and stable optimization strategies.

### 🛠️ Configuration Highlights

#### Architecture
model: yolo11x.pt        # yolo11x with pretrained weights
imgsz: 640               # Optimal resolution for precision

#### Training
epochs: 50               # Full training cycles
batch: 16                # Balanced memory/performance
optimizer: SGD           # With cosine learning rate
lr0: 0.0005              # Conservative starting rate

#### Augmentation
hsv_s: 0.7               # Controlled color variation
fliplr: 0.25             # Moderate horizontal flips
cutmix: 0.3              # Object blending augmentation

### 🚀 Other Critical Decisions

- single_cls=True: The task involved only one object class (e.g., plate, logo, etc.)

- overlap_mask=True & mask_ratio=4: Boosts performance for partially overlapping objects

- dropout=0.4: High dropout to fight overfitting (unusual for YOLO)

- auto_augment=randaugment: Dynamically applied augmentations for diverse training data

## 🧠 Model B: Wide-View Detector (YOLOv8s)
Model B is based on the compact and fast YOLOv8s architecture, trained at a higher image resolution (1280×1280) to improve detection of small and distant objects. It is optimized for generalization and coverage, using a combination of spatial augmentations and regularization techniques.

### 🛠️ Configuration Highlights

#### Architecture
model: yolov8s.pt        # Lightweight YOLOv8s with pretrained weights
imgsz: 1280              # High-resolution input for better object scale handling

#### Training
epochs: 50               # Same training duration as Model A
batch: 16                # Consistent batch size for fair comparison
optimizer: SGD           # Paired with cosine LR schedule
lr0: 0.001               # Slightly higher initial LR than Model A

#### Augmentation
hsv_s: 0.7               # Vary saturation for lighting diversity
fliplr: 0.2              # Mild horizontal flips
flipud: 0.4              # Aggressive vertical flipping
cutmix: 0.3              # Region-level data mixing
scale: 0.5               # Strong object scaling for robustness
erasing: 0.3             # Random patch erasing for regularization


### 🚀 Other Critical Decisions

- single_cls=True: Model B was also trained for a single-object class scenario

- dropout=0.3: Slightly lower dropout than Model A, allowing better retention of feature activations

- overlap_mask=True & mask_ratio=4: Helps detect occluded or overlapping instances

- auto_augment=randaugment: Auto-selected transformation policy per image

- imgsz=1280: Higher resolution helps detect smaller and far objects better

- weight_decay=0.0005: Heavier regularization than Model A for generalization

- erasing=0.3: Helps prevent over-reliance on specific spatial features
