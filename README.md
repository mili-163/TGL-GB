## 1. TGL-GB

This project builds an efficient and automated communication link type classification system based on GBDT and T-GNN, capable of intelligent identification and classification of various link types.

---

## 2. Code Structure

```
TGL-GB/
│
├── models/                  # Model-related code
│   ├── gbdt_model.py        # GBDT model and leaf embedding
│   ├── tgnn.py              # T-GNN model, spatiotemporal fusion, loss functions
│   └── feature_initializer.py # Feature initialization and concatenation
│
├── data/                   # Data processing and feature engineering
│   ├── data_loader.py       # Data loading, label generation, feature concatenation
│   ├── feature_extractor.py # Static feature extraction
│   └── dynamic_feature_extractor.py # Dynamic feature extraction (RTT, etc.)
│
├── train.py               # Training script (renamed from main.py)
├── test.py                # Testing script for model evaluation
├── requirements.txt       # Dependencies list
└── README.md             # Project documentation
```

---

## 3. Core Principles

1. **Feature Engineering**
   - Static Features: IP, ASN, ISP, etc., encoded using OneHot encoding
   - Dynamic Features: RTT variations with hop count, extracted using polynomial fitting and statistical measures
   - Feature Concatenation: Combining static, dynamic, and edge features

2. **GBDT Model**
   - Initial classification and feature enhancement
   - Leaf node indices encoded and reduced via OneHot and MLP

3. **T-GNN Model**
   - Combines temporal and spatial information
   - Spatial attention mechanism and spatiotemporal fusion
   - Outputs final node/link embeddings and classification results

4. **Link Feature and Temporal Fusion**
   - Constructs link pair features
   - Supports multi-time-step fusion (e.g., sliding window weighted)

5. **Classification and Evaluation**
   - Uses TFN (Temporal Fusion Network) for final classification
   - Supports loss calculation, accuracy, and class distribution evaluation

---

## 4. Dependencies

- Python >= 3.8
- torch >= 2.0.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn>=0.11.0
- tqdm>=4.62.0
- lightgbm>=3.3.0
- networkx>=2.6.0
- scipy>=1.7.0
- joblib>=1.1.0
- tensorboard>=2.8.0 

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 5. Usage

1. **Data Preparation**
   The system processes two types of features:

   a. **Static Features**:
   - IP addresses (e.g., 192.168.x.x)
   - ASN numbers (e.g., AS1000-AS1049)
   - ISP information (5 different ISPs)
   - Each link type has 50 samples

   b. **Dynamic Features**:
   - RTT (Round Trip Time) measurements
   - 10 time steps per sample
   - 10 hop counts per time step
   - Different RTT patterns for each link type:
     * Fiber: Low latency (5 * hop_count + noise)
     * Satellite: High latency (100 + 2 * hop_count + noise)
     * Mobile: Medium latency (20 + 8 * hop_count + noise)
     * Microwave: Low-medium latency (15 + 4 * hop_count + noise)

2. **Training**
   ```bash
   python train.py
   ```
   - Trains the GBDT model
   - Trains the T-GNN model
   - Performs feature fusion and classification

3. **Testing**
   ```bash
   python test.py
   ```
   - Evaluates model performance
   - Prints feature shapes, model parameters, loss, accuracy, and class distribution

---

## 6. Model Architecture

1. **GBDT + Leaf Embedding**
   - Multiple decision trees for initial classification
   - Leaf node embedding for feature enhancement

2. **T-GNN Components**
   - Temporal Recurrent Encoder
   - Spatial Attention Mechanism
   - Spatiotemporal Fusion Module
   - Temporal Fusion Network (TFN)

3. **Feature Processing Pipeline**
   - Static feature extraction
   - Dynamic feature extraction
   - Feature initialization and concatenation
   - Temporal fusion

---

## 7. Common Issues

- **Dependencies**: Ensure all required packages are installed
- **Custom Link Types**: Can be extended in `data_loader.py`
- **Model Parameters**: Adjustable in respective model files

---

## 8. Contact
For questions or suggestions, please contact the project maintainer. 
