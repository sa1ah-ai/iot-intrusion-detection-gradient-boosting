# IoT Intrusion Detection using Gradient Boosting

A comprehensive machine learning project for detecting IoT network intrusions using various gradient boosting algorithms. This project implements binary classification, 8-class multiclass classification, and full multiclass classification on the CICIoT2023 dataset.

## 📊 Project Overview

This project addresses the critical challenge of IoT security by developing robust intrusion detection systems using state-of-the-art gradient boosting algorithms. The work demonstrates the effectiveness of different machine learning approaches for identifying various types of cyber attacks in IoT networks.

## 🎯 Key Features

- **Multiple Classification Tasks**: Binary, 8-class, and full multiclass classification
- **Advanced Algorithms**: XGBoost, LightGBM, CatBoost, NGBoost, and traditional ML methods
- **Comprehensive Evaluation**: Detailed performance metrics, confusion matrices, and inference latency analysis
- **Feature Selection**: Automated feature selection using LightGBM gain-based importance
- **Data Preprocessing**: Stratified undersampling and standardization for balanced datasets

## 📁 Project Structure

```
iot-intrusion-detection-gradient-boosting/
├── data/
│   └── Dataset_link.txt              # Link to CICIoT2023 dataset
├── notebooks/
│   ├── binary_classification.ipynb   # Binary classification (Benign vs Attack)
│   ├── multiclass_8class_classification.ipynb  # 8-class attack categorization
│   └── multiclass_full_classification.ipynb    # Full multiclass classification
├── visualizations & results/
│   ├── binary_distribution.jpeg      # Binary class distribution
│   ├── original_label_distribution.jpeg  # Original dataset distribution
│   ├── reduced_distribution.jpeg     # After undersampling
│   ├── mapped_8class_distribution.jpeg   # 8-class mapping visualization
│   ├── feature_importance_all_gain.jpeg  # Feature importance analysis
│   ├── classification_report_*.jpeg  # Performance reports for each model
│   └── confusion_matrix_*.jpeg       # Confusion matrices for each model
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 16GB+ RAM (for processing large datasets)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sa1ah-ai/iot-intrusion-detection-gradient-boosting.git
cd iot-intrusion-detection-gradient-boosting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CICIoT2023 dataset from the [official source](https://www.unb.ca/cic/datasets/iotdataset-2023.html) and extract it to your local directory.

4. Update the `DATA_PATH` variable in the notebooks to point to your dataset location.

### Usage

1. **Binary Classification**: Run `notebooks/binary_classification.ipynb`
   - Detects benign traffic vs. all attack types
   - Implements 6 different algorithms
   - Includes inference latency benchmarking

2. **8-Class Classification**: Run `notebooks/multiclass_8class_classification.ipynb`
   - Categorizes attacks into 8 main types: DDoS, DoS, Recon, WebBased, BruteForce, Spoofing, Mirai, and Benign
   - Uses XGBoost, CatBoost, and LightGBM

3. **Full Multiclass Classification**: Run `notebooks/multiclass_full_classification.ipynb`
   - Classifies all 33 attack types plus benign traffic
   - Most comprehensive analysis with detailed performance metrics

## 📈 Results Summary

### Binary Classification Performance
| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| XGBoost | 99.61% | 0.99 | 1.00 | 1.00 |
| LightGBM | 99.51% | 0.99 | 1.00 | 0.99 |
| CatBoost | 99.57% | 0.99 | 1.00 | 0.99 |
| NGBoost | 99.58% | 0.99 | 1.00 | 0.99 |
| ExtraTrees | 98.92% | 0.98 | 0.99 | 0.99 |
| Logistic Regression | 98.22% | 0.97 | 0.98 | 0.98 |

### 8-Class Classification Performance
- **XGBoost**: 99.59% accuracy
- **CatBoost**: 98.75% accuracy  
- **LightGBM**: High performance with detailed per-class metrics

### Inference Latency (per sample)
- **XGBoost**: ~0.0008ms
- **LightGBM**: ~0.0007ms
- **CatBoost**: ~0.0054ms
- **Logistic Regression**: ~0.0000ms (fastest)

## 🔬 Methodology

### Data Preprocessing
1. **Stratified Undersampling**: Reduces dataset size while maintaining class distribution
2. **Feature Selection**: Uses LightGBM gain-based importance to select most relevant features
3. **Standardization**: Applies StandardScaler for consistent feature scaling
4. **Class Balancing**: Implements sample weighting for imbalanced datasets

### Model Training
- **Cross-validation**: 80/20 train-test split with stratification
- **Hyperparameter Tuning**: Optimized parameters for each algorithm
- **Early Stopping**: Prevents overfitting with validation monitoring
- **GPU Acceleration**: Utilizes CUDA for faster training when available

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices for detailed analysis
- Per-class performance metrics
- Inference latency benchmarking

## 🎨 Visualizations

The project generates comprehensive visualizations including:
- Class distribution plots (original, reduced, mapped)
- Feature importance analysis
- Classification reports as images
- Confusion matrices for all models
- Performance comparison charts

## 🛠️ Technical Details

### Dataset
- **Source**: CICIoT2023 dataset from University of New Brunswick
- **Size**: ~46.7 million samples (reduced to 4-5 million for processing)
- **Features**: 46 network traffic features
- **Classes**: 33 attack types + benign traffic

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only processing
- **Recommended**: 16GB+ RAM, CUDA-compatible GPU
- **Storage**: 4GB+ for dataset and results

### Software Dependencies
- Python 3.8+
- Jupyter Notebook
- CUDA Toolkit (for GPU acceleration)
- See `requirements.txt` for complete list

## 📚 Research Context

This project implements and evaluates various gradient boosting algorithms for IoT intrusion detection, contributing to the field of cybersecurity and machine learning. The comprehensive evaluation provides insights into:

- Algorithm performance comparison
- Feature importance analysis
- Inference speed optimization
- Class imbalance handling strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

## 🙏 Acknowledgments

- University of New Brunswick for providing the CICIoT2023 dataset
- The open-source community for the excellent machine learning libraries
- Contributors and researchers in the field of IoT security

---

**Note**: This project is for research and educational purposes. Always ensure proper security measures when implementing intrusion detection systems in production environments.
