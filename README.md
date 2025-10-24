<div align="center">

# 🎬 IMDb Sentiment Analysis with DistilBERT

### 🚀 Fine-Tuning Large Language Models for Movie Review Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

*Achieving 93%+ accuracy on binary sentiment classification through strategic hyperparameter optimization*

[🎯 Overview](#-overview) • [📊 Results](#-results) • [🚀 Quick Start](#-quick-start) • [📁 Project Structure](#-project-structure) • [🎓 Methodology](#-methodology)

</div>

---

## 🎯 Overview

This project demonstrates **state-of-the-art fine-tuning techniques** for sentiment analysis using DistilBERT on the IMDb movie review dataset. Through systematic hyperparameter optimization and rigorous evaluation, we achieved a **43+ percentage point improvement** over the baseline model.

### ✨ Key Highlights

- 🎯 **93.04% Accuracy** - Best performing configuration
- ⚡ **66M Parameters** - Efficient DistilBERT architecture
- 🔬 **3 Configurations** - Comprehensive hyperparameter search
- 📈 **+43pp Improvement** - Significant baseline enhancement
- 🎨 **Production Ready** - Complete inference pipeline included

---

## 📊 Results

### 🏆 Performance Comparison

| Configuration | Accuracy | F1 Score | Precision | Recall | Training Time |
|--------------|----------|----------|-----------|---------|---------------|
| 🔵 Baseline (No Fine-tuning) | 50.00% | 0.5000 | 0.5000 | 0.5000 | - |
| 🟢 **Config 1: Standard** | **93.04%** | **0.9304** | **0.9305** | **0.9304** | 15.2 min |
| 🟡 Config 2: High LR | 92.88% | 0.9288 | 0.9289 | 0.9288 | 15.0 min |
| 🟣 Config 3: Small Batch | 92.96% | 0.9296 | 0.9297 | 0.9296 | 20.3 min |

### 📈 Key Insights

- ✅ **Standard fine-tuning** (Config 1) achieved the best results
- ✅ **Higher learning rates** showed minimal performance trade-off
- ✅ **Smaller batches** with more regularization maintained competitive accuracy
- ✅ **All configurations** significantly outperformed the baseline

---

## 🚀 Quick Start

### 📋 Prerequisites
```bash
# Python 3.8+
pip install transformers datasets torch accelerate evaluate scikit-learn pandas matplotlib seaborn
```

### ⚡ Run in Google Colab

1. **Open Colab** and enable GPU (`Runtime` → `Change runtime type` → `T4 GPU`)
2. **Clone & Setup**:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q transformers datasets accelerate evaluate scikit-learn
```

3. **Load & Run** the complete notebook (see [notebooks/](notebooks/) folder)

### 🎯 Using the Fine-Tuned Model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained('./models/config_1/final_model')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Make prediction
text = "This movie was absolutely fantastic! I loved every minute of it."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
print(f"Sentiment: {'Positive 😊' if prediction == 1 else 'Negative 😞'}")
```

**Output:**
```
Sentiment: Positive 😊
```

---

## 📁 Project Structure
```
imdb_sentiment_finetuning/
│
├── 📓 notebooks/
│   └── fine_tuning_complete.ipynb    # Complete training notebook
│
├── 🤖 models/
│   ├── config_1/final_model/         # Best model (93.04% accuracy)
│   ├── config_2/final_model/         # High LR variant
│   └── config_3/final_model/         # Small batch variant
│
├── 📊 results/
│   ├── baseline_results.json         # Pre-fine-tuning metrics
│   ├── config_1_results.json         # Config 1 performance
│   ├── config_2_results.json         # Config 2 performance
│   ├── config_3_results.json         # Config 3 performance
│   └── comparison.csv                # Side-by-side comparison
│
├── 📈 visualizations/
│   ├── dataset_statistics.png        # Data distribution plots
│   ├── performance_comparison.png    # Metrics bar charts
│   ├── confusion_matrix.png          # Classification results
│   └── error_patterns.png            # Error analysis
│
├── 🔍 error_analysis/
│   └── detailed_errors.csv           # Misclassified examples
│
├── 📝 logs/
│   ├── config_1/                     # TensorBoard logs
│   ├── config_2/
│   └── config_3/
│
├── 🐍 inference_script.py            # Production inference code
├── 📄 FINAL_SUMMARY_REPORT.txt       # Comprehensive report
└── 📖 README.md                       # This file
```

---

## 🎓 Methodology

### 1️⃣ Dataset Preparation

- **Source**: IMDb Movie Reviews Dataset
- **Size**: 50,000 reviews (25k train, 5k validation, 25k test)
- **Split**: 80% training, 20% validation from original training set
- **Preprocessing**: Tokenization with max length 512, padding and truncation
- **Balance**: 50/50 positive/negative sentiment distribution

### 2️⃣ Model Architecture
```
🏗️ DistilBERT (distilbert-base-uncased)
├── 📦 Parameters: 66,955,010
├── 🔢 Layers: 6 transformer layers
├── 🎯 Output: Binary classification (positive/negative)
└── ⚡ Efficiency: 40% smaller, 60% faster than BERT
```

### 3️⃣ Hyperparameter Configurations

#### 🟢 Configuration 1: Standard Fine-Tuning
```python
Learning Rate: 2e-5
Batch Size: 16
Epochs: 3
Weight Decay: 0.01
Warmup Steps: 500
```

#### 🟡 Configuration 2: Higher Learning Rate
```python
Learning Rate: 5e-5  # 2.5x higher
Batch Size: 16
Epochs: 3
Weight Decay: 0.01
Warmup Steps: 500
```

#### 🟣 Configuration 3: Smaller Batch + More Regularization
```python
Learning Rate: 3e-5
Batch Size: 8       # 50% smaller
Epochs: 4           # +1 epoch
Weight Decay: 0.02  # 2x higher
Warmup Steps: 500
```

### 4️⃣ Training Strategy

- ✅ **Mixed Precision Training** (FP16) for faster computation
- ✅ **Early Stopping** with patience of 3 evaluations
- ✅ **Best Model Selection** based on F1 score
- ✅ **Gradient Checkpointing** for memory efficiency
- ✅ **Learning Rate Warmup** for stable training

### 5️⃣ Evaluation Metrics

- 📊 **Accuracy**: Overall correctness
- 🎯 **F1 Score**: Harmonic mean of precision and recall
- 🔍 **Precision**: Accuracy of positive predictions
- 📈 **Recall**: Coverage of actual positives
- 🧩 **Confusion Matrix**: Detailed error breakdown

---

## 🔬 Error Analysis

### 📉 Error Distribution

- **Total Errors**: 1,740 out of 25,000 (6.96%)
- **False Positives**: 870 (50%)
- **False Negatives**: 870 (50%)

### 🎯 Common Error Patterns

1. **Sarcastic Reviews** 😏
   - Model struggles with subtle sarcasm and irony
   - Example: "Oh great, another predictable plot twist..."

2. **Mixed Sentiment** 🤔
   - Reviews with both positive and negative aspects
   - Example: "Great acting, but terrible storyline"

3. **Short Reviews** 📝
   - Limited context makes classification challenging
   - Example: "Okay." or "Not bad."

4. **Nuanced Language** 🎭
   - Complex vocabulary and subtle expressions
   - Example: "A bittersweet meditation on..."

### 💡 Suggested Improvements

- 🔄 **Data Augmentation**: Add paraphrased versions of misclassified examples
- 🤝 **Ensemble Methods**: Combine predictions from multiple configurations
- 🎯 **Domain Adaptation**: Pre-train on additional movie review corpora
- 🔧 **Advanced Techniques**: Implement LoRA for parameter-efficient fine-tuning

---

## 📸 Visualizations

<div align="center">

### 📊 Performance Metrics

![Performance Comparison](visualizations/performance_comparison.png)

### 🎯 Confusion Matrix

![Confusion Matrix](visualizations/confusion_matrix.png)

### 📈 Dataset Statistics

![Dataset Statistics](visualizations/dataset_statistics.png)

</div>

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| 🧠 **Deep Learning** | PyTorch, Transformers (Hugging Face) |
| 📊 **Data Processing** | Pandas, NumPy, Datasets |
| 📈 **Visualization** | Matplotlib, Seaborn |
| 🎯 **Evaluation** | scikit-learn |
| ☁️ **Infrastructure** | Google Colab (T4 GPU) |
| 💾 **Storage** | Google Drive |

---

## 🎥 Demo

### 🖥️ Command Line Interface
```python
# Run inference on custom text
python inference_script.py --text "This movie exceeded all my expectations!"

# Output:
# 🎬 Sentiment Analysis Result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input: "This movie exceeded all my expectations!"
# Prediction: Positive 😊
# Confidence: 98.7%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📚 Documentation

### 📖 Key Files

- 📄 **FINAL_SUMMARY_REPORT.txt** - Comprehensive technical report
- 📓 **notebooks/fine_tuning_complete.ipynb** - Complete training pipeline
- 🐍 **inference_script.py** - Production-ready inference code
- 📊 **results/comparison.csv** - Detailed metrics comparison

### 🎓 Academic Context

This project was completed as part of a **Large Language Model Fine-Tuning** assignment, demonstrating:

- ✅ Systematic hyperparameter optimization
- ✅ Rigorous evaluation methodology
- ✅ Professional documentation practices
- ✅ Production-ready implementation

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🎉 **Open** a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🤗 **Hugging Face** for the Transformers library
- 🎬 **IMDb** for the movie review dataset
- 🧠 **DistilBERT** authors for the efficient architecture
- ☁️ **Google Colab** for free GPU access

---

## 📞 Contact

**Mayuresh Satao** - satao.m@northeastern.edu(mailto:satao.m@northeastern.edu)

Project Link: [https://github.com/mayureshsatao/Distilbert-IMDB-Sentiment-Analysis](https://github.com/mayureshsatao/Distilbert-IMDB-Sentiment-Analysis)

---

<div align="center">

### 🌟 If you found this project helpful, please give it a star! 🌟

**Made with ❤️ and 🤖**

[![GitHub Stars](https://img.shields.io/github/stars/mayureshsatao/Distilbert-IMDB-Sentiment-Analysis?style=social)](https://github.com/mayureshsatao/Distilbert-IMDB-Sentiment-Analysis)
[![GitHub Forks](https://img.shields.io/github/forks/mayureshsatao/Distilbert-IMDB-Sentiment-Analysis?style=social)](https://github.com/mayureshsatao/Distilbert-IMDB-Sentiment-Analysis)

</div>