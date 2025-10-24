# Phishing Email Detection using BERT

An intelligent phishing email detection system using BERT (Bidirectional Encoder Representations from Transformers) to classify emails as legitimate or phishing with **95.6% accuracy**.

## Project Overview

With the rapid rise of AI-generated content, phishing emails have become more sophisticated and harder to detect through traditional methods. This project leverages a BERT model to analyze email content and distinguish between phishing and legitimate messages.

## Model Performance

- **Accuracy**: 95.6%
- **Precision**: 99.2%
- **Recall**: 92.4%
- **F1 Score**: 95.7%

## Technology Stack

- **Model**: BERT (bert-base-uncased)
- **Framework**: PyTorch, Hugging Face Transformers
- **Libraries**: 
  - `transformers` - BERT model and tokenization
  - `torch` - Deep learning framework
  - `scikit-learn` - Evaluation metrics
  - `pandas` - Data manipulation
  - `seaborn`, `matplotlib` - Visualization
  - `nltk` - Text processing

## 📁 Project Structure

```
phishing-detection/
├── Phishing detection (1).ipynb    # Main training notebook
├── final_model/                     # Trained model files
│   ├── model.safetensors           # Model weights (418 MB)
│   ├── config.json                 # Model configuration
│   ├── tokenizer_config.json       # Tokenizer settings
│   └── vocab.txt                   # BERT vocabulary
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

##Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- 2GB+ RAM recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/phishing-email-detection.git
cd phishing-email-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset (automatic in notebook):
```bash
# Dataset will be automatically downloaded from Kaggle via kagglehub
```

### Usage

#### Option 1: Run the Notebook
```bash
jupyter notebook "Phishing detection (1).ipynb"
```

#### Option 2: Use the Interactive Predictor
Run the last cell in the notebook to classify any email:
```python
# The cell will prompt you to paste an email
# Press Enter on an empty line to get prediction
```

## Dataset

- **Source**: [Kaggle Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
- **Total Samples**: 82,486 emails
- **Training Sample**: 2,500 emails (balanced)
- **Split**: 70% train, 10% validation, 20% test

## Model Architecture

- **Base Model**: BERT (bert-base-uncased)
- **Fine-tuning**: 2 epochs
- **Batch Size**: 8 (with gradient accumulation)
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Device**: CPU/GPU/MPS (auto-detection)

## Methodology

### Preprocessing
- **Lowercasing only** - Preserves phishing indicators like:
  - URLs (suspicious links)
  - Special characters (@, $, !)
  - Numbers (fake IDs, OTPs)
  - HTML tags
  - Deliberate misspellings

### Training Pipeline
1. Data loading and sampling
2. Text preprocessing (lowercase)
3. Tokenization (BERT tokenizer)
4. Model fine-tuning (2 epochs)
5. Evaluation on test set
6. Model saving (SafeTensors format)

## 📈 Results

### Confusion Matrix
```
                Predicted
              Legitimate  Phishing
Actual  Legitimate    XXX       XX
        Phishing       XX      XXX
```

### Classification Report
```
              precision    recall  f1-score   support

  Legitimate       0.99      0.92      0.96       XXX
    Phishing       0.93      0.99      0.96       XXX

    accuracy                           0.96       500
   macro avg       0.96      0.96      0.96       500
weighted avg       0.96      0.96      0.96       500
```

##Example Usage

### Phishing Email Example
```
URGENT: Your account has been compromised!
Click here immediately to verify your identity: http://suspicious-link.com
If you don't act within 24 hours, your account will be suspended.

Result: Phishing (87.7% confidence)
```

### Legitimate Email Example
```
Hi team, our weekly meeting is scheduled for tomorrow at 2 PM.
Please review the attached agenda beforehand.
Looking forward to seeing everyone there!

Result: Legitimate (91.2% confidence)
```

## Key Features

- High accuracy (95.6%) with minimal false positives
- Real-time email classification
- Preserves phishing indicators in preprocessing

## Model Files

The trained model is saved in SafeTensors format for:
- **Safety**: No arbitrary code execution
- **Performance**: Fast loading and inference
- **Compatibility**: Cross-platform support
- **Size**: ~418 MB (optimized)

## Deployment Options

### Hugging Face Spaces (Recommended)
```bash
# Upload final_model/ folder to Hugging Face
# Deploy with Gradio interface
```

### Streamlit Cloud
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t phishing-detector .
docker run -p 8080:8080 phishing-detector
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Your Name**
- GitHub: [Tasneem Zaman Labeeb](https://github.com/tasneemlabeeb)
- Email: your.email@example.com

## Acknowledgments

- Dataset: [Naser Abdullah Alam on Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
- BERT Model: [Google Research](https://github.com/google-research/bert)
- Hugging Face Transformers: [Hugging Face](https://huggingface.co/)

## References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Support

If you have any questions or issues, please open an issue in the GitHub repository.

---

**Star this repository if you find it helpful!**
