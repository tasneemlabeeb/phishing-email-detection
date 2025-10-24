# 🚀 Quick Start Guide

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/phishing-email-detection.git
cd phishing-email-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook "Phishing detection (1).ipynb"
```

## Using the Interactive Predictor

1. Open the notebook
2. Run all cells (or just run from the saved model loading section)
3. Scroll to the last cell
4. Run it and paste any email when prompted
5. Press Enter on an empty line
6. Get instant prediction!

## Model Performance Summary

- **Accuracy**: 95.6%
- **F1 Score**: 95.7%
- **Training Time**: ~15 minutes (on MPS/GPU)
- **Model Size**: 418 MB

## Example Predictions

### Phishing Email
```
Input: "URGENT: Your account has been suspended. Click here: http://fake-bank.com"
Output: 🚨 Phishing (confidence: 87.7%)
```

### Legitimate Email
```
Input: "Hi team, meeting tomorrow at 2 PM. Please review the agenda."
Output: ✅ Legitimate (confidence: 91.2%)
```

## Troubleshooting

### Model Files Not Found
If you get an error about missing model files, ensure you have:
1. Cloned the repository with Git LFS enabled
2. Or download the model separately from [releases](https://github.com/YOUR_USERNAME/phishing-email-detection/releases)

### Out of Memory
If you encounter memory issues:
- Reduce batch size in training arguments
- Use CPU instead of GPU (change device setting)
- Close other applications

### Dependencies Issues
```bash
# Reinstall all dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Contact

For questions or issues, please open an issue on GitHub.
