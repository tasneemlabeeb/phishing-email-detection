# 📤 GitHub Upload Instructions

## Quick Upload (Without Git LFS)

Since Git LFS is not installed and your model files are large (427 MB), here are two options:

---

## ⚡ OPTION 1: Upload Without Model Files (Recommended for Quick Share)

### Step 1: Install Git LFS (Recommended)
```bash
# Install Git LFS via Homebrew
brew install git-lfs

# Initialize Git LFS
git lfs install
```

### Step 2: Initialize Git Repository
```bash
cd ~/Downloads

# Initialize git
git init

# Configure Git LFS for large files
git lfs track "*.safetensors"
git lfs track "final_model/*.bin"

# Add all files
git add .

# Commit
git commit -m "Initial commit: Phishing Email Detection with BERT"
```

### Step 3: Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `phishing-email-detection`
3. Description: `BERT-based phishing email detector with 95.6% accuracy`
4. Choose `Public` (so your faculty can access it)
5. Do NOT check "Initialize this repository with a README"
6. Click `Create repository`

### Step 4: Push to GitHub
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/phishing-email-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 5: Share with Faculty
Your repository will be at:
```
https://github.com/YOUR_USERNAME/phishing-email-detection
```

---

## 🎯 OPTION 2: Upload Model to Hugging Face (Better for Large Files)

If Git LFS doesn't work, upload the model separately:

### Step 1: Upload Project Without Model
```bash
cd ~/Downloads

# Initialize git
git init

# Temporarily ignore model files
echo "final_model/*.safetensors" >> .gitignore
echo "final_model/*.bin" >> .gitignore

# Add and commit
git add .
git commit -m "Initial commit: Phishing Email Detection (model hosted on HF)"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/phishing-email-detection.git
git branch -M main
git push -u origin main
```

### Step 2: Upload Model to Hugging Face
1. Go to https://huggingface.co/new
2. Create account (if needed)
3. Click "New Model"
4. Upload your `final_model/` folder
5. Add this note to your README:

```markdown
## 📦 Model Files

The trained model is hosted on Hugging Face due to file size:
https://huggingface.co/YOUR_USERNAME/phishing-detector

Download with:
```python
from transformers import BertForSequenceClassification, BertTokenizer
model = BertForSequenceClassification.from_pretrained("YOUR_USERNAME/phishing-detector")
tokenizer = BertTokenizer.from_pretrained("YOUR_USERNAME/phishing-detector")
```

---

## 🚀 QUICK START (Easiest Method)

Run the automated setup script:

```bash
cd ~/Downloads
./setup_github.sh
```

Then follow the on-screen instructions!

---

## ✅ Verification

After uploading, your repository should contain:
- ✅ `README.md` - Project documentation
- ✅ `requirements.txt` - Python dependencies
- ✅ `Phishing detection (1).ipynb` - Training notebook
- ✅ `final_model/` - Trained model (if Git LFS works)
- ✅ `.gitignore` - Git ignore file
- ✅ `LICENSE` - MIT License

---

## 🆘 Troubleshooting

### Issue: "File size exceeds 100 MB"
**Solution**: Install Git LFS (see Option 1) or use Hugging Face (Option 2)

### Issue: "Git LFS not installed"
**Solution**: 
```bash
brew install git-lfs
git lfs install
```

### Issue: "Permission denied"
**Solution**: Set up SSH keys or use HTTPS with Personal Access Token
- GitHub SSH setup: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### Issue: "Repository already exists"
**Solution**: 
```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/YOUR_USERNAME/phishing-email-detection.git
```

---

## 📧 Share with Your Faculty

Once uploaded, share this link:
```
https://github.com/YOUR_USERNAME/phishing-email-detection
```

Your faculty can:
1. View the code and README
2. Clone the repository
3. Run the notebook
4. See your model performance

---

## 💡 Pro Tip

Add a nice badge to your README:
```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95.6%25-success.svg)
```

Good luck! 🎓
