#!/bin/bash

# Phishing Email Detection - GitHub Setup Script
# This script helps you upload your project to GitHub

echo "🔍 Phishing Email Detection - GitHub Upload Guide"
echo "=================================================="
echo ""

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first:"
    echo "   brew install git"
    exit 1
fi

echo "✅ Git is installed"

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS is not installed (needed for large model files)"
    echo ""
    echo "Install Git LFS:"
    echo "   brew install git-lfs"
    echo ""
    read -p "Do you want to continue WITHOUT Git LFS? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    USE_LFS=false
else
    echo "✅ Git LFS is installed"
    USE_LFS=true
fi

# Initialize git repository
echo ""
echo "📁 Initializing Git repository..."
if [ -d ".git" ]; then
    echo "⚠️  Git repository already exists"
else
    git init
    echo "✅ Git repository initialized"
fi

# Set up Git LFS if available
if [ "$USE_LFS" = true ]; then
    echo ""
    echo "📦 Setting up Git LFS for large files..."
    git lfs install
    git lfs track "*.safetensors"
    git lfs track "final_model/*.bin"
    git add .gitattributes
    echo "✅ Git LFS configured"
fi

# Add files
echo ""
echo "📝 Adding files to Git..."
git add .

# Show status
echo ""
echo "📊 Git Status:"
git status

# Commit
echo ""
read -p "Enter commit message (or press Enter for default): " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit: Phishing Email Detection with BERT"
fi

git commit -m "$commit_msg"
echo "✅ Files committed"

# Instructions for GitHub
echo ""
echo "=================================================="
echo "📤 Next Steps to Upload to GitHub:"
echo "=================================================="
echo ""
echo "1. Go to https://github.com/new"
echo "2. Create a new repository named: phishing-email-detection"
echo "3. Do NOT initialize with README (we already have one)"
echo "4. Copy the repository URL"
echo ""
echo "5. Then run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/phishing-email-detection.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "=================================================="
echo ""

if [ "$USE_LFS" = false ]; then
    echo "⚠️  WARNING: Model files are >100MB and Git LFS is not installed"
    echo "   Your push might fail. Options:"
    echo "   1. Install Git LFS: brew install git-lfs"
    echo "   2. Upload model separately to Hugging Face"
    echo "   3. Use GitHub Releases for model files"
    echo ""
fi

echo "Done! 🎉"
