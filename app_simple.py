"""
Phishing Email Detection - Simple Gradio App
Works with existing notebook kernel
"""

import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "./final_model"

print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")

def predict_phishing(email_text):
    """Predict if email is phishing"""
    if not email_text or len(email_text.strip()) == 0:
        return "⚠️ Please enter email text"
    
    # Tokenize
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    result = "🚨 PHISHING EMAIL" if pred == 1 else "✅ LEGITIMATE EMAIL"
    confidence = probs[0][pred].item() * 100
    
    return f"{result}\nConfidence: {confidence:.1f}%"

# Examples
examples = [
    "Dear Customer, Your account has been suspended. Click here to verify: http://suspicious.com",
    "Hi Team, Please find the quarterly report attached. Best regards, John"
]

# Create interface
demo = gr.Interface(
    fn=predict_phishing,
    inputs=gr.Textbox(label="📧 Email Content", lines=10, placeholder="Paste email here..."),
    outputs=gr.Textbox(label="🎯 Result"),
    title="🔒 Phishing Email Detector",
    description="BERT-based AI Classifier | 95.6% Accuracy",
    examples=examples,
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
