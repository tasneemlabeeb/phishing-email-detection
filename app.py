"""
Phishing Email Detection - Gradio App
BERT-based classifier with 95.6% accuracy
"""

import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Load model and tokenizer
MODEL_PATH = "./final_model"

print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded successfully on {device}")

def predict_phishing(email_text):
    """
    Predict if an email is phishing or legitimate
    
    Args:
        email_text: Email content to analyze
        
    Returns:
        Dictionary with prediction and confidence scores
    """
    if not email_text or len(email_text.strip()) == 0:
        return {
            "⚠️ Error": "Please enter email text",
            "Legitimate": 0.0,
            "Phishing": 0.0
        }
    
    # Tokenize
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # Prepare results
    legitimate_prob = probabilities[0][0].item()
    phishing_prob = probabilities[0][1].item()
    
    result_label = "🚨 PHISHING EMAIL DETECTED" if prediction == 1 else "✅ LEGITIMATE EMAIL"
    
    return {
        result_label: f"{confidence * 100:.1f}% confidence",
        "Legitimate": legitimate_prob,
        "Phishing": phishing_prob
    }

# Example emails
examples = [
    ["Dear Customer, Your account has been suspended. Click here immediately to verify your identity: http://suspicious-link.com/verify"],
    ["Hi Team, Please find attached the quarterly report for Q3 2024. Let me know if you have any questions. Best regards, John"],
    ["URGENT: Your PayPal account will be closed in 24 hours! Confirm your information now: http://paypal-security.xyz"],
    ["Meeting reminder: Our team sync is scheduled for tomorrow at 2 PM in Conference Room B. Please bring your progress updates."],
    ["Congratulations! You've won $1,000,000 in our lottery. Send your bank details to claim your prize immediately!"]
]

# Custom CSS
custom_css = """
#output-label {
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
}
.phishing {
    background-color: #ff4444;
    color: white;
}
.legitimate {
    background-color: #44ff44;
    color: black;
}
"""

# Create Gradio interface
with gr.Blocks(title="Phishing Email Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔒 Phishing Email Detection System
        ### BERT-based AI Classifier | 95.6% Accuracy
        
        This system uses a fine-tuned BERT model to detect phishing emails with high accuracy.
        Simply paste any email content below to analyze it.
        
        **Performance Metrics:**
        - ✅ Accuracy: 95.6%
        - ✅ Precision: 99.2%
        - ✅ Recall: 92.4%
        - ✅ F1 Score: 95.7%
        """
    )
    
    with gr.Row():
        with gr.Column():
            email_input = gr.Textbox(
                label="📧 Email Content",
                placeholder="Paste the email content here...",
                lines=10,
                max_lines=20
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Analyze Email", variant="primary")
                clear_btn = gr.ClearButton([email_input], value="Clear")
        
        with gr.Column():
            output = gr.Label(
                label="🎯 Analysis Result",
                num_top_classes=3
            )
    
    gr.Examples(
        examples=examples,
        inputs=email_input,
        label="📋 Try These Example Emails"
    )
    
    gr.Markdown(
        """
        ---
        ### 📊 About This Model
        
        **Model Details:**
        - Base Model: `bert-base-uncased`
        - Training Dataset: 2,500 phishing/legitimate emails
        - Framework: PyTorch + Hugging Face Transformers
        
        **How It Works:**
        1. Email text is tokenized using BERT tokenizer
        2. BERT model analyzes linguistic patterns and context
        3. Binary classification: Legitimate (0) or Phishing (1)
        4. Confidence scores provided for transparency
        
        **Warning Signs the Model Detects:**
        - Urgent/threatening language
        - Requests for sensitive information
        - Suspicious links and domains
        - Poor grammar/spelling patterns
        - Unusual sender behavior
        
        ---
        **GitHub Repository:** [tasneemlabeeb/phishing-email-detection](https://github.com/tasneemlabeeb/phishing-email-detection)
        
        **Model Size:** 418 MB (SafeTensors format)
        """
    )
    
    # Connect button to function
    submit_btn.click(
        fn=predict_phishing,
        inputs=email_input,
        outputs=output
    )
    
    # Also allow Enter key to submit
    email_input.submit(
        fn=predict_phishing,
        inputs=email_input,
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
