# 🔥 Ultimate Intent Classifier 2025

An intelligent text classification application that automatically detects user intent from text input. Perfect for identifying whether someone is shopping for software, hiring employees, discussing HR topics, or just making casual conversation.

## 📋 What is This?

This application uses advanced AI (DeBERTa-v3 model) to understand what people really mean when they write text. It can classify messages into four main categories:

- **HIGH INTENT** - Shopping for software or tools
- **MEDIUM INTENT** - Hiring employees or growing team
- **LOW INTENT** - Discussing HR trends or articles
- **NO INTENT** - Off-topic, greetings, spam, or personal messages

## ✨ Features

- 🤖 Powered by state-of-the-art DeBERTa-v3 zero-shot classification model
- 🌐 Web interface using Streamlit (easy to use, no coding required)
- 💻 Command-line version for developers
- ⚡ Fast predictions with GPU support (automatically uses GPU if available)
- 📊 Shows confidence scores for each prediction
- 🎯 Pre-loaded example texts to try out

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this project** to your computer

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `streamlit` - For the web interface
   - `torch` - For AI model support
   - `transformers` - For the classification model

### Running the Application

#### Option 1: Web Interface (Recommended)

Run the Streamlit web app:

```bash
streamlit run App.py
```

This will open a web browser automatically. You can:
- Type your own text in the text box
- Select from predefined examples
- Click "Predict" to see the results

#### Option 2: Command Line

Run the command-line version:

```bash
python "Intent Detection Classifier.py"
```

Type your text when prompted. Type `exit` to quit.

## 📖 How to Use

1. **Enter your text** - Type any sentence or message you want to classify
2. **Click Predict** - The AI will analyze your text
3. **View results** - See the intent level (HIGH/MEDIUM/LOW/NONE) and confidence score

### Example Inputs

- "Looking for HR automation tools" → **HIGH INTENT**
- "Hiring an HR Manager" → **MEDIUM INTENT**
- "Top HR trends in 2025" → **LOW INTENT**
- "I love playing football" → **NO INTENT**

## 🔧 Technical Details

- **Model**: MoritzLaurer/deberta-v3-large-zeroshot-v2.0
- **Type**: Zero-shot classification (no training needed)
- **Performance**: Automatically uses GPU if available for faster processing
- **Accuracy**: High accuracy on intent detection tasks

## 📝 Project Structure

```
Task-2/
├── App.py                              # Streamlit web application
├── Intent Detection Classifier.py      # Command-line version
├── Intent Detection Classifier.ipynb   # Jupyter notebook (optional)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 💡 Tips

- The first time you run the app, it will download the AI model (this may take a few minutes)
- For best performance, use a computer with a GPU (NVIDIA graphics card)
- The model works best with clear, complete sentences
- You can try the predefined examples to see how it works

## 🤝 Need Help?

If you encounter any issues:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that you have Python 3.7 or higher
3. Ensure you have a stable internet connection (needed for first-time model download)

## 📄 License

This project is part of the Betopia Task-2 assignment.

---

**Built with ❤️ using DeBERTa-v3 and Streamlit**

