# 📰 AI News Lab – Fake News Generator & Detector

AI News Lab is an educational machine learning project that demonstrates both the creative and analytical capabilities of modern Natural Language Processing (NLP). The application combines **GPT-2**, a powerful language generation model, with a **RoBERTa-based Fake News Detector** to showcase how AI can generate realistic news articles and analyze news content for authenticity.

The project features two interactive modules built with **Gradio**. The **Fake News Generator** allows users to provide a headline or opening sentence, which GPT-2 expands into a complete, realistic-looking news article using configurable generation parameters such as temperature, top-k sampling, top-p sampling, and output length. This helps users understand how generative AI can produce convincing text.

The **Fake News Detector** uses the pre-trained **winterForestStump/Roberta-fake-news-detector** model from Hugging Face to classify news articles as **Fake** or **Real**. Instead of simply returning a label, the model provides confidence scores, giving users insight into the prediction probabilities. While the detector has been fine-tuned for fake news classification, it is intended solely for educational and research purposes and should not be considered a replacement for professional fact-checking.

The application automatically utilizes GPU acceleration when available and falls back to CPU execution if necessary. A clean, modern interface built with Gradio makes the project easy to use without requiring technical expertise.

## Features

* 🤖 AI-powered fake news generation using GPT-2
* 🔍 Fake vs Real news detection using a fine-tuned RoBERTa model
* 📊 Confidence score visualization
* 🎛 Adjustable text generation parameters
* 💻 Interactive Gradio web interface
* ⚡ Automatic GPU/CPU support
* 🎓 Ideal for learning NLP, Transformers, and AI ethics

This project was developed as an educational demonstration to highlight both the strengths and limitations of AI-generated content and automated misinformation detection. It serves as a practical example of how transformer-based models can be applied to real-world NLP tasks while promoting responsible AI usage.
