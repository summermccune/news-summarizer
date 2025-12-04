# News Summarizer with Visual Context & RAG

A comprehensive system for summarizing news articles and answering questions using both textual and visual content, powered by Retrieval-Augmented Generation (RAG).

## Overview

This project addresses two major challenges in today's media landscape:
- **Misinformation**: By providing accurate, source-grounded summaries and answers
- **Information Overload**: By condensing large volumes of news into digestible summaries

## Features

### Phase 1: Text-Only Summarization
- Summarize news articles using state-of-the-art transformer models
- Support for multiple summarization datasets (CNN/DailyMail, XSum, Newsroom)
- Evaluation metrics (ROUGE, BLEU, BERTScore)

### Phase 2: Visual Context Integration
- Extract and process images from news articles
- Generate image captions and visual context
- Enhance summaries with visual understanding

### Phase 3: RAG-based Q&A
- Answer questions about articles using RAG
- Retrieve relevant sentences and images from source material
- Ground answers in original content to prevent hallucinations

## Project Structure

```
news-summarizer/
├── src/
│   ├── data/              # Dataset loaders and preprocessing
│   ├── models/            # Summarization and Q&A models
│   ├── vision/            # Image processing and understanding
│   ├── rag/               # RAG implementation
│   └── evaluation/        # Evaluation metrics
├── examples/              # Example scripts
├── tests/                 # Unit tests
├── data/                  # Data storage (gitignored)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Text Summarization

```python
from src.models.summarizer import TextSummarizer

summarizer = TextSummarizer()
summary = summarizer.summarize("Your news article text here...")
print(summary)
```

### Multimodal Summarization

```python
from src.models.multimodal_summarizer import MultimodalSummarizer

summarizer = MultimodalSummarizer()
summary = summarizer.summarize(
    text="Article text...",
    images=["path/to/image1.jpg", "path/to/image2.jpg"]
)
print(summary)
```

### Question Answering with RAG

```python
from src.rag.qa_system import QASystem

qa = QASystem()
qa.index_article("Article text...", images=["image1.jpg"])
answer = qa.answer_question("What is the main topic?")
print(answer)
```

## Datasets

### Text Summarization
- **CNN/DailyMail**: ~300k articles with human-written summaries
- **XSum**: ~220k articles with single-sentence summaries  
- **Newsroom**: ~1.3M articles with summaries and titles

### Visual Content
- **N24News**: ~24k articles with images and captions

## Evaluation

Run evaluation on test datasets:

```bash
python examples/evaluate_summarization.py --dataset cnn_dailymail
```

## Challenges & Solutions

### Challenge 1: Combining Text and Image Data
**Solution**: Use vision-language models (CLIP, BLIP) to encode images and create unified embeddings

### Challenge 2: Evaluating Summaries
**Solution**: Implement multiple metrics (ROUGE, BLEU, BERTScore) and human evaluation framework

## Development Roadmap

- [x] Project setup and structure
- [ ] Text-only summarization implementation
- [ ] Dataset loaders for all datasets
- [ ] Image understanding module
- [ ] Multimodal summarization
- [ ] RAG system for Q&A
- [ ] Evaluation framework
- [ ] Example scripts and documentation

## License

MIT License

## Contributors

Summer McCune
