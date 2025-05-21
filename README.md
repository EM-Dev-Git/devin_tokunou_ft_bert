# BERT Fine-tuning for Job Classification

This project implements a complete pipeline for fine-tuning BERT (Bidirectional Encoder Representations from Transformers) models to classify job descriptions into different categories. The system allows users to take a pre-trained BERT model and adapt it to recognize and categorize job postings into roles such as Data Scientist, Machine Learning Engineer, Software Engineer, and Consultant.

## Project Overview

The project consists of a sequential pipeline with four main stages:

1. **Data Preparation**: Creates a sample dataset of job descriptions with corresponding category labels
2. **Resource Acquisition**: Downloads the pre-trained BERT model and vocabulary files
3. **Model Training**: Fine-tunes the BERT model on the job description dataset
4. **Inference**: Uses the fine-tuned model to predict categories for new job descriptions

## Features

- Complete BERT fine-tuning pipeline for text classification
- Custom tokenizer implementation for processing job descriptions
- Visualization of learning curves and model performance metrics
- Pre-defined job categories with sample data generation
- Easy-to-use inference module for classifying new job descriptions

## Project Structure

```
devin_tokunou_ft_bert/
├── data/                # Directory for training and testing datasets
├── vocab/               # Directory for BERT vocabulary files
├── weights/             # Directory for pre-trained model weights
├── results/             # Directory for fine-tuned model and learning curves
├── utils/               # Utility modules
│   ├── bert.py          # BERT model implementation
│   ├── tokenizer.py     # Tokenization functionality
│   └── __init__.py      # Package initialization
├── main.py              # Main pipeline orchestration
├── download_bert_files.py  # Downloads vocabulary and pre-trained weights
├── create_sample_dataset.py # Generates training and testing datasets
├── train_model.py       # Implements the fine-tuning process
└── predict.py           # Provides inference functionality
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/EM-Dev-Git/devin_tokunou_ft_bert.git
   cd devin_tokunou_ft_bert
   ```

2. Install the required dependencies:
   ```bash
   pip install torch pandas numpy matplotlib sklearn tqdm transformers
   ```

## Usage

### Running the Complete Pipeline

To execute the entire fine-tuning pipeline from data preparation to inference:

```bash
python main.py
```

This will:
1. Download the BERT vocabulary file and pre-trained model
2. Create sample job description datasets
3. Fine-tune the BERT model on the datasets
4. Run inference on example job descriptions

### Individual Components

You can also run each component separately:

1. Download BERT files:
   ```bash
   python download_bert_files.py
   ```

2. Create sample dataset:
   ```bash
   python create_sample_dataset.py
   ```

3. Train the model:
   ```bash
   python train_model.py
   ```

4. Run inference:
   ```bash
   python predict.py
   ```

## Vocabulary File Usage

The BERT vocabulary file (bert-base-uncased-vocab.txt) is a critical component that serves several purposes:

1. Acts as a reference dictionary for tokenizing text input
2. Assigns unique IDs to each token for model processing
3. Provides mapping between tokens and their numerical representations
4. Handles unknown words by replacing them with [UNK] token
5. Enables conversion between token IDs and human-readable tokens

## Model Architecture

The project implements a custom BERT architecture with the following components:

- **BertTokenizer**: Handles text tokenization and conversion between tokens and IDs
- **BertModel**: Implements the core BERT architecture with transformer layers
- **BertForSequenceClassification**: Extends BertModel for text classification tasks
- **JobDataset**: Custom dataset class for handling job description data

## License

This project is available for educational and research purposes.
