# Project: Sentiment Classification of Tweets
Some issues such as the model files being a bit too large can be resolved by downloading them in 
https://drive.google.com/file/d/1Dnymf-N61KrMNLQi5rIhSE96Ungg8s7N/view?usp=drive_link
## 1. Introduction

This project aims to solve a nuanced sentiment classification task: predicting whether a tweet originally contained a positive `:)` or negative `:(` smiley based on its textual content.

We explored a wide range of methodologies to find the most effective solution, progressing from classic machine learning baselines to modern Transformer architectures. This repository contains the code for all experiments conducted, including data preprocessing, model training, evaluation, and visualization.

Our final and best-performing model is a **fine-tuned DistilBERT**, which demonstrates the power of transfer learning for this task.

## 2. Project Structure

The repository is organized into several key scripts, each representing a different model or utility.

### Core Model Scripts

* `run.py`: **(Final Submission Script)** Loads the best-performing fine-tuned DistilBERT model (`best_model_state.bin`) and generates the `submission.csv` file for the test dataset.
* `transformer_finetune.py`: The main script for fine-tuning the pre-trained `distilbert-base-uncased` model. This script trains the model and saves the best version.
* `from_zero.py`: An experimental script to train a Transformer model completely from scratch to benchmark against the pre-trained approach.
* `baseline_model.py`: A simple feed-forward neural network that serves as a strong baseline, trained with custom embeddings.
* `glove_pretrain.py`: A baseline script that uses pre-trained GloVe embeddings with a Logistic Regression classifier.
* `train_glove.py`: A script to train custom Word2Vec embeddings from the tweet corpus and evaluate them with a Logistic Regression classifier.

### LLM Baseline Scripts

These scripts were used to evaluate the zero-shot performance of various Large Language Models.
**Note:** All of these scripts require a valid API key to be set within the file.

* `4o.py`: Evaluates **GPT-4o** using the generic sentiment analysis prompt (Prompt 1).
* `4o_smile.py`: Evaluates **GPT-4o** using the more specific smiley-prediction prompt (Prompt 2).
* `gemini.py`: Evaluates **gemini-1.5-flash-0520** using the smiley-prediction prompt (Prompt 2).
* `llm_baseline.py`: Evaluates **DeepSeek v3 0324** using the generic sentiment analysis prompt (Prompt 1).

### Utilities & Visualization

* `download_model.py`: A utility to download the `distilbert-base-uncased` model and tokenizer from Hugging Face for offline use.
* `plot_results.py`: A Python script using `matplotlib` to generate the publication-quality performance comparison chart (`model_performance_comparison.png`) from the experimental results.
* `build_vocab.sh` / `cut_vocab.sh`: Shell scripts for building and pruning a custom vocabulary from the raw text files.
* `pickle_vocab.py`: A utility to serialize the vocabulary into a `.pkl` file.
* `cooc.py`: A utility to build a co-occurrence matrix, used in early custom GloVe experiments.

## 3. Requirements

To run the scripts in this project, you will need Python 3.8+ and the following libraries:

* `torch`
* `transformers`
* `datasets`
* `scikit-learn`
* `numpy`
* `tqdm`
* `matplotlib`
* `openai`
* `google-generativeai`
* `gensim`

You can install them using pip:
```bash
pip install torch transformers datasets scikit-learn numpy tqdm matplotlib openai google-generativeai gensim
```

## 4. Execution Guide

### Step 1: Dataset Setup
Ensure the dataset files (`train_pos.txt`, `train_neg.txt`, etc.) are located inside a folder named `twitter-datasets/` in the project's root directory.

### Step 2: Running the Final Model (Recommended)

This is the main workflow to get the best result and generate the submission file.

1.  **Download Pre-trained Model (for offline use):**
    First, run the download script to save the DistilBERT model locally. This only needs to be done once.
    ```bash
    python download_model.py
    ```

2.  **Train the Fine-tuned Model:**
    Run the fine-tuning script. This will train the DistilBERT model for 2 epochs and save the best checkpoint as `best_model_state.bin`.
    ```bash
    python transformer_finetune.py
    ```

3.  **Generate Submission File:**
    Execute the final run script to generate the predictions for the test set.
    ```bash
    python run.py
    ```
    This will create the `submission.csv` file.

### Step 3: Running Baseline Experiments (Optional)

If you wish to reproduce the results from the various baseline models, you can run their respective scripts.

* **To run the simple NN baseline:**
    ```bash
    python baseline_model.py
    ```

* **To run an LLM baseline (e.g., GPT-4o):**
    * Edit the `4o.py` file and insert your OpenAI API key.
    * Then, run the script:
    ```bash
    python 4o.py
    ```

### Step 4: Generating the Performance Chart

After running the experiments, you can generate the summary chart.
```bash
python plot_results.py
```
This will create a high-resolution PNG file named `model_performance_comparison.png`.

## 5. Results

Our experiments consistently showed that the **fine-tuned DistilBERT model achieved the highest performance**, with a validation accuracy of **88.07%**. This significantly outperforms all other methods, including LLMs in a zero-shot setting and models trained from scratch.

For a detailed breakdown of all results and a comprehensive analysis, please refer to the `report.pdf` document included in this project.

