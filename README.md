# Figurative Language Understanding: Sentiment and Sarcasm Analysis on BESSTIE

- Elena Nespolo s345176
- Romeo Vercellone s341967
- Giuseppe Mallo s346884
- Carla Finocchiaro s337024
- Alessandra Marchese s349536

This repository contains the implementation for the **Figurative Language Understanding** project for the Deep Natural Language Processing course (2025).

This project focuses on Sentiment Analysis and Sarcasm Detection across different varieties of English (Australian, Indian, British). It is based on the **BESSTIE benchmark** and explores architectural extensions (CNN, LSTM heads), efficiency improvements (Sparse Attention), and domain shift robustness (Twitter, Bicodemix).


## How to run

The entire pipeline (Data download, Training, Validation, Testing) is self-contained in a single Jupyter Notebook designed to run on Google Colab.

### Google Colab
You can run the training directly in the cloud using a GPU.

1.  Open the notebook `train_on_BESSTIE.ipynb`.
2.  Click on the "Open in Colab" button (if available) or upload it to your Drive.
3.  **Important:** Make sure to select a GPU Runtime (`Runtime` > `Change runtime type` > `T4 GPU`).
4.  Run the cells sequentially. The notebook will automatically clone this repository to fetch the necessary utility scripts.

### Alternative: Local Exectuion
If you prefer running it locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/elenanespolo/Sentiment_Sarcasm_Analysis.git
   cd Sentiment_Sarcasm_Analysis
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy torch os transformers collections matplotlib seaborn scikit-learn
   ```
3. Run the notebook or the script.

  
 ## Configuration
 You can reproduce different experiments by modifying the `CFG` dictionary at the beginning of the notebook:

*   **`training_dataset_name`**: Choose the source dataset (`'BESSTIE'`, `'twitter'`, `'bicodemix'`).
*   **`classification_head`**: Select the architecture (`'linear'`, `'conv'`, `'lstm'`).
*   **`use_spAtten`**: Set to `True` to enable Sparse Attention, `False` for standard BERT.
*   **`variety`**: Select the English variety for training (`'en-AU'`, `'en-IN'`, `'en-UK'`).

Example configuration for the Baseline:
```python
training_dataset_name = 'BESSTIE'
classification_head = 'linear'
use_spAtten = False

