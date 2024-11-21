# XARM: Experiments in Sarcasm Detection

This package contains datasets and code for various machine learning experiments on sarcasm detection, including preprocessing, oversampling/undersampling techniques, and model training using deep learning architectures.

## Directory Structure

### Data Files
- **Amazone_Dataset.csv**: Contains text data for sarcasm detection experiments.
- **requirment_annotatted_dataset.csv**: Annotated dataset used in various models.

### Code Files

#### Preprocessing and Sampling
- **`Oversampling_and_Under_samplimpling_sarcasm_countvatrizer (2).ipynb`**: Demonstrates cleaning the text and balancing datasets with oversampling/undersampling techniques using CountVectorizer.
- **`Oversampling_and_Under_samplimpling_sarcasm_tfiddf_multiclassification.ipynb`**: Similar to the above but utilizes TF-IDF for text representation.

#### Model Training
- **`BIGRU Model.ipynb`**: Implements the Bidirectional GRU model for sarcasm classification.
- **`CNN Modell.ipynb`**: Uses Convolutional Neural Networks for sarcasm detection.
- **`GRU Model.ipynb`**: Employs a GRU-based architecture for text classification.
- **`LSTM, BILSTM_models.ipynb`**: Explores LSTM and BiLSTM models for sarcasm detection.
- **`RNN_MODEL.ipynb`**: Implements the RNN model for sarcasm classification.
- **`ROC BILSTM_OVER_UNDER_SAMPLING.ipynb`**: Focuses on evaluating the BiLSTM model's performance with balanced datasets.

### Documentation Files
- **`README.md`**: General overview of the repository.
- **`CNN LIME.pdf`**: Analysis of CNN model using LIME.
- **`CNN_SHAP_KAGGLE.pdf`**: SHAP analysis applied to CNN on Kaggle data.
- **`MLP_LIME.pdf`**: MLP model analysis using LIME.

## Requirements
- Python 3.x
- Jupyter Notebook or JupyterLab
- Libraries: `pandas`, `numpy`, `scikit-learn`, `keras`, `tensorflow`

## Running the Experiments

### Preprocessing
1. Start with one of the preprocessing notebooks:
   - **`Oversampling_and_Under_samplimpling_sarcasm_countvatrizer (2).ipynb`**
   - **`Oversampling_and_Under_samplimpling_sarcasm_tfiddf_multiclassification.ipynb`**
2. Review text cleaning and dataset balancing approaches.

### Model Training
1. Choose a notebook based on the desired model:
   - **`GRU Model.ipynb`**
   - **`BIGRU Model.ipynb`**
   - **`CNN Modell.ipynb`**
   - **`LSTM, BILSTM_models.ipynb`**
   - **`RNN_MODEL.ipynb`**
2. Run the selected notebook and follow the prompts to train the model.
3. (Optional) Use evaluation notebooks like **`ROC BILSTM_OVER_UNDER_SAMPLING.ipynb`** to analyze model performance.

### Evaluation
- Use the associated PDF reports for insights on model interpretability using LIME and SHAP.
