
# Self-Analysis Mental Health Model

## Overview
The **Self-Analysis Mental Health Model** is designed to predict possible mental health conditions based on user-provided symptoms. It integrates **Random Forest** and **Neural Networks** for classification and includes a **LLM-powered chatbot** for general mental health discussions. The Gradio-based UI consists of two tabs:

- **Tab 0**: Prediction and diagnosis using **Random Forest** and **Neural Networks** based on the dataset `mentalhealth_disorder.xlsx` and 'survey.csv'.
- **Tab 1**: A conversational **LLM chatbot** (fine-tuned Mistral 7B) to assist users with mental health queries. It does not diagnose but provides general support and guidance.

---

## Dataset Preprocessing Steps

The model uses **two datasets**:
1. **mentalhealth_disorder.xlsx** – Used for classification of mental health conditions.
2. **Survey Dataset** – Used for classification .

### Steps:
1. **Data Cleaning**: Removed null values and duplicate entries.
2. **Feature Engineering**: Encoded categorical variables and normalized numerical features.
3. **Splitting Data**: Separated into training (80%) and testing (20%) datasets.
4. **Label Encoding**: Applied label encoders to map categorical labels to numerical values.
5. **Saving Preprocessed Data**: Stored the processed dataset for model training.

---

## Model Selection Rationale

The **Random Forest** and **Neural Networks** were chosen for their advantages:
- **Random Forest**: Handles high-dimensional data well and reduces overfitting.
- **Neural Networks**: Captures complex patterns and relationships in data for better accuracy.

### Model Comparison
Both models were evaluated based on the following metrics:
- **Precision**
- **Recall**
- **F1-score**
- **Accuracy**
- **ROC-AUC** (for binary or multi-class classification)

### Model Interpretation
- **SHAP** or **LIME** was used to interpret model predictions.

### Testing the Model
A sample script (`sample_prediction.py`) is provided to test the model.
---

## Running the Inference Script

1. Install required dependencies:
   ```sh
   pip install gradio langchain langchain_community pandas scikit-learn
   ```

2. Load the trained models before running the UI:
   ```python
   import pickle
   
   with open("random_forest_model.pkl", "rb") as f:
       random_forest_model = pickle.load(f)
   
   with open("neural_network_model.pkl", "rb") as f:
       neural_network_model = pickle.load(f)
   ```

3. Run the **Gradio UI** for interaction:
   ```sh
   python mental_health_UI(1).py
   ```

---

## UI/CLI Usage Instructions

### **Gradio Interface**
1. **Install Gradio**:
   ```sh
   pip install gradio
   ```
2. **Run the UI files**:
   ```sh
   python mental_health_UI(1).py
   python mental_health_UI(2).py
   ```

### **Gradio Tabs:**
![Prediction Interface](Results/Screenshot%202025-02-07%20131918.png)
![Chatbot Interface](Results/Screenshot%202025-02-07%20131949.png)




---

## LLM Chatbot Integration
The chatbot is  fine-tuned on **survey-based mental health queries**. It provides general guidance, emotional support, and self-care tips, ensuring responsible AI usage.

**Prompt Engineering** was applied to enhance chatbot responses while avoiding medical claims.

---

## Conclusion
This project demonstrates a robust approach to **mental health prediction and conversational AI**. Future improvements will focus on **reinforcement learning** and **explainability methods** for enhanced interpretability and user trust.
