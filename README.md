
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
<div align="centre">
  <img src="Results/Screenshot%202025-02-07%20131918.png" alt="Prediction Interface">
  <p><b>Figure 1:</b> Prediction Interface</p>
</div>

<div align="centre">
  <img src="Results/Screenshot%202025-02-07%20131949.png" alt="Chatbot Interface">
  <p><b>Figure 2:</b> Chatbot Interface</p>
</div>





---

## LLM Chatbot Integration
The chatbot is  fine-tuned on **survey-based mental health queries**. It provides general guidance, emotional support, and self-care tips, ensuring responsible AI usage.

**Prompt Engineering** was applied to enhance chatbot responses while avoiding medical claims.

---
## RESULTS 

### Model Evaluation

#### Evaluation for Random Forest:
- **Accuracy:** 0.8095238095238095  
- **Precision:** 0.7836538461538461  
- **Recall:** 0.8578947368421053  
- **F1 Score:** 0.8190954773869347  
- **ROC-AUC:** 0.8092665173572228  

#### Evaluation for Neural Network:
- **Accuracy:** 0.7619047619047619  
- **Precision:** 0.7717391304347826  
- **Recall:** 0.7473684210526316  
- **F1 Score:** 0.7593582887700535  
- **ROC-AUC:** 0.7619820828667414  

  


## Conclusion
This project demonstrates a robust approach to **mental health prediction and conversational AI**. Future improvements will focus on **reinforcement learning** and **explainability methods** for enhanced interpretability and user trust.
