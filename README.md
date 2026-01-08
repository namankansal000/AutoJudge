# AutoJudge: Predicting Programming Problem Difficulty

## Overview
AutoJudge is a machine learning system that automatically predicts the
difficulty of programming problems using **only textual information**.
The system performs:

- **Classification** → Easy / Medium / Hard  
- **Regression** → Numerical difficulty score  

This project removes the need for manual difficulty labeling and demonstrates
how natural language processing and feature engineering can be applied to
competitive programming problems.

---
## Demo Video 
Drive link :- https://drive.google.com/file/d/1B6FFjjkyr0R0_LySDIGXJoVq_I5vEGbR/view?usp=sharing
___

## Dataset
Each problem in the dataset contains:

- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numeric difficulty score)

Total problems: **~4100**

The dataset is provided and already labeled.

---

## Data Preprocessing
- Missing values handled using empty strings
- Text fields combined into a single `combined_text`
- Text converted to lowercase for normalization

---

## Feature Extraction

### 1. TF-IDF Features
- TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text
  into numerical vectors
- Limited to the top **5000** features
- English stop words removed

### 2. Feature Engineering (Creative Features)
To improve performance and interpretability, domain-specific features were added:

- Text length
- Mathematical symbol count and density
- Algorithm keyword frequency (graph, dp, tree, bfs, dfs, etc.)
- Algorithm hint count
- Constraint count and maximum constraint value
- Constraint-related word frequency
- Sample input/output complexity

These features capture signals that raw text statistics may miss.

---

## Models

### Classification Model
- **Model:** Random Forest  
- **Task:** Predict problem difficulty class  
- **Evaluation:**
  - Accuracy ≈ **53.10%**
  - Confusion Matrix: **[[ 297 56 36]**
                       **[ 171 62 48]**
                       **[ 46 29 78]]**
  - The confusion matrix indicates that the model performs best on **Easy** and **Hard** problem classes.
  - **Hard** problems show relatively strong separability due to distinctive keywords and structural complexity in problem statements.
  - **Medium** difficulty problems are the most ambiguous, with frequent misclassification into Easy and Hard classes.
  - This ambiguity is expected, as Medium problems often overlap in textual patterns with adjacent difficulty levels and are subject to inconsistent human labeling.


### Regression Model
- **Model:** Gradient Boosting Regressor  
- **Task:** Predict numerical difficulty score  
- **Evaluation Metrics:**
  - MAE ≈ **1.617**
  - RMSE ≈ **1.940**

Gradient Boosting Regressor was chosen for its ability to model non-linear
relationships between textual features and difficulty scores. By combining
multiple weak learners, the model effectively captures complex patterns in
problem statements that linear models may miss.


---

## Web Interface
A Streamlit-based web application allows users to:

- Paste a problem description
- Predict difficulty class and score instantly

Name - Naman Kansal 
Enrollment no - 24113088
Branch - Chemical Engineering

Run the app with:
```bash
streamlit run app.py


