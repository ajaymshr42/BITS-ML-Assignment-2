
# Email Spam Detection Using Machine Learning

## BITS Pilani | M.Tech Artificial Intelligence & Machine Learning

**Course:** Machine Learning  
**Assignment:** Assignment 2 — Classification Model Comparison and Deployment  
**Student ID:** 2026AK05228  
**Dataset:** UCI Spambase  

**GitHub Repository:** https://github.com/ajaymshr42/BITS-ML-Assignment-2

**Live Streamlit Application:** https://bits-ml-assignment-2-wntmytgybrlabdsytqxlj4.streamlit.app/

---

## 1. Project Overview

Email spam detection is a binary classification problem in which a machine learning model distinguishes legitimate messages from unsolicited spam.

This project investigates how five supervised machine learning algorithms perform on the same numerical email dataset, with particular attention to the trade-off between **missing spam emails** and **incorrectly classifying legitimate emails as spam**.

The project implements an end-to-end machine learning workflow covering:

- Dataset exploration and statistical analysis.
- Data cleaning and preprocessing.
- Stratified train-test splitting.
- Training five supervised classification algorithms.
- Performance evaluation using six classification metrics.
- Comparative analysis and model selection.
- Model serialization using Joblib.
- Interactive deployment using Streamlit Community Cloud.

### 1.1 Project Objectives

The primary objectives are:

1. Select a publicly available classification dataset containing at least 500 observations and 12 input features.
2. Perform exploratory data analysis to understand dataset structure and quality.
3. Prepare a consistent training and testing dataset.
4. Implement five classification algorithms using identical training and testing observations.
5. Evaluate each algorithm using Accuracy, ROC AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).
6. Compare model performance and identify the most suitable classification algorithm.
7. Save trained models and deploy an interactive Streamlit application.
8. Evaluate uploaded test data using the saved models without retraining.

---

## 2. Problem Statement

Email spam detection is an important application of supervised machine learning.

The objective is to predict whether an email is:

- **Spam (`1`):** An unsolicited or unwanted email.
- **Not Spam (`0`):** A legitimate email.

The prediction is based on measurable numerical characteristics extracted from email content.

### 2.1 Classification Challenges

An inaccurate spam detection system can produce two types of classification errors.

#### False Positives

A false positive occurs when a legitimate email is classified as spam.

This may cause important personal or professional communications to be incorrectly filtered.

#### False Negatives

A false negative occurs when a spam email is classified as legitimate.

This allows unwanted messages to reach the user's inbox.

### 2.2 Evaluation Approach

Accuracy alone is insufficient for evaluating a spam detection system because different classification errors can have different consequences.

Therefore, this experiment evaluates all classifiers using six complementary performance metrics.

The final model selection considers overall performance as well as the trade-off between precision and recall.

---

## 3. Dataset Description

### 3.1 Dataset Source

**Dataset Name:** Spambase

**Repository:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/94/spambase

Spambase is a publicly available dataset designed for binary email spam classification.

It contains numerical features derived from email content, including word frequencies, character frequencies, and capital-letter patterns.

### 3.2 Dataset Characteristics

| Attribute | Description |
|---|---|
| Original observations | 4,601 |
| Input features | 57 |
| Target variable | `spam` |
| Classification type | Binary |
| Feature type | Numerical |
| Missing values | 0 |
| Duplicate records identified | 391 |
| Records after cleaning | 4,210 |
| Training observations | 3,368 |
| Testing observations | 842 |
| Train-test ratio | 80:20 |

### 3.3 Input Feature Categories

The dataset contains 57 numerical input features divided into three groups.

#### Word Frequency Features

There are 48 word-frequency features.

These represent the frequency of selected words appearing in an email.

#### Character Frequency Features

There are 6 character-frequency features.

These represent the frequency of characters such as `$`, `!`, and `#`.

#### Capital Letter Features

There are 3 capital-letter features.

These describe statistical characteristics of consecutive uppercase letters.

### 3.4 Target Variable

The target variable is named `spam`.

| Target Value | Classification |
|---|---|
| 0 | Not Spam |
| 1 | Spam |

### 3.5 Dataset Selection Rationale

Spambase was selected because:

1. It satisfies the assignment requirements of at least 500 observations and 12 features.
2. It represents a practical binary classification problem.
3. It supports comparisons between linear, nonlinear, probabilistic, and distance-based algorithms.
4. Its numerical features allow straightforward implementation using scikit-learn.
5. It provides an opportunity to study classification errors and precision-recall trade-offs.

---

## 4. Exploratory Data Analysis

### 4.1 Objective

Exploratory Data Analysis (EDA) was performed to understand the dataset before model development.

### 4.2 Analysis Performed

The following checks were conducted:

- Dataset dimensions.
- Input feature data types.
- Missing-value detection.
- Duplicate-record identification.
- Descriptive statistical analysis.
- Target class distribution.
- Visualization of spam and non-spam frequencies.

### 4.3 Initial Data Quality Findings

| Parameter | Observation |
|---|---|
| Total observations | 4,601 |
| Input features | 57 |
| Missing values | 0 |
| Exact duplicate records | 391 |
| Non-spam observations | 2,788 |
| Spam observations | 1,813 |

### 4.4 Class Distribution

The original dataset contains:

- **Non-Spam:** 60.60%
- **Spam:** 39.40%

This indicates moderate class imbalance.

To maintain similar class proportions in the training and testing datasets, stratified sampling was selected.

### 4.5 EDA Conclusion

The dataset contains no missing values and all input features are numerical.

However, duplicate records and differences in numerical feature scales require attention during preprocessing.

---

## 5. Data Preprocessing and Experimental Methodology

### 5.1 Duplicate Removal

The original dataset contains 391 exact duplicate records.

These records were removed before train-test splitting.

After cleaning, the dataset contained:

**4,210 observations and 57 input features.**

Removing exact duplicates reduces the risk of identical records appearing in both training and testing datasets.

However, this decision may also remove legitimate repeated observations.

### 5.2 Train-Test Split

The cleaned dataset was divided into:

| Dataset | Observations | Percentage |
|---|---:|---:|
| Training | 3,368 | 80% |
| Testing | 842 | 20% |
| Total | 4,210 | 100% |

The split used:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification preserved approximately:

- 60.12% non-spam observations.
- 39.88% spam observations.

### 5.3 Feature Scaling

Feature standardization was applied selectively according to algorithm requirements.

| Algorithm | Scaling Applied |
|---|---|
| Logistic Regression | StandardScaler |
| Decision Tree | Not required |
| KNN | StandardScaler |
| Gaussian Naive Bayes | Not required |
| SVM | StandardScaler |

### 5.4 Data Leakage Prevention

The experiment uses scikit-learn pipelines to ensure that preprocessing transformations are fitted exclusively on the training dataset.

The same fitted transformation is subsequently applied to unseen test observations.

The test dataset is not used for model fitting.

### 5.5 Experimental Consistency

All five algorithms were trained and evaluated using:

- The same cleaned dataset.
- The same training observations.
- The same testing observations.
- The same target variable.
- The same six evaluation metrics.

This supports a consistent experimental comparison.

---

## 6. Machine Learning Algorithms Implemented

Five supervised classification algorithms were implemented.

### 6.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of an observation belonging to a class.

It uses the sigmoid function to convert a linear combination of input features into a probability.

**Configuration:**

| Parameter | Value |
|---|---|
| Regularization | L2 |
| C | 1.0 |
| Maximum iterations | 2000 |
| Random state | 42 |
| Feature scaling | StandardScaler |

### 6.2 Decision Tree Classifier

A Decision Tree recursively partitions observations using feature-based rules.

The algorithm selects splits that reduce class impurity.

**Configuration:**

| Parameter | Value |
|---|---|
| Criterion | Gini |
| Maximum depth | 6 |
| Minimum samples per leaf | 5 |
| Random state | 42 |

### 6.3 K-Nearest Neighbors

KNN is an instance-based classification algorithm.

It predicts a class by examining the nearest training observations using a distance metric.

**Configuration:**

| Parameter | Value |
|---|---|
| Number of neighbors | 7 |
| Distance metric | Euclidean |
| Neighbor weights | Uniform |
| Feature scaling | StandardScaler |

### 6.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem.

It assumes conditional independence between features given the class and models each numerical feature using a Gaussian distribution.

**Configuration:**

| Parameter | Value |
|---|---|
| Algorithm | Gaussian Naive Bayes |
| Variance smoothing | 1e-9 |
| Feature scaling | Not required |

### 6.5 Support Vector Machine

Support Vector Machine identifies a decision boundary that separates classes while maximizing the margin between them.

The RBF kernel enables nonlinear classification.

**Configuration:**

| Parameter | Value |
|---|---|
| Kernel | RBF |
| C | 1.0 |
| Gamma | scale |
| Probability estimation | Enabled |
| Feature scaling | StandardScaler |

---

## 7. Model Evaluation Metrics

All classification models were evaluated using six metrics.

### 7.1 Accuracy

Accuracy measures the proportion of observations correctly classified.

### 7.2 ROC AUC

ROC AUC evaluates the model's ability to distinguish positive and negative classes across different decision thresholds.

### 7.3 Precision

Precision measures the proportion of predicted spam observations that are actually spam.

Higher precision indicates fewer false-positive predictions relative to predicted positives.

### 7.4 Recall

Recall measures the proportion of actual spam observations correctly identified.

Higher recall indicates fewer missed spam emails.

### 7.5 F1-Score

F1-score is the harmonic mean of precision and recall.

It summarizes the balance between these two metrics.

### 7.6 Matthews Correlation Coefficient

MCC is a correlation-based classification metric that incorporates true positives, true negatives, false positives, and false negatives.

It is useful for evaluating overall binary classification performance.

---

## 8. Model Performance Comparison

All five models were evaluated on the same **842 held-out test observations**.

### 8.1 Final Evaluation Results

| Classification Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.9347** | 0.9768 | 0.9297 | 0.9048 | **0.9170** | **0.8634** |
| Decision Tree | 0.8848 | 0.9259 | 0.8567 | 0.8542 | 0.8554 | 0.7597 |
| KNN | 0.9050 | 0.9570 | 0.9103 | 0.8452 | 0.8765 | 0.8009 |
| Gaussian Naive Bayes | 0.8361 | 0.9525 | 0.7240 | **0.9524** | 0.8226 | 0.6975 |
| Support Vector Machine | 0.9323 | **0.9772** | **0.9319** | 0.8958 | 0.9135 | 0.8584 |

### 8.2 Best Model by Evaluation Metric

| Metric | Best Performing Model |
|---|---|
| Accuracy | Logistic Regression |
| AUC | Support Vector Machine |
| Precision | Support Vector Machine |
| Recall | Gaussian Naive Bayes |
| F1-Score | Logistic Regression |
| MCC | Logistic Regression |

---

## 9. Model-Specific Performance Observations

### 9.1 Logistic Regression

Logistic Regression achieved the highest overall accuracy of **93.47%**.

It correctly classified 787 out of 842 test observations.

Its F1-score of **0.9170** and MCC of **0.8634** were the highest among the five evaluated models.

The classifier produced:

- True Negatives: 483
- False Positives: 23
- False Negatives: 32
- True Positives: 304

The results indicate strong overall classification performance and a good balance between precision and recall.

### 9.2 Decision Tree

Decision Tree achieved **88.48% accuracy** and an F1-score of **0.8554**.

The classifier produced:

- True Negatives: 458
- False Positives: 48
- False Negatives: 49
- True Positives: 287

Although Decision Trees can capture nonlinear relationships and feature interactions, the selected depth-limited configuration performed below Logistic Regression.

### 9.3 K-Nearest Neighbors

KNN achieved **90.50% accuracy** and an F1-score of **0.8765**.

The classifier produced:

- True Negatives: 478
- False Positives: 28
- False Negatives: 52
- True Positives: 284

KNN outperformed Decision Tree in accuracy, F1-score, AUC, and MCC.

However, it missed 52 spam emails, resulting in comparatively lower recall.

### 9.4 Gaussian Naive Bayes

Gaussian Naive Bayes achieved **83.61% accuracy**.

Its most notable result was the highest recall of **95.24%**.

The classifier produced:

- True Negatives: 384
- False Positives: 122
- False Negatives: 16
- True Positives: 320

The model successfully detected 320 out of 336 spam emails.

However, it also incorrectly classified 122 legitimate emails as spam.

This demonstrates a clear precision-recall trade-off.

The classifier may be useful when detecting most spam messages is prioritized over minimizing false positives.

### 9.5 Support Vector Machine

SVM achieved **93.23% accuracy**.

It produced the highest AUC of **0.9772** and the highest precision of **0.9319**.

The classifier produced:

- True Negatives: 484
- False Positives: 22
- False Negatives: 35
- True Positives: 301

SVM achieved performance close to Logistic Regression.

Although it performed slightly better in AUC and precision, Logistic Regression achieved higher accuracy, recall, F1-score, and MCC.

---

## 10. Overall Winning Model

### Selected Model: Logistic Regression

Logistic Regression was selected as the preferred classifier for the evaluated experimental configuration.

### 10.1 Selection Justification

The model achieved:

| Metric | Value |
|---|---:|
| Accuracy | 93.47% |
| AUC | 0.9768 |
| Precision | 92.97% |
| Recall | 90.48% |
| F1-Score | 0.9170 |
| MCC | 0.8634 |

The selection is based primarily on:

1. Highest overall classification accuracy.
2. Highest F1-score.
3. Highest MCC.
4. Balanced precision and recall.
5. Relatively low numbers of false positives and false negatives.

### 10.2 Alternative Models

**Support Vector Machine** is a competitive alternative when AUC or precision is prioritized.

**Gaussian Naive Bayes** may be considered when maximizing spam detection recall is more important than avoiding false alarms.

### 10.3 Final Interpretation

Logistic Regression provides the strongest overall classification performance for this experiment.

However, model superiority is specific to the selected dataset, train-test split, and hyperparameter configurations.

---

## 11. Streamlit Web Application

### 11.1 Application Overview

An interactive Streamlit application was developed to demonstrate the five trained classification models.

**Live Application:**

https://bits-ml-assignment-2-wntmytgybrlabdsytqxlj4.streamlit.app/

### 11.2 Application Features

The application supports:

1. CSV test dataset upload.
2. Selection of one of five classification models.
3. Evaluation on previously unseen labeled test observations.
4. Calculation of six classification metrics.
5. Comparison of classification models.
6. Confusion matrix visualization.
7. Classification report.
8. ROC curve visualization.
9. Prediction distribution visualization.
10. Downloading classification predictions.

### 11.3 Model Loading

The application loads previously trained models stored in Joblib format.

All model training and preprocessing are performed before deployment.

**The application does not retrain models on uploaded test data.**

### 11.4 Test Dataset Requirements

The uploaded CSV must contain:

- All 57 numerical input features.
- The target column named `spam`.
- Binary target labels (`0` or `1`).

The labeled testing dataset is available at:

[View Test Dataset](./outputs/test_data_with_labels.csv)

### 11.5 How to Use the Application

1. Open the deployed Streamlit application.
2. Upload the labeled test CSV.
3. Review the classification performance comparison.
4. Select a model from the dropdown.
5. Examine its evaluation metrics and confusion matrix.
6. Inspect the ROC curve and prediction distribution.
7. Download prediction results if required.

---

## 12. Repository Structure

The project repository contains the following components:

```text
BITS-ML-Assignment-2/
|
|-- ML_Assignment_2.ipynb
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- models/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- feature_names.joblib
|
|-- outputs/
|   |-- test_data_with_labels.csv
|   |-- test_data_for_prediction.csv
|   |-- model_comparison.csv
```

### 12.1 File Descriptions

| File or Folder | Purpose |
|---|---|
| `ML_Assignment_2.ipynb` | Model development, training, EDA, and evaluation |
| `app.py` | Streamlit application source code |
| `requirements.txt` | Required Python dependencies |
| `README.md` | Project documentation |
| `models/` | Trained classification models |
| `models/feature_names.joblib` | Required input feature schema |
| `outputs/test_data_with_labels.csv` | Labeled held-out test dataset |
| `outputs/test_data_for_prediction.csv` | Feature-only test dataset |
| `outputs/model_comparison.csv` | Model evaluation results |

---

## 13. Installation and Execution

### 13.1 Prerequisites

The application requires Python and the following libraries:

- Streamlit
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Joblib

### 13.2 Clone the Repository

```bash
git clone https://github.com/ajaymshr42/BITS-ML-Assignment-2.git

cd BITS-ML-Assignment-2
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### 13.3 Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 13.4 Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

The application normally opens at:

http://localhost:8501

### 13.5 Run the Jupyter Notebook

Open the notebook:

`ML_Assignment_2.ipynb`

Execute all cells sequentially to reproduce:

1. Dataset loading.
2. Exploratory data analysis.
3. Duplicate removal.
4. Train-test splitting.
5. Model training.
6. Evaluation metrics.
7. Confusion matrices.
8. Model comparison.
9. Model serialization.
10. Test dataset exports.

---

## 14. Limitations and Future Improvements

### 14.1 Single Train-Test Split

All models were evaluated using one stratified train-test split.

Cross-validation would provide a more reliable estimate of expected performance.

### 14.2 Hyperparameter Optimization

The models were trained using selected baseline configurations.

Systematic hyperparameter tuning was not performed.

Future experiments could explore GridSearchCV or RandomizedSearchCV.

### 14.3 Historical Dataset

Spambase represents an established email classification dataset.

Its characteristics may not fully reflect contemporary spam patterns.

### 14.4 Numerical Features

The application accepts 57 numerical input features.

It does not directly classify raw email text.

A future enhancement could introduce text preprocessing and feature extraction.

### 14.5 Classification Thresholds

A fixed decision threshold may not be optimal for all use cases.

Future improvements could evaluate different classification thresholds based on the relative costs of false positives and false negatives.

---

## 15. Conclusion

This project demonstrates an end-to-end supervised machine learning workflow for email spam detection.

Five classification algorithms were implemented using the UCI Spambase dataset.

The models were evaluated consistently using six classification metrics and a common held-out test dataset.

The experiment produced three important findings:

1. Logistic Regression provided the strongest overall classification performance based on accuracy, F1-score, and MCC.
2. Support Vector Machine achieved the highest AUC and precision.
3. Gaussian Naive Bayes achieved the highest recall but produced considerably more false positives.

Based on the evaluated configurations, **Logistic Regression was selected as the preferred classification model**.

The trained models were saved and integrated into an interactive Streamlit application, demonstrating a complete workflow from model development to deployment.

---

## 16. References

### Dataset

UCI Machine Learning Repository — Spambase

https://archive.ics.uci.edu/dataset/94/spambase

### Machine Learning Framework

scikit-learn

https://scikit-learn.org/stable/

### Streamlit Documentation

https://docs.streamlit.io/

### GitHub Repository

https://github.com/ajaymshr42/BITS-ML-Assignment-2

### Live Streamlit Application

https://bits-ml-assignment-2-wntmytgybrlabdsytqxlj4.streamlit.app/

---

## Project Information

**Institution:** BITS Pilani  
**Programme:** M.Tech Artificial Intelligence & Machine Learning  
**Course:** Machine Learning  
**Assignment:** 2  
**Student ID:** 2026AK05228  

**Project:** Comparative Analysis of Machine Learning Classifiers for Email Spam Detection
