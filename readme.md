BITS Pilani - Machine Learning Assignment 2
Student ID: 2026AK05228
Course: Machine Learning, M.Tech (AIML)
Repository: https://github.com/ajaymshr42/BITS-ML-Assignment-2
Live Streamlit app: https://bits-ml-assignment-2-wntmytgybrlabdsytqxlj4.streamlit.app/
1. Problem statement
Classify emails as spam (1) or legitimate/non-spam (0) using the same UCI Spambase dataset across five supervised learning classifiers. Evaluate discrimination, error trade-offs, and predictive performance on a held-out test set.
2. Dataset description
- Source: UCI Machine Learning Repository - Spambase (https://archive.ics.uci.edu/dataset/94/spambase)
- Original rows: 4,601; input features: 57; target: spam (0 = not spam; 1 = spam)
- Feature groups: 48 word-frequency, 6 character-frequency, and 3 capitalization statistics.
- Data quality: No missing values; 391 exact duplicate records removed prior to splitting.
- Cleaned rows: 4,210; training: 3,368 (80%); testing: 842 (20%).
- Partitioning: train_test_split(..., test_size=0.2, random_state=42, stratify=y).
- Leakage prevention: Scalers are fitted on training data only through pipelines.
3. Models and implementation
- Logistic Regression: L2 regularization, C=1.0, max_iter=2000, StandardScaler.
- Decision Tree: Gini, max_depth=6, min_samples_leaf=5.
- KNN: k=7, Euclidean distance, uniform weights, StandardScaler.
- Gaussian Naive Bayes: var_smoothing=1e-9.
- SVM: RBF kernel, C=1.0, gamma='scale', probability=True, StandardScaler.
All five classifiers use identical training and test observations. Scoring uses spam (1) as the positive class.
4. Evaluation metric comparison
Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9347	0.9768	0.9297	0.9048	0.9170	0.8634
Decision Tree	0.8848	0.9259	0.8567	0.8542	0.8554	0.7597
KNN	0.9050	0.9570	0.9103	0.8452	0.8765	0.8009
Gaussian Naive Bayes	0.8361	0.9525	0.7240	0.9524	0.8226	0.6975
Support Vector Machine	0.9323	0.9772	0.9319	0.8958	0.9135	0.8584


5. Observations on model performance
- Logistic Regression: Highest accuracy (93.47%), F1 (0.9170), and MCC (0.8634); 23 false positives and 32 missed spam messages. Strongest overall balance in this experiment.
- Decision Tree: Accuracy of 88.48%, with 48 false positives and 49 false negatives. The depth-constrained tree underperformed the linear baseline.
- KNN: Accuracy of 90.50%; 28 false positives and 52 false negatives. Higher F1 than the tree, but misses more spam than logistic regression.
- Gaussian Naive Bayes: Highest recall (95.24%) and only 16 missed spam messages, but 122 legitimate emails were incorrectly flagged; precision falls to 72.40%.
- Support Vector Machine: Highest AUC (0.9772) and precision (0.9319); 22 false positives and 35 false negatives. A close alternative to logistic regression.
6. Overall winner
Logistic Regression achieved the strongest overall balance: top Accuracy, F1, and MCC in this experiment. SVM leads marginally in AUC and Precision; Naive Bayes leads in Recall but produces many false positives. Winner is specific to this dataset, train/test split, and chosen baseline hyperparameters.
7. Streamlit functionality
The app loads five pre-trained .joblib models and the saved feature-column order. It supports uploading a CSV with the 57 predictors and the spam target, a dropdown for detailed model inspection, live comparison of six metrics for all five models, confusion matrix/classification report, ROC curves, prediction counts, and downloadable predictions. It never retrains the models on the uploaded test set.

To reproduce the reported evaluation, upload outputs/test_data_with_labels.csv (842 rows and 58 columns). The separate outputs/test_data_for_prediction.csv contains no labels and therefore cannot be used to compute evaluation metrics.
8. Repository files and running the app
- ML_Assignment_2.ipynb - EDA, data preparation, model training and evaluation.
- app.py - Streamlit frontend.
- requirements.txt - Python dependencies.
- models/ - Five pre-trained classifiers plus feature_names.joblib.
- outputs/test_data_with_labels.csv - Labeled held-out test records.
- outputs/model_comparison.csv - Experiment metric table.
Run: python -m pip install -r requirements.txt then python -m streamlit run app.py from the repository root.
9. Limitations and reproducibility
Results are based on one stratified holdout split without systematic cross-validation or hyperparameter optimization. Exact duplicate removal may remove legitimate repeated observations. The historical Spambase features do not represent modern raw-email text. The application accepts numerical feature vectors, not email text. The test set is used strictly for evaluation.
