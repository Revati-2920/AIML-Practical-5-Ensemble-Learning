# Ensemble Learning and Performance Evaluation
 
**TY B.Tech. – Artificial Intelligence and Machine Learning | Practical Assignment 5**
 
- **Name:** Revati Bhosale
- **Division:** C
- **Batch:** C2
- **Roll No.:** 173
- **PRN:** 202401040048
---
 
## About the Project
 
This project compares **ensemble learning** methods with individual machine learning models on a binary classification problem (diagnosing a breast tumour as *malignant* or *benign*).
 
Three ensemble methods are implemented using scikit-learn:
 
1. **Stacking** – five base classifiers at level 0 and a Logistic Regression meta-classifier at level 1, trained on 5-fold out-of-fold predictions.
2. **AdaBoost** – 100 weak learners, using decision stumps (depth 1) and depth-3 decision trees.
3. **Voting** – hard (majority) voting of the five base classifiers.
Base classifiers: Logistic Regression, K-Nearest Neighbours (k = 5), Decision Tree (max depth 5), SVM (RBF kernel) and Gaussian Naive Bayes.
 
## Dataset
 
- **Breast Cancer Wisconsin (Diagnostic)** – loaded directly from `sklearn.datasets.load_breast_cancer`, so no download is needed.
- 569 samples, 30 numeric features, 2 classes (212 malignant, 357 benign).
- Stratified split: 80% training (455 samples) and 20% test (114 samples), `random_state = 42`.
## Evaluation Metrics
 
Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, classification report, and 5-fold stratified cross-validation (mean and standard deviation). Benign is treated as the positive class.
 
## Project Structure
 
```
.
├── ensemble_learning.py   # complete source code
├── requirements.txt       # Python dependencies
├── README.md              # this file
├── results.csv            # generated: comparison of all models
├── fig_metrics.png        # generated: bar chart of metrics
├── fig_confusion.png      # generated: confusion matrices of the ensembles
├── fig_roc.png            # generated: ROC curves
└── screenshots/           # terminal output screenshots
```
 
## Requirements
 
- Python 3.9 or newer
- scikit-learn 1.2 or newer (needed for the `estimator=` parameter of `AdaBoostClassifier`)
- NumPy, pandas, Matplotlib
## How to Run
 
1. Clone the repository and open the folder:
```bash
git clone <your-repository-url>
cd AIML-Practical-5-Ensemble-Learning
```
 
2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
```
 
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
 
4. Run the program:
```bash
python ensemble_learning.py
```
 
The program prints the dataset details, the comparison of all models, the confusion matrices, the classification reports, the stacking meta-classifier coefficients and a single-sample prediction. It also saves `results.csv` and the three figures in the same folder.
 
## Results (test set, 114 samples)
 
- **Logistic Regression:** accuracy 0.9825, F1 0.9861
- **SVM (RBF):** accuracy 0.9825, F1 0.9861
- **Voting (hard):** accuracy 0.9737, F1 0.9793
- **Stacking (LR meta):** accuracy 0.9737, F1 0.9793, ROC-AUC 0.9917, lowest cross-validation standard deviation (0.0102)
- **AdaBoost (depth-3 trees):** accuracy 0.9649, F1 0.9726
- **AdaBoost (stumps):** accuracy 0.9561, F1 0.9660
- **KNN (k = 5):** accuracy 0.9561, F1 0.9655
- **Naive Bayes:** accuracy 0.9386, F1 0.9517
- **Decision Tree:** accuracy 0.9211, F1 0.9362
## Key Findings
 
- All ensembles clearly improved on the weak base models (Decision Tree and Naive Bayes).
- Stacking was the most stable across cross-validation folds.
- On this easy, almost linearly separable dataset, no ensemble beat the best single models (Logistic Regression and SVM). An ensemble is not automatically better than a strong single model.
- With only 114 test samples, one extra mistake changes accuracy by about 0.88 percentage points, so the differences among the top models are small.
## When to Use Each Method
 
- **Voting:** quick and simple when several good and different models already exist.
- **AdaBoost:** when the base learner is weak (stumps or shallow trees) and the data is reasonably clean.
- **Stacking:** when the base models are diverse and there is enough data for cross-validated training, and the extra complexity is acceptable.
## Future Improvements
 
- Hyperparameter tuning with grid or Bayesian search.
- Soft and weighted voting; bagging, Random Forest and gradient boosting (XGBoost, LightGBM).
- Testing on harder and imbalanced datasets.
- Repeated cross-validation with statistical tests (McNemar, paired t-test).
- Probability calibration and threshold tuning to reduce dangerous false positives.
- Explainability with SHAP.
## Author
 
Revati Bhosale – PRN 202401040048
