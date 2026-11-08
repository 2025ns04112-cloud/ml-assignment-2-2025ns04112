# Breast Cancer Diagnosis — Classifier Comparison (ML Assignment 2)

**Student:** Gopi  **Roll No:** 2025NS04112  **Programme:** M.Tech (DSE), BITS Pilani WILP

## a. Problem statement

Given 30 numeric measurements of cell nuclei taken from a digitised fine-needle aspirate (FNA) image of a breast mass, predict whether the tumour is **malignant (1)** or **benign (0)**. This is a **binary classification** problem. Five classical ML classifiers are trained on the same data split, compared on six evaluation metrics, and served through an interactive Streamlit app.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository, loaded via `sklearn.datasets.load_breast_cancer` |
| Instances | 569 (requirement: ≥ 500) |
| Features | 30 numeric, real-valued (requirement: ≥ 12) |
| Target | `diagnosis`: 1 = malignant (212, 37.3 %), 0 = benign (357, 62.7 %) |
| Missing values | None |

There are 10 base measurements of each nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each one is recorded three ways: the **mean**, the **standard error** and the **worst** (mean of the three largest values). That gives 30 features.

sklearn's original encoding (0 = malignant) was flipped so that **malignant is the positive class**. Precision, recall, F1 and MCC therefore describe how well each model detects cancer.

**Split:** stratified 75 % train / 25 % test (`random_state = 4112`, from the roll number) → 426 train rows, 143 test rows (53 malignant, 90 benign). The test rows are saved as `test_data.csv`.

**Preprocessing:** `StandardScaler` inside a `Pipeline` for the distance- and margin-based models (Logistic Regression, kNN, SVM). Decision Tree and Gaussian NB use the raw features. The scaler is fit on training data only, so nothing leaks from the test set.

## c. GitHub repository link

`https://github.com/2025ns04112-cloud/ml-assignment-2-2025ns04112.git`


```
ml-assignment-2/
├── app.py                 # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv          # 143-row hold-out test set (features + diagnosis)
└── model/
    ├── train_models.py    # trains all 5 models, writes metrics + pkl files
    ├── metrics.csv
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── support_vector_machine.pkl
```

**Run locally**
```bash
pip install -r requirements.txt
python model/train_models.py      # retrains and regenerates pkl files + test_data.csv
streamlit run app.py
```

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler + L2, C = 0.5, max_iter = 5000 |
| Decision Tree | max_depth = 5, min_samples_leaf = 4 |
| kNN | StandardScaler + k = 7, distance-weighted |
| Naive Bayes | GaussianNB (features are continuous) |
| Support Vector Machine | StandardScaler + RBF kernel, C = 2.0, gamma = "scale" |

AUC uses `predict_proba` for every model except SVM, which uses its `decision_function` margin.

### Comparison table (test set, n = 143)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9860 | 0.9914 | 1.0000 | 0.9623 | 0.9808 | 0.9702 |
| Decision Tree | 0.9231 | 0.9362 | 0.9375 | 0.8491 | 0.8911 | 0.8343 |
| kNN | 0.9580 | 0.9763 | 1.0000 | 0.8868 | 0.9400 | 0.9118 |
| Naive Bayes | 0.9161 | 0.9780 | 0.9184 | 0.8491 | 0.8824 | 0.8188 |
| Support Vector Machine | 0.9790 | 0.9904 | 0.9808 | 0.9623 | 0.9714 | 0.9550 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on every metric (Accuracy 0.986, MCC 0.970). It made only 2 errors, both missed malignant cases, and had zero false alarms. After scaling, the two classes are close to linearly separable in this 30-D space, so a regularised linear boundary generalises very well and is easy to interpret. |
| Decision Tree | Weakest AUC (0.936) and second-lowest MCC (0.834). It missed 8 of 53 malignant cases and raised 3 false alarms. Axis-aligned splits approximate the smooth class boundary only coarsely, and its probability outputs are step-like (leaf frequencies), which is why AUC drops. Even with depth limits it has higher variance than the other models. |
| kNN | Perfect precision (no false alarms), but recall was only 0.887 (6 missed malignant cases). Scaling is essential here. Malignant tumours are the minority class and more spread out, so some of them have mostly benign neighbours and are outvoted. |
| Naive Bayes | Lowest accuracy (0.916) and MCC (0.819). It missed 8 malignant cases and raised 4 false alarms. Its ranking ability is still good (AUC 0.978), so the model orders cases well but its hard 0.5 decision is poorly placed. The independence assumption is clearly violated: radius, perimeter and area are almost perfectly correlated, so the same evidence gets counted several times. |
| Support Vector Machine | A very close second (MCC 0.955, AUC 0.990). It matched Logistic Regression's recall (2 missed malignant cases) and added 1 false alarm. The RBF kernel can model non-linear boundaries, but this data does not need much non-linearity, so the extra flexibility brings no gain over the linear model. |
| **Overall winner for this dataset** | **Logistic Regression.** It has the highest Accuracy, AUC, F1 and MCC, ties for the best recall, and has perfect precision. SVM is a near-equal alternative. Recall matters most in a cancer-screening setting, because a missed malignant case is the costliest error, and both LR and SVM are strongest there. |

## Streamlit app features

- **CSV upload** for test data, plus a download button for the sample `test_data.csv` and an option to use the bundled file
- **Model selection dropdown** covering all 5 models
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** heat-map with a missed-malignant / false-alarm summary, **ROC curve**, and **classification report**
- **Compare tab**: all 5 models scored side by side on the uploaded data, with the best value per metric highlighted and a bar chart
- Download predictions as CSV. A file without the `diagnosis` column gets predictions only.
