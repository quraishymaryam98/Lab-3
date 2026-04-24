# 🌳 Lab 3: Classification Models II — Decision Trees and Random Forests

A hands-on machine learning lab exploring **Decision Trees** and **Random Forests** for multi-class classification using the classic Wine dataset from scikit-learn.

---

## 📋 Overview

This lab walks through the full ML pipeline — from data loading and exploration to model training, evaluation, hyperparameter tuning, and comparison — using tree-based classification models.

**Dataset:** [Wine Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) (178 samples × 13 features, 3 classes)  
**Task:** Classify wines into 3 categories based on chemical properties  
**Models Covered:** Decision Tree Classifier, Random Forest Classifier

---

## 🗂️ Notebook Structure

| Cell | Title | Description |
|------|-------|-------------|
| 1 | Import Libraries | Loads numpy, pandas, matplotlib, seaborn, and all sklearn modules |
| 2 | Load Wine Dataset | Loads dataset, creates a DataFrame, maps target labels |
| 3 | Explore Target Distribution | Bar chart and pie chart of class distribution |
| 4 | Prepare Data | Feature/target split; 80/20 stratified train-test split |
| 5 | Train Basic Decision Tree | Trains full-depth Decision Tree; computes train/test accuracy |
| 6 | Visualize Decision Tree | Full tree visualization using `plot_tree` |
| 7 | Simple Tree (Depth=3) | Trains a shallower tree; compares with full tree |
| 8 | Find Optimal Tree Depth | Iterates depths 1–15; plots overfitting gap |
| 9 | Train Random Forest | Trains Random Forest with 100 trees |
| 10 | Compare Models | Side-by-side bar chart of all model accuracies |
| 11 | Detailed Evaluation | Confusion matrices and classification reports |
| 12 | Feature Importance (RF) | Visualizes RF feature importances (bar + pie) |
| 13 | Feature Importance Comparison | Compares DT vs RF feature importance rankings |
| 14 | Hyperparameter Tuning | GridSearchCV over 108 parameter combinations |
| 15 | Evaluate Best Model | Tests the optimized RF on the test set |
| 16 | Final Comparison Summary | Complete model comparison table and visualization |
| 17 | Key Concepts Summary | Printed recap of all major concepts |

---

## 🧰 Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML models, metrics, and datasets |

---

## 🚀 How to Run

1. Clone or download this repository
2. Install the required dependencies (see above)
3. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook Lab-3_Classification_Models_II-Decision_Trees_and_Random_Forests.ipynb
   ```
4. Run all cells in order (Cell 1 → Cell 17)

> ⚠️ Run Cell 1 first — all subsequent cells depend on the imported libraries and `random_state=42` for reproducibility.

---

## 📊 Models & Results

### Decision Tree (Full, Unconstrained)
- Grows until all leaves are pure
- Typically achieves **100% training accuracy** — a sign of overfitting
- Test accuracy may be lower than a pruned tree

### Decision Tree (max_depth=3)
- Limits tree to 3 levels for better generalization
- More interpretable — can visualize every decision rule

### Random Forest (Default, 100 trees)
- Ensemble of 100 Decision Trees
- Uses **Bootstrap sampling** and **feature randomization**
- Significantly reduces overfitting compared to a single tree

### Random Forest (Optimized via GridSearchCV)
- Best parameters found from 108 combinations × 5-fold CV (540 total fits)
- Tuned hyperparameters:
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [3, 5, 10, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

---

## 🔑 Key Concepts

### 🌳 Decision Trees
- Recursively split data using **Gini impurity** or **Entropy**
- Fully interpretable — the entire model can be visualized
- Prone to overfitting without depth constraints (`max_depth`)

### 🌲 Random Forests
- An **ensemble** of many Decision Trees
- Combines **Bootstrap sampling** (Bagging) with **random feature selection**
- More robust and less prone to overfitting than single trees
- Provides built-in **feature importance** scores

### 📈 Feature Importance
- Measures each feature's contribution to reducing impurity across all splits
- Random Forest importances are more stable than single tree importances
- Useful for feature selection and model interpretability

### ⚙️ Key Hyperparameters
| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of trees — more trees = better stability, but slower |
| `max_depth` | Maximum tree depth — controls overfitting |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required in a leaf node |
| `criterion` | Split quality measure: `gini` or `entropy` |

---

## 📁 File

```
Lab-3_Classification_Models_II-Decision_Trees_and_Random_Forests.ipynb
```

