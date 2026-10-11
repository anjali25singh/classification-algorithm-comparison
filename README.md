# Classification Algorithms Comparison

A single script that trains and benchmarks 20+ classical machine learning classifiers on the same dataset, so you can compare their performance side by side.

## Algorithms Covered

| Category | Algorithms |
|---|---|
| Linear | Logistic Regression, Perceptron, Ridge Classifier, LDA, QDA, Linear SVM, SGD Classifier |
| Distance-Based | k-Nearest Neighbors, Radius Neighbors, Nearest Centroid |
| Tree-Based | Decision Tree, Extra Trees |
| Bagging | Random Forest, Bagging Classifier |
| Boosting | AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Margin-Based | SVM (RBF Kernel) |
| Probabilistic | Gaussian NB, Multinomial NB, Bernoulli NB |
| Neural Network | Multilayer Perceptron (MLP) |

> Deep learning models (CNN, RNN, LSTM, Transformers, GNN) and models such as Bayesian Networks, OneR, RIPPER, CRF and HMM are not included, as they need specialised libraries or data types.

## Project Structure

```
classification-algorithms-comparison/
├── classification_algorithms_comparison.py
├── requirements.txt
├── results.csv
└── README.md
```


`requirements.txt`:
```
scikit-learn
pandas
xgboost
lightgbm
catboost
```

The script will:
1. Load the Breast Cancer dataset (scikit-learn).
2. Split it 80/20 with stratification.
3. Scale features and train every model in a pipeline.
4. Print accuracy and F1 score for each model, sorted by accuracy.
5. Save the table to `results.csv`.

XGBoost, LightGBM and CatBoost are optional; if one is not installed, it is skipped automatically.

## Tech Stack

Python, scikit-learn, pandas, XGBoost, LightGBM, CatBoost

## License
MIT