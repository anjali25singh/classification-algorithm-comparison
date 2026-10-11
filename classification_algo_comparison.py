import warnings
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ---------- Data ----------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- Models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Perceptron": Perceptron(),
    "Ridge Classifier": RidgeClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Linear SVM": LinearSVC(max_iter=5000),
    "SGD Classifier": SGDClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Radius Neighbors": RadiusNeighborsClassifier(radius=10.0),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(),
    "Gaussian NB": GaussianNB(),
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB(),
    "MLP": MLPClassifier(max_iter=1000, random_state=42),
    "Nearest Centroid": NearestCentroid(),
}


try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(eval_metric="logloss", verbosity=0)
except ImportError:
    print("xgboost not installed, skipping")
try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(verbose=-1)
except ImportError:
    print("lightgbm not installed, skipping")
try:
    from catboost import CatBoostClassifier
    models["CatBoost"] = CatBoostClassifier(verbose=0)
except ImportError:
    print("catboost not installed, skipping")

# ---------- Train & Evaluate ----------
results = []
for name, model in models.items():
    try:
        
        scaler = MinMaxScaler() if name == "Multinomial NB" else StandardScaler()
        pipe = make_pipeline(scaler, model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "F1": round(f1_score(y_test, pred), 4),
        })
    except Exception as e:
        print(f"{name} failed: {e}")

df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print(df.to_string())
df.to_csv("results.csv", index=False)