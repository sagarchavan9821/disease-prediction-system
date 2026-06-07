from preprocessing import x, y
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

disease_columns = [
    'heart_disease', 'diabetes', 'stroke', 'kidney_disease',
    'cancer', 'alzheimers_disease', 'copd', 'liver_disease',
    'parkinsons_disease', 'tuberculosis'
]

# ── Split & Scale ────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train one tuned SVM per disease ─────────────────────────────────────────
trained_models = {}

for disease in disease_columns:
    print(f"\n[{disease}] Training...")

    y_train_single = y_train[disease]
    y_test_single  = y_test[disease]

    classifier = GridSearchCV(
        SVC(),
        {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        cv=5,
        return_train_score=False,
        n_jobs=-1          # use all CPU cores → faster
    )
    classifier.fit(X_train, y_train_single)

    best_model = classifier.best_estimator_
    acc = accuracy_score(y_test_single, best_model.predict(X_test))

    print(f"  Best Params : {classifier.best_params_}")
    print(f"  Tuned SVM Accuracy: {acc:.2%}")

    trained_models[disease] = best_model

# ── Bundle scaler + all models and save ─────────────────────────────────────
bundle = {
    "scaler": scaler,
    "models": trained_models,          # dict  { disease_name: SVC }
    "disease_columns": disease_columns
}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n✅  model.pkl saved successfully!")
print(f"   Contains: scaler + {len(trained_models)} SVM models")



