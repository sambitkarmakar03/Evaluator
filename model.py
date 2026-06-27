import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

X = pd.read_csv(
    "/Users/sambit_03/Desktop/Student Response/X_features.csv"
)

y = pd.read_csv(
    "/Users/sambit_03/Desktop/Student Response/y_labels.csv"
)

y = y["grade_encoded"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)


sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[
        'contradictory',
        'correct',
        'irrelevant',
        'partially_correct_incomplete'
    ],
    yticklabels=[
        'contradictory',
        'correct',
        'irrelevant',
        'partially_correct_incomplete'
    ]
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 68% accuracy 



joblib.dump(
    model,
    "/Users/sambit_03/Desktop/Student Response/xgboost_grade_classifier.pkl"
)

print("Model saved successfully")