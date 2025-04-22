# logistic_classifier_eval.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    classification_report
)

# Load dataset
df = pd.read_csv(r"C:\Users\shreyass krishna\Downloads\email-classification-system\dataset\combined_emails_with_natural_pii.csv")
print("Actual Columns:", df.columns.tolist())

text_column = "email"      # or whatever shows up
label_column = "type"


# Drop rows with missing text/label
#df = df[[text_column, label_column]].dropna()

# Plot class distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=label_column, order=df[label_column].value_counts().index)
plt.title("Class Distribution")
plt.ylabel("Email Count")
plt.xlabel("Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df[text_column], df[label_column], test_size=0.2, random_state=42, stratify=df[label_column]
)

# Vectorize email text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# F1 score (macro)
f1_macro = f1_score(y_test, y_pred, average='macro')
print("F1 Score (macro):", f1_macro)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# F1-score per class plot
f1_scores = {label: metrics['f1-score'] for label, metrics in report.items() if label in model.classes_}
plt.figure(figsize=(8, 5))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
plt.title("F1 Score per Class")
plt.ylabel("F1 Score")
plt.xlabel("Class")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
