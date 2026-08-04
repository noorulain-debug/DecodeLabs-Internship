from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. LOAD THE DATASET
iris = load_iris()

# 2. CONVERT TO A PANDAS DATAFRAME (a nice table view)
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target   # add the answer/label column

# 3. LOOK AT IT
print("First 5 rows:")
print(df.head())

print("\nShape (rows, columns):", df.shape)

print("\nSpecies distribution:")
print(df['species'].value_counts())

print("\nSpecies names (what 0, 1, 2 actually mean):")
print(iris.target_names)


# 4. SEPARATE FEATURES (X) FROM LABELS (y)
X = iris.data          # the 4 measurements (inputs)
y = iris.target        # the species (correct answers)

# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing, 80% for training
    random_state=42,    # ensures the same "random" split every time we run it
    stratify=y          # keeps species proportions balanced in both splits
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# 6. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn scale from training data, then apply
X_test_scaled = scaler.transform(X_test)          # apply the SAME scale to test data

print("\nFirst training row BEFORE scaling:", X_train[0])
print("First training row AFTER scaling:", X_train_scaled[0])

# 7. INSTANTIATE — build the model frame
model = KNeighborsClassifier(n_neighbors=5)

# 8. FIT — let the model memorize the training data
model.fit(X_train_scaled, y_train)

# 9. PREDICT — apply the model to unseen test data
predictions = model.predict(X_test_scaled)

print("\nActual species (test set):   ", y_test)
print("Predicted species (by model):", predictions)
# 10. ACCURACY
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.2%}")

# 11. CONFUSION MATRIX
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

# 12. FULL CLASSIFICATION REPORT (Precision, Recall, F1 per class)
report = classification_report(y_test, predictions, target_names=iris.target_names)
print("\nClassification Report:")
print(report)

#  ENHANCEMENT 1: Optimal K via Cross-Validation
X_scaled_full = scaler.fit_transform(X)   # scale the full dataset for CV

cv_mean_scores = []
k_range = range(1, 21)

for k in k_range:
    knn_test = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_test, X_scaled_full, y, cv=5)
    cv_mean_scores.append(scores.mean())

plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_mean_scores, marker='o', linestyle='dashed', color='green')
plt.title("Cross-Validated Accuracy vs K")
plt.xlabel("K Value")
plt.ylabel("Average CV Accuracy")
plt.xticks(k_range)
plt.grid(True)
plt.savefig("elbow_curve_cv.png")
plt.show()

best_k = k_range[cv_mean_scores.index(max(cv_mean_scores))]
print(f"\nBest K (via cross-validation): {best_k} (avg CV accuracy: {max(cv_mean_scores):.2%})")

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
final_knn_preds = final_knn.predict(X_test_scaled)
print(f"Final KNN (K={best_k}) Test Accuracy: {accuracy_score(y_test, final_knn_preds):.2%}")

# ENHANCEMENT 2: COMPARE A DIFFERENT ALGORITHM
# Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_scaled, y_train)
log_reg_preds = log_reg.predict(X_test_scaled)
log_reg_acc = accuracy_score(y_test, log_reg_preds)

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train_scaled, y_train)
tree_preds = tree.predict(X_test_scaled)
tree_acc = accuracy_score(y_test, tree_preds)

print("\n--- Algorithm Comparison ---")
print(f"KNN (K={best_k}) Accuracy:        {accuracy_score(y_test, final_knn_preds):.2%}")
print(f"Logistic Regression Accuracy: {log_reg_acc:.2%}")
print(f"Decision Tree Accuracy:       {tree_acc:.2%}")

# ENHANCEMENT 3: PREDICT A BRAND NEW FLOWER
new_flower = [[5.0, 3.4, 1.5, 0.2]]   # made-up measurements: sepal_len, sepal_wid, petal_len, petal_wid
new_flower_scaled = scaler.transform(new_flower)   # MUST use the same scaler, just transform (not fit)

new_prediction = final_knn.predict(new_flower_scaled)
predicted_species = iris.target_names[new_prediction[0]]

print(f"\nNew flower measurements: {new_flower[0]}")
print(f"Predicted species: {predicted_species}")

# ENHANCEMENT 4: VISUALIZE THE CONFUSION MATRIX
cm_final = confusion_matrix(y_test, final_knn_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title(f"Confusion Matrix (KNN, K={best_k})")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.savefig("confusion_matrix.png")
plt.show()

# ENHANCEMENT 5: CROSS-VALIDATION
# Scale the FULL dataset once for cross-validation (since CV handles its own splitting)
X_scaled_full = scaler.fit_transform(X)

cv_scores = cross_val_score(final_knn, X_scaled_full, y, cv=5)   # 5-fold cross-validation

print("\n--- Cross-Validation (5-fold) ---")
print("Scores for each fold:", cv_scores)
print(f"Average accuracy: {cv_scores.mean():.2%}")
print(f"Standard deviation: {cv_scores.std():.2%}")