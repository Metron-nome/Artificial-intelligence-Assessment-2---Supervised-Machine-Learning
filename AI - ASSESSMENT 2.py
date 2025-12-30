import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv(r"C:\Users\annca\Downloads\spotify_analysis_dataset.csv")

#Define Label
df['is_hit'] = (df['popularity'] >= 70).astype(int)

#Features
features = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features]
y = df['is_hit']

#Standardizations
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Data Split (80% Train / 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#LOGISTIC REGRESSION
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

#KNN CLASSIFIER (k=5) ---
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

#RESULTS (LOGISTIC REGRESSION)
plt.figure(figsize=(6, 4))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(range(len(y_pred_log)), y_pred_log, color='red', marker='x', label='Predicted')
plt.title('Logistic Regression: Predictions vs Actual')
acc_log = accuracy_score(y_test, y_pred_log)
plt.legend()
plt.savefig('logistic_plot.png')

#RESULTS (KNN)
plt.figure(figsize=(6, 4))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.5)
plt.scatter(range(len(y_pred_knn)), y_pred_knn, color='green', marker='x', label='Predicted')
plt.title('KNN Classifier: Predictions vs Actual')
acc_knn = accuracy_score(y_test, y_pred_knn)
plt.legend()
plt.savefig('knn_plot.png')

#RESULTS (COMPARISON)
plt.figure(figsize=(7, 5))
plt.bar(['Logistic Regression', 'KNN Classifier'], [acc_log, acc_knn], color=['red', 'green'])
plt.title('Model Comparison: Accuracy Score (Higher is Better)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('accuracy_vs_graph.png')

plt.show()