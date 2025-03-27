import os               # used to interact with operating system
import numpy as np      # provides support for numerical operations and array

from sklearn.datasets import load_breast_cancer         # breast cancer dataset
from sklearn.model_selection import train_test_split    # training and testing sets
from sklearn.preprocessing import StandardScaler        # removing mean and scaling to unit var
from sklearn.neighbors import KNeighborsClassifier      # implements K-Nearest Neightbors Classifier
from sklearn.metrics import accuracy_score, classification_report   # measures acurracy for prediction

# fixes joblib warning by setting logical core count
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

# loads the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# splits into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# random_stare=42 ensures reproducibility

# features scaling (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# initialize and train KNN model
k = 7  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# makes predictions
y_pred = knn.predict(X_test)

# evalueates the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_report(y_test, y_pred))
