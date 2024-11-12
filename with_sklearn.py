# Importowanie potrzebnych bibliotek
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import numpy as np

# # Wczytanie zbioru danych Iris
# iris = load_iris()
# X = iris.data  # cechy
# y = iris.target  # etykiety klas

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
features = mnist.data.astype(int)
labels = mnist.target.astype(int)

X = features[:2000]
y = labels[:2000]


# Podział zbioru danych na zestawy treningowy i testowy w stosunku 60:40
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

# Redukcja wymiarów do 3 przy użyciu PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Tworzenie i trenowanie modelu KNN
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_pca, y_train)

# Dokonywanie predykcji na zestawie testowym
y_pred = knn.predict(X_test_pca)

# Obliczanie dokładności
accuracy = accuracy_score(y_test, y_pred)

# Wyświetlanie wyniku
print(f'{accuracy:.2f}')
