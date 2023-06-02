import csv
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(filename):
    X = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            X.append(list(map(float, row[:-1])))
            y.append(int(row[-1]))
    return np.array(X), np.array(y)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

def cosine_similarity(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    cosine_similarity = dot_product / (norm_x1 * norm_x2)
    return cosine_similarity

class kNNClassifier:
    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        # Trenowanie klasyfikatora
        # Przykład: Zapisz dane treningowe wewnątrz klasyfikatora
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        # Predykcja na danych testowych
        y_pred = []
        for x_test in X_test:
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self.get_distance(x_train, x_test)
                distances.append((dist, self.y_train[i]))
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            classes = [neighbor[1] for neighbor in neighbors]
            y_pred.append(max(set(classes), key=classes.count))
        return np.array(y_pred)

    def get_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return manhattan_distance(x1, x2)
        elif self.distance_metric == 'chebyshev':
            return chebyshev_distance(x1, x2)
        elif self.distance_metric == 'cosine':
            return 1 - cosine_similarity(x1, x2)
        else:
            raise ValueError(f"Nieznana metryka odległości: {self.distance_metric}")

# Wczytanie danych
X, y = load_dataset('dataset2.csv')

# Przykładowe wartości parametru k
k_values = [1, 3, 5]

# Przykładowe wartości metryki odległości
distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']

for k in k_values:
    for distance_metric in distance_metrics:
        # Podział danych na zbiór treningowy i testowy w stosunku 80:20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Utworzenie i wytrenowanie klasyfikatora kNN
        classifier = kNNClassifier(k, distance_metric)
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # Obliczenie dokładności klasyfikacji
        accuracy = np.sum(y_pred == y_test) / len(y_test)

        # Wyświetlenie wyników
        print(f"Parametr k = {k}")
        print(f"Metryka odległości: {distance_metric}")
        print(f"Przewidziane klasy: {y_pred}")
        print(f"Dokładność klasyfikacji: {accuracy}")
        print("----------------------------------")
