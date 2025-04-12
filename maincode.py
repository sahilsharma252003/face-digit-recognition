import numpy as np
import random
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

digit_train_data_path = 'data/digitdata/trainingimages'
digit_train_label_path = 'data/digitdata/traininglabels'
digit_val_data_path   = 'data/digitdata/validationimages'
digit_val_label_path  = 'data/digitdata/validationlabels'
digit_test_data_path  = 'data/digitdata/testimages'
digit_test_label_path = 'data/digitdata/testlabels'

face_train_data_path  = 'data/facedata/facedatatrain'
face_train_label_path = 'data/facedata/facedatatrainlabels'
face_val_data_path    = 'data/facedata/facedatavalidation'
face_val_label_path   = 'data/facedata/facedatavalidationlabels'
face_test_data_path   = 'data/facedata/facedatatest'
face_test_label_path  = 'data/facedata/facedatatestlabels'

def load_data(data_path, label_path, height, width):
    with open(data_path, 'r') as dfile:
        data_lines = dfile.readlines()
    with open(label_path, 'r') as lfile:
        labels = [int(line.strip()) for line in lfile]
    images = []
    line_idx = 0
    for _ in labels:
        img = data_lines[line_idx : line_idx + height]
        line_idx += height
        images.append([line.rstrip('\n') for line in img])
    return images, np.array(labels)

def extract_digit_features(images, height=28, width=28):
    features = []
    for img in images:
        row_feat = []
        for line in img:
            for char in line:
                row_feat.append(0 if char == ' ' else 1)
        features.append(row_feat)
    return np.array(features)

def extract_face_features(images, height=70, width=60, block_size=5, threshold_ratio=0.0):
    features = []
    block_area = block_size * block_size
    threshold = threshold_ratio * block_area
    for img in images:
        arr = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                c = img[i][j] if j < len(img[i]) else ' '
                arr[i, j] = 1 if c == '#' else 0
        grid_h = height // block_size
        grid_w = width // block_size
        row_feat = []
        for gh in range(grid_h):
            for gw in range(grid_w):
                block = arr[gh*block_size:(gh+1)*block_size, gw*block_size:(gw+1)*block_size]
                row_feat.append(1 if np.sum(block) > threshold else 0)
        features.append(row_feat)
    return np.array(features)

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}

    def train(self, X, y):
        start = time()
        n_samples, n_features = X.shape
        classes = np.unique(y)
        for c in classes:
            self.class_priors[c] = (np.sum(y == c) + self.alpha) / (n_samples + self.alpha * len(classes))
        self.feature_probs = {}
        for c in classes:
            idx_c = (y == c)
            count_c = np.sum(idx_c)
            feature_sum = np.sum(X[idx_c], axis=0)
            self.feature_probs[c] = (feature_sum + self.alpha) / (count_c + 2*self.alpha)
        end = time()
        return end - start

    def predict(self, X):
        eps = 1e-9
        classes = list(self.class_priors.keys())
        log_priors = {c: np.log(self.class_priors[c] + eps) for c in classes}
        preds = []
        for x in X:
            best_c = None
            best_score = -1e9
            for c in classes:
                p = self.feature_probs[c]
                log_lik = np.sum(x * np.log(p + eps) + (1 - x) * np.log(1 - p + eps))
                score = log_priors[c] + log_lik
                if score > best_score:
                    best_score = score
                    best_c = c
            preds.append(best_c)
        return np.array(preds)

class MultiClassPerceptron:
    def __init__(self, classes=None, max_iter=1000, learning_rate=0.01):
        self.classes = classes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = {}
        self.biases = {}

    def train(self, X, y):
        start = time()
        n_samples, n_features = X.shape
        if self.classes is None:
            self.classes = np.unique(y)
        for c in self.classes:
            self.weights[c] = np.zeros(n_features)
            self.biases[c] = 0.0
        for _ in range(self.max_iter):
            errors = 0
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for i in indices:
                xi = X[i]
                yi = y[i]
                for c in self.classes:
                    label = 1 if (yi == c) else -1
                    pred = np.dot(self.weights[c], xi) + self.biases[c]
                    if label * pred <= 0:
                        self.weights[c] += self.learning_rate * label * xi
                        self.biases[c] += self.learning_rate * label
                        errors += 1
            if errors == 0:
                break
        end = time()
        return end - start

    def predict(self, X):
        preds = []
        for xi in X:
            scores = {}
            for c in self.classes:
                scores[c] = np.dot(self.weights[c], xi) + self.biases[c]
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

digit_train_imgs, digit_train_lbls = load_data(digit_train_data_path, digit_train_label_path, 28, 28)
digit_val_imgs,   digit_val_lbls   = load_data(digit_val_data_path,   digit_val_label_path,   28, 28)
digit_test_imgs,  digit_test_lbls  = load_data(digit_test_data_path,  digit_test_label_path,  28, 28)

X_train_digits = extract_digit_features(digit_train_imgs)
X_val_digits   = extract_digit_features(digit_val_imgs)
X_test_digits  = extract_digit_features(digit_test_imgs)
y_train_digits = digit_train_lbls
y_val_digits   = digit_val_lbls
y_test_digits  = digit_test_lbls
X_trainval_digits = np.concatenate([X_train_digits, X_val_digits])
y_trainval_digits = np.concatenate([y_train_digits, y_val_digits])

face_train_imgs, face_train_lbls = load_data(face_train_data_path, face_train_label_path, 70, 60)
face_val_imgs,   face_val_lbls   = load_data(face_val_data_path,   face_val_label_path,   70, 60)
face_test_imgs,  face_test_lbls  = load_data(face_test_data_path,  face_test_label_path,  70, 60)

X_train_faces = extract_face_features(face_train_imgs)
X_val_faces   = extract_face_features(face_val_imgs)
X_test_faces  = extract_face_features(face_test_imgs)
y_train_faces = face_train_lbls
y_val_faces   = face_val_lbls
y_test_faces  = face_test_lbls
X_trainval_faces = np.concatenate([X_train_faces, X_val_faces])
y_trainval_faces = np.concatenate([y_train_faces, y_val_faces])

training_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
repeats = 2

def evaluate_classifier(clf_class, X_full, y_full, X_test, y_test, constructor_params={}, train_params={}, fractions=training_fractions, repeats=repeats):
    results = []
    n_full = len(X_full)
    for frac in fractions:
        subset_size = int(frac * n_full)
        accuracies = []
        times_ = []
        for _ in range(repeats):
            indices = random.sample(range(n_full), subset_size)
            X_sub = X_full[indices]
            y_sub = y_full[indices]
            clf = clf_class(**constructor_params)
            t0 = time()
            clf.train(X_sub, y_sub, **train_params)
            t1 = time()
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            accuracies.append(acc)
            times_.append(t1 - t0)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_time = np.mean(times_)
        results.append((frac, mean_acc, std_acc, mean_time))
    return results

nb_digits_res = evaluate_classifier(
    NaiveBayesClassifier,
    X_trainval_digits, y_trainval_digits,
    X_test_digits, y_test_digits,
    constructor_params={'alpha':1.0},
    train_params={}
)
perc_digits_res = evaluate_classifier(
    MultiClassPerceptron,
    X_trainval_digits, y_trainval_digits,
    X_test_digits, y_test_digits,
    constructor_params={'classes': np.unique(y_trainval_digits), 'max_iter': 500, 'learning_rate':0.01},
    train_params={}
)
nb_faces_res = evaluate_classifier(
    NaiveBayesClassifier,
    X_trainval_faces, y_trainval_faces,
    X_test_faces, y_test_faces,
    constructor_params={'alpha':1.0},
    train_params={}
)
perc_faces_res = evaluate_classifier(
    MultiClassPerceptron,
    X_trainval_faces, y_trainval_faces,
    X_test_faces, y_test_faces,
    constructor_params={'classes': np.unique(y_trainval_faces), 'max_iter': 500, 'learning_rate':0.01},
    train_params={}
)

def separate_results(res_list):
    fracs = [r[0] for r in res_list]
    accs = [r[1] for r in res_list]
    stds = [r[2] for r in res_list]
    times = [r[3] for r in res_list]
    return fracs, accs, stds, times

def print_results(title, results):
    print(title)
    print("Frac% | MeanAcc | StdAcc | MeanTrainTime(s)")
    for frac, mean_acc, std_acc, mean_time in results:
        print(f"{int(frac*100)}% | {mean_acc:.3f} | {std_acc:.3f} | {mean_time:.3f}")

print_results("Naive Bayes (Digits)", nb_digits_res)
print_results("Perceptron (Digits)", perc_digits_res)
print_results("Naive Bayes (Faces)", nb_faces_res)
print_results("Perceptron (Faces)", perc_faces_res)

fracs_nb_digits, acc_nb_digits, std_nb_digits, time_nb_digits = separate_results(nb_digits_res)
fracs_pc_digits, acc_pc_digits, std_pc_digits, time_pc_digits = separate_results(perc_digits_res)
fracs_nb_faces, acc_nb_faces, std_nb_faces, time_nb_faces = separate_results(nb_faces_res)
fracs_pc_faces, acc_pc_faces, std_pc_faces, time_pc_faces = separate_results(perc_faces_res)

x_nb_digits = [f*100 for f in fracs_nb_digits]
x_pc_digits = [f*100 for f in fracs_pc_digits]
x_nb_faces = [f*100 for f in fracs_nb_faces]
x_pc_faces = [f*100 for f in fracs_pc_faces]

err_nb_digits = [1.0 - a for a in acc_nb_digits]
err_pc_digits = [1.0 - a for a in acc_pc_digits]
err_nb_faces = [1.0 - a for a in acc_nb_faces]
err_pc_faces = [1.0 - a for a in acc_pc_faces]

plt.figure(figsize=(6,4))
plt.plot(x_nb_digits, time_nb_digits, marker='o', label='Naive Bayes (Digits)')
plt.plot(x_pc_digits, time_pc_digits, marker='o', label='Perceptron (Digits)')
plt.title('Digits: Training Time vs. Training Size')
plt.xlabel('Training Size (%)')
plt.ylabel('Training Time (s)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.errorbar(x_nb_digits, err_nb_digits, yerr=std_nb_digits, marker='o', capsize=5, label='Naive Bayes (Digits)')
plt.errorbar(x_pc_digits, err_pc_digits, yerr=std_pc_digits, marker='o', capsize=5, label='Perceptron (Digits)')
plt.title('Digits: Prediction Error vs. Training Size')
plt.xlabel('Training Size (%)')
plt.ylabel('Prediction Error')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(x_nb_faces, time_nb_faces, marker='o', label='Naive Bayes (Faces)')
plt.plot(x_pc_faces, time_pc_faces, marker='o', label='Perceptron (Faces)')
plt.title('Faces: Training Time vs. Training Size')
plt.xlabel('Training Size (%)')
plt.ylabel('Training Time (s)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.errorbar(x_nb_faces, err_nb_faces, yerr=std_nb_faces, marker='o', capsize=5, label='Naive Bayes (Faces)')
plt.errorbar(x_pc_faces, err_pc_faces, yerr=std_pc_faces, marker='o', capsize=5, label='Perceptron (Faces)')
plt.title('Faces: Prediction Error vs. Training Size')
plt.xlabel('Training Size (%)')
plt.ylabel('Prediction Error')
plt.grid(True)
plt.legend()
plt.show()
