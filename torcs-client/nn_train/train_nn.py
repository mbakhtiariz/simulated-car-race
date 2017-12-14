from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib

import numpy as np


def load_data(path):
    data = []
    f = open(path)
    c = 0
    for l in f:
        if c == 0:
            c += 1
            continue
        values = [float(x) for x in (l.strip().split(","))]
        data.append(values)
        c += 1
    return data


def write_csv(data, path):
    f = open(path, "w")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write(str(data[i, j]) + ",")
        f.write("\n")
    f.close()


def train_classifier(data_train, label_train, data_valid, label_valid, save_name):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(100, 100, 20, 2), random_state=1)
    clf.fit(data_train, label_train)
    predictions = np.array(clf.predict(data_valid))
    correct = np.sum(predictions == label_valid)
    acc = float(correct) / (data_valid.shape[0])
    joblib.dump(clf, save_name)
    print("acc: %f" % acc)


def train_regressor(data_train, label_train, data_valid, label_valid, save_name):
    reg = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100, 20, 2), random_state=1)
    reg.fit(data_train, label_train)
    predictions = np.array(reg.predict(data_valid))
    error = np.mean(np.abs(np.subtract(predictions, label_valid)))
    joblib.dump(reg, save_name)
    print("MSE: %f" % error)


def analyse_data(data, col):
    ranges = [-0.2, -0.05, -0.01, -0.001, 0.001, 0.01, 0.05, 0.2]
    hist = np.zeros((len(ranges)), dtype=int)
    for d in data[:, col]:
        start = -np.inf
        found = False
        for i in range(len(ranges)):
            if d > start and d <= ranges[i]:
                hist[i] += 1
                found = True
                break
        if not found:
            hist[len(ranges) - 1] += 1
    print(hist)


def discritize_data(data, thresholds):
    out = data.copy()
    for f in range(data.shape[1]):
        prev = -np.inf
        for ind, t in enumerate(thresholds[f]):
            (out[:, f])[np.where(np.logical_and(data[:, f] > prev, data[:, f] <= t))] = ind
            prev = t
    return out



dataset_names = ['aalborg', 'alpine-1', 'f-speedway']
path = "train_data/"
raw_data = []
for ds in dataset_names:
    raw_data += load_data(path + ds + ".csv")
data = np.array(raw_data)
thresholds = [[0.01, np.inf], [0.3, 0.7, np.inf], [-0.2, -0.05, -0.01, 0.01, 0.05, 0.2, np.inf]]
discrete_data = discritize_data(data[:, :3], thresholds)
data[:, :3] = discrete_data
np.random.shuffle(data)
end_train = int(data.shape[0] * 0.9)
train_classifier(data[:end_train, 3:], data[:end_train, 0], data[end_train:, 3:], data[end_train:, 0], "ann_acc.pkl")
train_classifier(data[:end_train, 3:], data[:end_train, 1], data[end_train:, 3:], data[end_train:, 1], "ann_brk.pkl")
train_classifier(data[:end_train, 3:], data[:end_train, 2], data[end_train:, 3:], data[end_train:, 2], "ann_str_2.pkl")

