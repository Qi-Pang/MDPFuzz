import pickle
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def preprocess(model='ACAS'):
    if model == 'ACAS':
        with open('./results/tsne_crash.pkl', 'rb') as handle:
            crashes = pickle.load(handle)
        with open('./results/tsne_nocrash.pkl', 'rb') as handle:
            noncrashes = pickle.load(handle)
        return_crash = []
        return_nocrash = []
        for i in range(len(crashes)):
            return_crash.append(np.hstack((crashes[i][0], crashes[i][1])).flatten())
        return_crash = np.array(return_crash)

        for i in range(len(noncrashes)):
            for j in range(len(noncrashes[i])):
                return_nocrash.append(np.hstack((noncrashes[i][j][0], noncrashes[i][j][1])).flatten())
        return_nocrash = np.array(return_nocrash)
    elif model == 'RLCARLA':
        with open('./results/tsne_crash.pkl') as handle:
            crashes = pickle.load(handle)
        print(crashes.shape)
    return return_crash, return_nocrash

def cluster(crashes, noncrashes):
    X = np.vstack((crashes, noncrashes))
    clustering = MeanShift(bandwidth=2, n_jobs=16).fit(X)
    return clustering

def load_cluster(path):
    with open(path, 'rb') as handle:
        clustering = pickle.load(handle)
    return clustering

if __name__ == '__main__':
    crashes, noncrashes = preprocess()
    Y_crashes = np.ones(crashes.shape[0])
    Y_noncrashes = np.zeros(noncrashes.shape[0])
    crashes_train = crashes[:10000]
    crashes_test = crashes[10000:]
    noncrashes_train = noncrashes[:9900]
    noncrashes_test = noncrashes[9900:]
    # Saving clusters
    # clustering = cluster(crashes_train, noncrashes_train)
    # with open('./results/culster.pkl', 'wb') as handle:
    #     pickle.dump(clustering, handle, protocol=pickle.HIGHEST_PROTOCOL)
    clustering = load_cluster('./results/culster.pkl')
    labels = np.zeros((np.max(clustering.labels_) + 1))
    for i in range(crashes_train.shape[0]):
        y = clustering.predict(crashes_train[i:i+1, :])
        labels[y[0]] += 1
    for i in range(noncrashes_train.shape[0]):
        y = clustering.predict(noncrashes_train[i:i+1, :])
        labels[y[0]] -= 1
    print(labels.shape)
    acc_crash = 0
    acc_nocrash = 0
    roc_y = []
    ground_truth = []
    for i in range(crashes_test.shape[0]):
        y = clustering.predict(crashes_test[i:i+1, :])
        if labels[y[0]] > 0:
            acc_crash += 1
        min_dis = np.inf
        for j in range(clustering.cluster_centers_.shape[0]):
            if labels[j] > 0:
                dis = np.linalg.norm(clustering.cluster_centers_[j] - crashes_test[i])
                if dis < min_dis:
                    min_dis = dis
        roc_y.append(min_dis)
        ground_truth.append(0)
    for i in range(noncrashes_test.shape[0]):
        y = clustering.predict(noncrashes_test[i:i+1, :])
        if labels[y[0]] < 0:
            acc_nocrash += 1
        min_dis = np.inf
        for j in range(clustering.cluster_centers_.shape[0]):
            if labels[j] > 0:
                dis = np.linalg.norm(clustering.cluster_centers_[j] - noncrashes_test[i])
                if dis < min_dis:
                    min_dis = dis
        roc_y.append(min_dis)
        ground_truth.append(1)
    print(acc_crash / crashes_test.shape[0], acc_nocrash / noncrashes_test.shape[0])
    print(ground_truth)

    fpr, tpr, thresholds = roc_curve(ground_truth, roc_y)
    score = roc_auc_score(ground_truth, roc_y)

    print(score)
    print(tpr)

    np.save('./results/fpr.txt', fpr)
    np.save('./results/tpr.txt', tpr)
