import numpy as np
import matplotlib.pyplot as plt
from time import sleep


scores = np.load("results/scores_glove_pca100_2c.npy")
wuj = np.load("results/scores_glove_kue.npy")

print(scores.shape)
print(wuj.shape)

metrics=("recall", "recall_score", "precision", "precision_score", "specificity", "f1_score", "geometric_mean_score_1", "geometric_mean_score_2", "bac", "balanced_accuracy_score")


raz= scores[3]
dwa= wuj[0]

for metric_id, metric in enumerate(metrics):

    plt.plot(raz[:, metric_id], label="wuj")
    plt.plot(dwa[:, metric_id], label="korekta")
    plt.legend()
    plt.title(metric)
    plt.savefig("wuj.png")
    sleep(1)
    plt.close()