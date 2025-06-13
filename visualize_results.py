# visualize_results.py
import json
import matplotlib.pyplot as plt
import numpy as np


with open("results.json") as f:
    data = json.load(f)
mlp_accs = data["mlp_accs"]
knn_accs = data["knn_accs"]


def plot_fold_accuracies(mlp_accs, knn_accs):
    folds = np.arange(1, len(mlp_accs)+1)
    plt.figure(figsize=(8,5))
    plt.plot(folds, mlp_accs, marker='o', label='MLP')
    plt.plot(folds, knn_accs, marker='s', label='k-NN')
    plt.xticks(folds)
    plt.ylim(0,1)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_fold_accuracies(mlp_accs, knn_accs)
