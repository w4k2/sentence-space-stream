import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

# REPLICATION x METHODS x CHUNKS x (EXTRACTION, TEST, TRAIN)
time_ref = np.load("results/time_complexity_ref.npy")
time_ss = np.load("results/time_complexity_sent_space.npy")
times = np.concatenate((time_ref, time_ss), axis=1)
# METHODS x CHUNKS x (EXTRACTION, TEST, TRAIN)
times = np.mean(times, axis=0)
methods = [
            "HF",
            "CDS",
            "NIE",
            "KUE",
            "ROSE",
            "SSS"
        ]

colors = ['silver', 'darkorange', 'seagreen', 'darkorchid', 'dodgerblue', 'red']
lws = [1.5, 1.5, 1.5 ,1.5 ,1.5 ,1.5]
lss = ["-", "-", "-", "-", "-", "-"]
axis_titles = ["Extraction", "Prediction", "Training"]

fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax = ax.ravel()
for axis in range(3):
    for method_id, method in enumerate(methods):
        ax[axis].plot(times[method_id, 10:, axis], ls=lss[method_id], lw=lws[method_id], c=colors[method_id], label=method)

        ax[axis].set_xlabel("chunks")
        ax[axis].set_ylabel("time [s]")
        ax[axis].legend(frameon=False)
        ax[axis].grid(ls=":", c=(0.7, 0.7, 0.7))
        ax[axis].set_title("%s" % axis_titles[axis])

ax[3].bar(methods, np.mean(times[:, 10:, 0], axis=1), color="red", label="extraction")
ax[3].bar(methods, np.mean(times[:, 10:, 1], axis=1), color="blue", bottom=np.mean(times[:, 10:, 0], axis=1), label="test")
ax[3].bar(methods, np.mean(times[:, 10:, 2], axis=1), color="green", bottom=np.mean(times[:, 10:, 0], axis=1)+np.mean(times[:, 10:, 1], axis=1), label="train")

ax[3].set_ylabel("time [s]")
ax[3].legend(frameon=False)
ax[3].grid(ls=":", c=(0.7, 0.7, 0.7))

plt.tight_layout()
plt.savefig("figures/4_time.png", dpi=200)