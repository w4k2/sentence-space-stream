import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

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
axis_titles = ["Extraction", "Testing", "Training", 'Accumulated time']

fig, ax = plt.subplots(2, 2, figsize=(12, 12*.618))
ax = ax.ravel()
for axis in range(3):
    for method_id, method in enumerate(methods):
        ax[axis].plot(times[method_id, 10:, axis], ls=lss[method_id], lw=lws[method_id], c=colors[method_id], label=method)

        ax[axis].set_xlabel("number of processed chunks")
        ax[axis].set_ylabel("log of computation time [s]")
        ax[axis].legend(frameon=False, loc="upper right", ncol=2)
        ax[axis].grid(ls=":", c=(0.7, 0.7, 0.7))
        ax[axis].set_title("%s" % axis_titles[axis])
        ax[axis].set_yscale('log')
        
        ax[axis].set_ylim(1e-3, 1e2)

ax[3].bar(methods, np.mean(times[:, 10:, 0], axis=1), color="black", label="extraction", width=.25)
ax[3].bar(methods, np.mean(times[:, 10:, 1], axis=1), color="red", bottom=np.mean(times[:, 10:, 0], axis=1), label="test", width=.25)
ax[3].bar(methods, np.mean(times[:, 10:, 2], axis=1), color="blue", bottom=np.mean(times[:, 10:, 0], axis=1)+np.mean(times[:, 10:, 1], axis=1), label="train", width=.25)

# ax[3].set_yscale('log')
ax[3].set_ylabel("computation time [s]")
# ax[3].set_ylim(1e-1, 1e1)
ax[3].legend(frameon=False)
ax[3].grid(ls=":", c=(0.7, 0.7, 0.7))
ax[3].set_title(axis_titles[-1])

for aa in ax.ravel():
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("figures/5_time.png", dpi=200)
plt.savefig('foo.png')