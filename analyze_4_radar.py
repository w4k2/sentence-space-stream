import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tabulate import tabulate


# matplotlib.rcParams.update({'font.size': 16, "font.family" : "monospace"})

# CHUNKS x METRICS
scores_sentence_space_glove = np.load("results/scores_sentence_space_glove_imgfixed.npy")
# METHODS x CHUNKS x METRICS
scores_ref = np.load("results/scores_MiniLM_2c.npy")
# METHODS x CHUNKS x METRICS
scores = np.concatenate((scores_ref, scores_sentence_space_glove.reshape(1, 2730, 10)[:, :2727]), axis=0)

# Unique metrics
scores = scores[:, :, [1, 3, 4, 5, 6, 7, 9]]
mean_scores = np.mean(scores, axis=1)

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

metrics=["recall", "precision", "specificity", "f1_score", "gmean_1", "gmean_2", "bac"]

mean_scores = np.concatenate((mean_scores, mean_scores[:, :1]), axis=1)

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(metrics)+1)
plt.figure(figsize=(6, 6))
ax = plt.subplot(polar=True)

for method_id, method in enumerate(methods):
    # print(method)
    m = mean_scores[method_id]
    # s = std_drift_scores[method_id]
    plt.plot(label_loc, m, label=method, c=colors[method_id], lw=lws[method_id], ls=lss[method_id])
    plt.fill_between(label_loc, m, m*0, color=colors[method_id], lw=lws[method_id], ls=lss[method_id], alpha=.05)
    # plt.fill_between(label_loc, m-s, m+s, color=colors[method_id], alpha=0.2)

ax = plt.gca()
ax.spines['polar'].set_visible(False)
ax.spines['start'].set_visible(False)
ax.spines['end'].set_visible(False)
ax.spines['inner'].set_visible(False)

plt.ylim(0,1)

gpoints = np.linspace(0,1,6)
plt.gca().set_yticks(gpoints)
plt.legend(loc=(0.9, -0.1), frameon=False)

ax.grid(lw=0)
ax.set_xticks(label_loc[:-1])
ax.set_xticklabels([])

gc = {
    'c':'#999',
    'lw': 1,
    'ls': ':'
}
for loc, met in zip(label_loc[:-1], metrics):
    # print(loc,met)
    ax.plot([loc,loc],[0,1], **gc)
ax.plot(np.linspace(0,2*np.pi,100), np.zeros(100), **gc)
ax.plot(np.linspace(0,2*np.pi,100), np.ones(100), **gc)

for gpoint in gpoints:
    ax.plot(np.linspace(0,2*np.pi,100), 
        np.ones(100) * gpoint, **gc)
    
    
step = np.pi*1.9/(len(metrics)-1)
for llo, lla in zip(label_loc*step, metrics):
     a = np.rad2deg(llo+np.pi/2) if llo > np.pi else np.rad2deg(llo-np.pi/2)
    #  print(a)
     ax.text(llo, 1.05, lla, rotation=a, ha='center', va='center')

plt.tight_layout()
# plt.title("Mean metric values", fontsize=17, x=0.5, y=1.07)
plt.savefig("figures/4_radar.png", dpi=200)
plt.savefig('foo.png')