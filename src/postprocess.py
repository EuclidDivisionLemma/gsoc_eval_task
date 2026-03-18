import pandas
from matplotlib import pyplot as plt

pos_weights = list(pandas.read_csv("results/weights.csv").loc[:, "weight"])
fig, ax = plt.subplots()

all = pandas.read_csv("datasets/combined.csv")

hist_before_pt = ax.hist(all.loc[:, "pt"])
before_y = all.loc[:, "y"]
original_weights = list(all.loc[:, "weight"])

all.loc[:, "weight"] = pos_weights
all.to_csv("results/final.csv", index=False)
after_y = all.loc[:, "y"]

ax2 = ax.twinx()

hist_after_pt = ax2.hist(all.loc[:, "pt"])
fig.savefig("./results/overlaid_pt.jpeg")

plt.close()
fig, ax = plt.subplots()

hist_before_y = ax.hist(before_y)
ax1 = ax.twinx()
ax1.hist(after_y)

fig.savefig("./results/overlaid_y.jpeg")
