import matplotlib.pyplot as plt
import numpy as np

mde_5 = [12.86, 14.45, 15.24]
mde_10 = [16.06, 16.01, 18.33]
mde_15 = [16.67, 16.64, 17.95]
pd_5 = [8.76, 5.86, 3.75]
pd_10 = [14.75, 10.19, 6.34]
pd_15 = [16.28, 12.90, 8.11]

x = np.arange(3)
x_l = [128, 256, 512]
mde = [mde_5, mde_10, mde_15]
pd = [pd_5, pd_10, pd_15]
i = 0
fig, axes = plt.subplots(1,3)
plt.xticks(x, x_l)
K = [5, 10, 15]


axes[0].plot(x, mde_5, label="MDERank",color="royalblue", marker="o", markersize = 2)
axes[0].plot(x, pd_5, label="Phrase-Document",color="orange", marker="p", markersize = 2)
axes[1].plot(x, mde_10, label="MDERank",color="royalblue", marker="o", markersize = 2)
axes[1].plot(x, pd_10, label="Phrase-Document",color="orange", marker="p", markersize = 2)
axes[2].plot(x, mde_15, label="MDERank",color="royalblue", marker="o", markersize = 2)
axes[2].plot(x, pd_15, label="Phrase-Document",color="orange", marker="p", markersize = 2)

# axes[0].set_xticks(x_l)
# axes[1].set_xticks(x_l)
# axes[2].set_xticks(x_l)

axes[0].set_title("F1@5")
axes[1].set_title("F1@10")
axes[2].set_title("F1@15")

plt.subplots_adjust(left=0.125,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.35)
plt.legend(["MDERank", "Phrase-Document"], loc='upper left')
plt.show()

plt.savefig("sequence_length.png")