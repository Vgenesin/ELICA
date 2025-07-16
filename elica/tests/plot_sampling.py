import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots
from scipy.stats import gaussian_kde

output_file = "/Users/valentinagenesini/Documents/thesis/ELICA/elica/tests/.test/test_simple_sampling.1.txt"
output_file2= "/Users/valentinagenesini/Documents/thesis/ELICA/elica/tests/.test/test_simple_sampling10000sims.1.txt"
# param_names = ["weight", "minuslogpost", "tau", "logA", "As", "minuslogprior", "minuslogprior__0", "chi2", "chi2__elica.EE_100x143"]
param_names = ["tau"]
data = []
with open(output_file, "r") as f:
    for line in f:
        if not line.strip().startswith("#"): 
            data.append([float(x) for x in line.split()])
data = np.array(data)

data2 = []
with open(output_file2, "r") as f:
    for line in f:
        if not line.strip().startswith("#"): 
            data2.append([float(x) for x in line.split()])
data2 = np.array(data2)

tau = data[:, 2]
tau10000sims= data2[:, 2]


# logA = data[:, 3]  
# As = data[:, 4]  

plt.figure(figsize=(12, 6))

# plt.subplot(3, 1, 1)
plt.plot(tau, color="blue", label="500 sims")
plt.xlabel("Iterazioni")
plt.ylabel("tau")
plt.legend()

plt.plot(tau10000sims, color="red", label="10000 sims")
plt.xlabel("Iterazioni")
plt.ylabel("tau")
plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(logA, label="logA", color="green")
# plt.xlabel("Iterazioni")
# plt.ylabel("logA")
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.plot(As, label="As", color="red")
# plt.xlabel("Iterazioni")
# plt.ylabel("As")
# plt.legend()

plt.tight_layout()
plt.show()

# samples = MCSamples(samples=data[:, [2, 3, 4]], names=["tau", "logA", "As"], labels=["\\tau", "\\log(10^{10} A_s)", "A_s"])
samples = MCSamples(samples=data[300:, [2]], names=["tau"], labels=["\\tau"])
samples2 = MCSamples(samples=data2[300:, [2]], names=["tau"], labels=["\\tau"])


chain = data[300:, 2] 

# Stima della densità con KDE
kde = gaussian_kde(chain)
x = np.linspace(np.min(chain), np.max(chain), 1000)
pdf = kde(x)
peak = x[np.argmax(pdf)]
print("Picco (moda):", peak)

# Media e deviazione standard
mean = np.mean(chain)
std = np.std(chain)
print("Media:", mean)
print("Deviazione standard:", std)

# Intervallo di confidenza al 68%
low, high = np.percentile(chain, [16, 84])
print("Intervallo 68%:", (low, high))
print(high - low)
print("--------------------------------------------------")


print("\n")
chain = data2[300:, 2] 

# Stima della densità con KDE
kde = gaussian_kde(chain)
x = np.linspace(np.min(chain), np.max(chain), 1000)
pdf = kde(x)
peak = x[np.argmax(pdf)]
print("Picco (moda):", peak)

# Media e deviazione standard
mean = np.mean(chain)
std = np.std(chain)
print("Media:", mean)
print("Deviazione standard:", std)

# Intervallo di confidenza al 68%
low, high = np.percentile(chain, [16, 84])
print("Intervallo 68%:", (low, high))
print(high - low)

g = plots.get_subplot_plotter()
g.triangle_plot(
    [samples, samples2],  # lista di catene
    filled=True,
    legend_labels=["500 sims", "10000 sims"],
    legend_loc="upper right",
    contour_colors=["blue", "red"]
)

plt.savefig('/Users/valentinagenesini/Desktop/test_simple_sampling_triangle_plot.png')
plt.show()