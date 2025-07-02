import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots


output_file = "/Users/valentinagenesini/Documents/thesis/ELICA/elica/tests/.test/test_simple_sampling.1.txt"

param_names = ["weight", "minuslogpost", "tau", "logA", "As", "minuslogprior", "minuslogprior__0", "chi2", "chi2__elica.EE_WLxWL"]

data = []
with open(output_file, "r") as f:
    for line in f:
        if not line.strip().startswith("#"): 
            data.append([float(x) for x in line.split()])
data = np.array(data)


tau = data[:, 2]  
logA = data[:, 3]  
As = data[:, 4]  

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(tau, label="tau", color="blue")
plt.xlabel("Iterazioni")
plt.ylabel("tau")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(logA, label="logA", color="green")
plt.xlabel("Iterazioni")
plt.ylabel("logA")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(As, label="As", color="red")
plt.xlabel("Iterazioni")
plt.ylabel("As")
plt.legend()

plt.tight_layout()
plt.show()

samples = MCSamples(samples=data[:, [2, 3, 4]], names=["tau", "logA", "As"], labels=["\\tau", "\\log(10^{10} A_s)", "A_s"])

g = plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True)


# plt.savefig("triangle_plot_tau_logA_As.png")
plt.show()