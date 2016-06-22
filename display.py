import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt

data = dn4.my_loadtxt("clocks.txt")
posterior_sample = np.atleast_2d(dn4.my_loadtxt("posterior_sample.txt"))

x = np.arange(50, 250)

for i in range(0, posterior_sample.shape[0]):
    plt.plot(x, posterior_sample[i, :], "g-", alpha=0.1)
    plt.hold(True)

plt.plot(data[:,0], data[:,2], "ko", markersize=5)
plt.show()

