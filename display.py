import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt

data = dn4.my_loadtxt("fake_data.txt")
posterior_sample = np.atleast_2d(dn4.my_loadtxt("posterior_sample.txt"))

x = np.linspace(-20.0, 20.0, 2001)

plt.figure(figsize=(13, 4))
for i in range(0, posterior_sample.shape[0]):
    plt.plot(x, posterior_sample[i, :], "g-", alpha=0.1)

plt.plot(data[:,0], data[:,1], "ko", markersize=5, alpha=0.1)
plt.show()

