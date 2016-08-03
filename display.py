import dnest4.classic as dn4
import numpy as np
import matplotlib.pyplot as plt

data = dn4.my_loadtxt("fake_data.txt")
posterior_sample = np.atleast_2d(dn4.my_loadtxt("posterior_sample.txt"))

x = np.linspace(-10.0, 10.0, 2001)

for i in range(0, posterior_sample.shape[0]):
    plt.plot(x, posterior_sample[i, :], "g-", alpha=0.1)
    plt.hold(True)

plt.plot(data[:,0], data[:,1], "k.", markersize=10, alpha=0.2)
plt.show()

