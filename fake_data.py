import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

rng.seed(0)

# Some inputs
N = 1000
x = 5*rng.randn(N)

# An output
f = x - np.cos(x) + rng.randn(N)
f[x > 2] -= 5.0

# Save to file
np.savetxt("fake_data.txt", np.vstack([x, f]).T, header="1 1")

# Plot the data
plt.figure(figsize=(13, 4))
plt.plot(x, f, "ko", markersize=5, alpha=0.1)
plt.show()

