from pylab import *

seed(0)

# Some inputs
N = 10000
x = randn(N)
y = randn(N)

# An output
f = x - cos(x) + 0.1*(abs(x) + 0.2)*randn(N)
f[x > 2] -= 2.0
plot(x, f, "ko", markersize=1, alpha=0.2)
show()

savetxt('fake_data.txt', vstack([x, y, f]).T, header="2 1")

