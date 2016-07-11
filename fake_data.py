from pylab import *

seed(0)

# Some inputs
N = 100
x = randn(N)

# An output
f = x - cos(x) + 0.1*(abs(x) + 0.2)*randn(N)
f[x > 2] -= 2.0
plot(x, f, "ko", markersize=10, alpha=0.2)
savetxt('fake_data.txt', vstack([x, f]).T, header="1 1")

show()


