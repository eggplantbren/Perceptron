from pylab import *

seed(0)

# Some inputs
N = 100
x = 3*randn(N)

# An output
f = x - cos(x) + (0.1 + 0.1*abs(x))*randn(N)
f[x > 2] -= 5.0
plot(x, f, "ko", markersize=3, alpha=0.2)
savetxt('fake_data.txt', vstack([x, f]).T, header="1 1")

show()


