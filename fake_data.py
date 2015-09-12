from pylab import *

# Some inputs
N = 10000
x = randn(N)
y = randn(N)

# An output
f = x + 3*y + 0.1*randn(N)

savetxt('fake_data.txt', vstack([x, y, f]).T)

