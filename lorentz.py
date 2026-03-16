import numpy as np
import matplotlib.pyplot as plt
def lorentz(x_,sigma,beta,rho):
    x,y,z = x_
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = (x * y) - (beta * z)

    return np.array([dxdt, dydt, dzdt])

sigma = 10
rho = 28
beta = 8/3

h = 0.01

def rk4(f, h, x_, sigma, beta, rho):
    k1 = f(x_, sigma, beta, rho)

    k2 = f(x_ + (k1 * h/2), sigma, beta, rho)

    k3 = f(x_ + (k2 * h/2), sigma, beta, rho)

    k4 = f(x_ + (k3*h), sigma, beta, rho)

    x_ = x_ + h*(k1 + 2*k2 + 2*k3 + k4)/6


    return x_

# Cell 1: setup
n = 3000
h = 0.01  # make sure h is defined
x1, y1, z1 = [0]*n, [0]*n, [0]*n
x1[0], y1[0], z1[0] = 0.1, 1.0, 0.0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Cell 2: first trajectory
for i in range(n-1):
    x1[i+1], y1[i+1], z1[i+1] = rk4(lorentz, h, np.array([x1[i], y1[i], z1[i]]), sigma, beta, rho)
ax.plot(x1, y1, z1, linewidth=0.5, color='blue')

# Reset for second trajectory
x2, y2, z2 = [0]*n, [0]*n, [0]*n
x2[0], y2[0], z2[0] = 0.2, 1.0, 0.0

# Cell 3: second trajectory
for i in range(n-1):
    x2[i+1], y2[i+1], z2[i+1] = rk4(lorentz, h, np.array([x2[i], y2[i], z2[i]]), sigma, beta, rho)
ax.plot(x2, y2, z2, linewidth=0.5, color='orange')

plt.savefig('plt1.png')
plt.close('all')

x1 = np.array(x1)
x2 = np.array(x2)
t = np.arange(0,30,0.01)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(t,x1)
plt.plot(t,x2)
plt.savefig('plt2.png')
plt.close()
plt.plot(t, np.abs(x2-x1))
plt.savefig('plt3.png')