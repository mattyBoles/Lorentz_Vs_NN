import numpy as np
import matplotlib.pyplot as plt
from config import config as c


# --- Config ---
n = c['n_rollout_steps']
t     = np.arange(0, n * c['h'], c['h'])


# --- Generate RK4 trajectories ---

def lorentz(x_,sigma,beta,rho):
    x,y,z = x_
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = (x * y) - (beta * z)

    return np.array([dxdt, dydt, dzdt])


def rk4(f, h, x_, sigma, beta, rho):
    k1 = f(x_, sigma, beta, rho)

    k2 = f(x_ + (k1 * h/2), sigma, beta, rho)

    k3 = f(x_ + (k2 * h/2), sigma, beta, rho)

    k4 = f(x_ + (k3*h), sigma, beta, rho)

    x_ = x_ + h*(k1 + 2*k2 + 2*k3 + k4)/6


    return x_
def generate_rk4(ic, n, h, sigma, beta, rho):
    traj = np.zeros((n, 3))
    traj[0] = ic
    for i in range(n - 1):
        traj[i+1] = rk4(lorentz, h, traj[i], sigma, beta, rho)
    return traj



traj1 = generate_rk4(c['ic1'], n, c['h'], c['sigma'], c['beta'], c['rho'])
traj2 = generate_rk4(c['ic2'], n, c['h'], c['sigma'], c['beta'], c['rho'])

print(traj1[:5,])
print(traj2[:5,])
# --- Phase portrait ---
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], linewidth=0.5, color='blue',   label='IC 1')
ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], linewidth=0.5, color='orange', label='IC 2')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.savefig('Lorentz_Phase_Potrait.png')
plt.close()

# --- x(t) sync plot ---
fig, ax = plt.subplots()
ax.plot(t, traj1[:,0], color='blue',   label='IC 1')
ax.plot(t, traj2[:,0], color='orange', label='IC 2')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend()
plt.savefig('Lorentz_Sync.png')
plt.close()

# --- Separation ---
separation = np.linalg.norm(traj2 - traj1, axis=1)
fig, ax = plt.subplots()
ax.plot(t, separation, color='black')
ax.set_xlabel('t')
ax.set_ylabel('separation')
plt.savefig('Lorentz_Seperation.png')
plt.close()