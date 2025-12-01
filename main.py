# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def displacement(t, m, k, c, x0=1.0, v0=0.0):
    """Return displacement x(t) for a damped harmonic oscillator.
    Parameters
    ----------
    t : array_like
        Time points.
    m, k, c : float
        Mass, stiffness, damping coefficient.
    x0, v0 : float, optional
        Initial displacement and velocity.
    """
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * m * omega_n)
    t = np.asarray(t)
    if np.isclose(zeta, 1.0):
        # Critical damping
        A = x0
        B = v0 + omega_n * x0
        return (A + B * t) * np.exp(-omega_n * t)
    elif zeta < 1.0:
        # Underdamped
        omega_d = omega_n * np.sqrt(1 - zeta ** 2)
        A = x0
        B = (v0 + zeta * omega_n * x0) / omega_d
        return np.exp(-zeta * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    else:
        # Overdamped
        s1 = -omega_n * (zeta - np.sqrt(zeta ** 2 - 1))
        s2 = -omega_n * (zeta + np.sqrt(zeta ** 2 - 1))
        # Solve for constants C1, C2 using initial conditions
        C2 = (v0 - s1 * x0) / (s2 - s1)
        C1 = x0 - C2
        return C1 * np.exp(s1 * t) + C2 * np.exp(s2 * t)

def plot_underdamped():
    m = 1.0
    k = 10.0
    c = 1.0
    omega_n = np.sqrt(k / m)
    T = 2 * np.pi / omega_n
    t = np.linspace(0, 5 * T, 1000)
    x = displacement(t, m, k, c)
    plt.figure()
    plt.plot(t, x, label='Underdamped')
    plt.title('Underdamped Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('underdamped_displacement.png')
    plt.close()

def plot_damping_sweep():
    m = 1.0
    k = 10.0
    c_vals = [0.5, 2 * np.sqrt(k * m), 5.0]
    labels = ['Underdamped (c=0.5)', 'Critical (c=2√(km))', 'Overdamped (c=5)']
    omega_n = np.sqrt(k / m)
    T = 2 * np.pi / omega_n
    t = np.linspace(0, 5 * T, 1000)
    plt.figure()
    for c, label in zip(c_vals, labels):
        x = displacement(t, m, k, c)
        plt.plot(t, x, label=label)
    plt.title('Damping Ratio Sweep')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('damping_sweep_displacement.png')
    plt.close()

if __name__ == "__main__":
    plot_underdamped()
    plot_damping_sweep()
    # Primary numeric answer: damping ratio for the underdamped case (c=1)
    m = 1.0
    k = 10.0
    c = 1.0
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * m * omega_n)
    print('Answer:', round(zeta, 5))

