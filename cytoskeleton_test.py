import numpy as np
import matplotlib.pyplot as plt

from Membrane_dynamics_model import CytoskeletonRestriction


# Пример распространения и угасания сигнала таяния от движущегося источника
np.random.seed(0)

T = 30
n = 3
cr = CytoskeletonRestriction(diffusion_rate=0.01,
                             melting_activity=0.001,
                             tonic_signal_level=0.01,
                             bcr_number=100,
                             time_step=1,
                             epsilon=0.0001,
                             )

points = np.random.rand(n, 2)
# points = np.array([[0.1, 0.1]])
for i in range(T):
    cr.add_melting_sources(points, i)
    points += np.random.normal(0, 0.05, size=(n, 2))
    #     points += np.ones((n,2)) * 0.03
    points %= 1

N = 1000
grid = np.mgrid[0:1:1 / N, 0:1:1 / N].reshape(2, -1).T
coeffs = cr.get_diffusion_coeffs(grid, T)

plt.figure(figsize=(10, 10))
plt.imshow(coeffs.reshape(N, N), cmap='Greys', extent=[0, 1.4, 0, 1.4])
plt.show()
