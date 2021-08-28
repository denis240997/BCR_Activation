import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from settings import *
from Membrane_dynamics_model import MembraneDynamics, scale_radii_to_points


matplotlib.use("TkAgg")

######################
# Настройка анимации #
######################

side_length = np.sqrt(SURFACE_AREA)
time_step = SIMULATION_TIME / TIME_STEPS

np.random.seed(1)

membrane_lr_dynamics = MembraneDynamics(

    # Modeling area
    time_step=time_step,
    surface_area=SURFACE_AREA,

    # BCR
    bcr_number=BCR_NUMBER,
    bcr_radius=BCR_RADIUS,
    bcr_rafts_radius=BCR_RAFTS_RADIUS,
    bcr_diffusion_coef=BCR_DIFFUSION_COEFFICIENT,

    # Lipid rafts
    free_rafts_number=FREE_RAFTS_NUMBER,
    raft_diffusion_coef=RAFT_DIFFUSION_COEFFICIENT,
)

# set up figure and animation
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, side_length)
ax.set_ylim(0, side_length)
ax.set_xlabel('$\mu m$')
ax.set_ylabel('$\mu m$')

time_text = ax.text(0.02, 0.97, '', backgroundcolor='white', transform=ax.transAxes)
lipid_rafts = ax.scatter([], [], c='sandybrown', alpha=0.5)
bcrs = ax.scatter([], [], c='darkblue', alpha=0.8)

if MELTING_FIELD_RENDERING:
    N = 100
    grid = np.mgrid[0:side_length:side_length/N, 0:side_length:side_length/N].reshape(2, -1).T
    melting = ax.imshow(np.random.rand(N, N),
                        cmap='Blues',
                        vmin=TONIC_MELTING_LEVEL,
                        vmax=2,
                        extent=[0, side_length, 0, side_length],
                        )


def animate(i):
    """perform animation step"""
    membrane_lr_dynamics.evolve_system()
    output = []

    time_text.set_text(f'time = {membrane_lr_dynamics.time:.3f}')
    output.append(time_text)

    lipid_rafts_centers, lipid_rafts_radii = membrane_lr_dynamics.get_rafts()
    lipid_rafts.set_offsets(lipid_rafts_centers)
    lipid_rafts.set_sizes(scale_radii_to_points(lipid_rafts_radii, fig, ax).flatten())
    output.append(lipid_rafts)

    bcr_centers, bcr_radii, bcr_activated = membrane_lr_dynamics.get_bcrs()
    bcrs.set_offsets(bcr_centers)
    bcrs.set_sizes(scale_radii_to_points(bcr_radii, fig, ax).flatten())
    bcrs.set_color(np.array(['darkblue', 'red'])[bcr_activated.astype(int)])
    output.append(bcrs)

    if MELTING_FIELD_RENDERING:
        melting_field = membrane_lr_dynamics.cytoskeleton.get_diffusion_coeffs(
            grid,
            membrane_lr_dynamics.time,
        )
        melting.set_data(melting_field.reshape(N, N)[:, ::-1].T)
        output.append(melting)

    return output


ani = animation.FuncAnimation(fig, animate, frames=1, interval=1, blit=True)
plt.show()

