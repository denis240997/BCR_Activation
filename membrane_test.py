import numpy as np

from settings import *
from Membrane_dynamics_model import MembraneDynamics, scale_radii_to_points


time_step = SIMULATION_TIME / TIME_STEPS
raft_radii = np.empty((TIME_STEPS // 10, 10))

# membrane_lr_dynamics.get_image()
for j in range(10):

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

    for i in range(TIME_STEPS):
        if i % 10 == 0:
            raft_radii[i // 10, j] = membrane_lr_dynamics.lipid_rafts_radii.mean()
        membrane_lr_dynamics.evolve_system()

print(raft_radii.mean(axis=1))
print(raft_radii.std(axis=1))
