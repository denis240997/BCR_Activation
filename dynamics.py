import numpy as np
import importlib

import Membrane_dynamics_model
from settings import *


time_step = SIMULATION_TIME / TIME_STEPS

# membrane_lr_dynamics.get_image()
with open('param_log.txt', 'w', buffering=1) as file:
    for act in np.linspace(0.000005, 0.0005, 10):
        file.write(f'MELTING_ACTIVITY = {act}\n')
        print(f'MELTING_ACTIVITY = {act}')
        for diff in np.linspace(0.1, 1, 10):
            file.write(f'MELTING_DIFFUSION_RATE = {diff}\n')
            print(f'MELTING_DIFFUSION_RATE = {diff}')

            np.random.seed(1)

            membrane_lr_dynamics = Membrane_dynamics_model.MembraneDynamics(
                act,
                diff,

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

            for i in range(TIME_STEPS):  # TIME_STEPS
                if i % 50 == 0:
                    file.write(f'{i}\t{membrane_lr_dynamics.clustered_percent_and_mean_raft_radius()}\n')
                membrane_lr_dynamics.evolve_system()

# membrane_lr_dynamics.get_image()