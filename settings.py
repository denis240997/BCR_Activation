from math import pi
# Параметры модели

# Моделируемая доля мембраны
PATCH_COEFFICIENT = 0.008

# Modeling area
SURFACE_AREA = 160   # um^2

# Time
SIMULATION_TIME = 10    # seconds
TIME_STEPS = 1000

# BCR
BCR_NUMBER = 80000
BCR_RADIUS = 0.005    # um
BCR_RAFTS_RADIUS = 0.005    # um (СПОРНО!)
BCR_DIFFUSION_COEFFICIENT = 0.039 / 2    # um^2 / sec

# Antigen
ANTIGEN_VALENCY = 2
ANTIGEN_NUMBER = 10000
ANTIGEN_SPOT_DISTANCE = 0.01    # nm
assert ANTIGEN_VALENCY * ANTIGEN_NUMBER <= BCR_NUMBER    # для простоты не берем антигена больше чем нужно

# Lipid rafts
MEMBRANE_OCCUPIED = 0.3    # занимаемая рафтами доля мембраны
FREE_RAFTS_RADIUS_MIN = 0.002    # um
FREE_RAFTS_RADIUS_MAX = 0.1    # um
RAFT_DIFFUSION_COEFFICIENT = 0.039    # um^2 / sec    # того же порядка, что и для BCR???
# DEG_FREEDOM = 2    # chi square parameter (size distribution)
FREE_RAFTS_NUMBER = int(
    (MEMBRANE_OCCUPIED * SURFACE_AREA - BCR_NUMBER * pi * BCR_RAFTS_RADIUS**2) /
    (pi * ((FREE_RAFTS_RADIUS_MIN + FREE_RAFTS_RADIUS_MAX)/2)**2)
)
assert FREE_RAFTS_NUMBER >= 0
print(FREE_RAFTS_NUMBER)

# Cytoskeleton melting
MELTING_DIFFUSION_RATE = 0.5    # Подогнать
MELTING_ACTIVITY = 0.003   # Подогнать
TONIC_MELTING_LEVEL = 0.001    # Из отношения минимального медианного коэф-та диффузии к максимальному в статье
MELTING_SIGNAL_DEATH_EPSILON = 0.1    #################### ПОЧЕМУ ОТ ЭТОГО ПАРАМЕТРА ЗАВИСИТ ЦЕЛОСТНОСТЬ КЛАСТЕРОВ???

# BCR edge attraction
POTENTIAL_COEF = 0.08

# Activation
ACTIVATION_PROB = 0.01    # Lyn activity
DEACTIVATION_PROB = 0.5    # CD45 activity
INHIBITION_REDUCTION_FACTOR = 0.13    # CD45 in-raft tunneling ratio

# Animation
MELTING_FIELD_RENDERING = 0


############
# Patching
SURFACE_AREA *= PATCH_COEFFICIENT
BCR_NUMBER = int(BCR_NUMBER * PATCH_COEFFICIENT)
FREE_RAFTS_NUMBER = int(FREE_RAFTS_NUMBER * PATCH_COEFFICIENT)
ANTIGEN_NUMBER = int(ANTIGEN_NUMBER * PATCH_COEFFICIENT)