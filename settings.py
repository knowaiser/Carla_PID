'''
All the parameters used in the Simulation has been documented here.

Easily modifiable paramters with the quick access in this settings.py file \
    to achieve quick modifications especially during the training sessions.

Names of the parameters are self-explanatory therefore elimating the use of further comments.
'''


HOST = "localhost"
PORT = 2000
TIMEOUT = 20.0

CAR_NAME = 'model3'
EPISODE_LENGTH = 120
NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = True


RGB_CAMERA = 'sensor.camera.rgb'
SSC_CAMERA = 'sensor.camera.semantic_segmentation'

# Reward-related constants
# Change these later based on empirical experimentation
MAX_DEVIATION_ANGLE = 3.0
MAX_DEVIATION_DISTANCE = 6.0 
MAX_VELOCITY_THRESHOLD = 35.0
REWARD_CONSTANT_C = 1.0

# Constants from the paper or your specific application
K_d = 0.9  # Lateral deviation importance factor
K_h = 0.1  # Heading deviation importance factor
K_v = 0.2  # Velocity deviation importance factor


