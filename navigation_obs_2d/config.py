import numpy as np
import copy

from navigation_obs_2d.util import denormalize_position, normalize_position

# how many action do in 1 second
FPS = 5
# affects how fast-paced the game is, forces should be adjusted as well
SCALE = 30.0
# Drone's shape
DRONE_POLY = [
    (-11, +14), (-14, 0), (-14, -7),
    (+14, -7), (14, 0), (+11, +14)]
# obstacle initial velocity
OBSTACLE_INIT_VEL = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1/np.sqrt(2), 1/np.sqrt(2)), (1/np.sqrt(2), -1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)),
                     (-1/np.sqrt(2), -1/np.sqrt(2))]
# map size
VIEWPORT_W = 600
VIEWPORT_H = 600

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

# Shape of Walls
WALL_POLY = [
    (-50, +20), (50, 20),
    (-50, -20), (50, -20)
]


HORIZON_LONG = [(W, -0.3), (W, 0.3),
                  (-W, -0.3), (-W, 0.3)  ]
VERTICAL_LONG = [ (-0.3, H), (0.3, H),
                  (-0.3, -H), (0.3, -H)]

HORIZON_SHORT = [(W/3, -0.5), (W/3, 0.5),
                  (-W/3, -0.5), (-W/3, 0.5)  ]
                    # up         # right     # down    # left , left_one_third, right_one_third
WALL_INFOS = {"pos": [(int(W /2), int(H)), (int(W), int(H/2)), (int(W / 2), 0), (0, int(H/2))],
              "vertices": [HORIZON_LONG, VERTICAL_LONG, HORIZON_LONG, VERTICAL_LONG]
}

y_positions = [0.2, 0.35, 0.5, 0.7]
y_positions = [y__ * H for y__ in copy.copy(y_positions)]
# Initial Position of Drone and Goal which of each chosen randomly among vertical ends.
DRONE_INIT_POS =[(int(np.random.randint(1, 14)), np.random.choice(y_positions))]
GOAL_POS = [ (14, 11), (11, 11)]
VERTICAL = 1
HORIZONTAL = 0


OBSTACLE_POSITIONS = [[[0.08, 0.25], [0.65, 0.25], HORIZONTAL],
                      [[0.08, 0.4], [0.65,  0.4], HORIZONTAL],
                      [[0.08, 0.55], [0.65,  0.55], HORIZONTAL],
                      [[0.92, 0.25], [0.85, 0.25], HORIZONTAL],
                      [[0.92, 0.4], [0.85, 0.4], HORIZONTAL],
                      [[0.2,  0.9],  [0.2, 0.75], VERTICAL],
                      [[0.4,  0.9],  [0.4, 0.75], VERTICAL],
                      [[0.6,  0.9],  [0.6 ,0.75], VERTICAL],
                      [[0.8 , 0.9], [0.8,  0.75], VERTICAL]
                ]

OBSTACLE_POSITIONS = [[denormalize_position(x[0], W, H), denormalize_position(x[1], W, H), x[2]] for x in OBSTACLE_POSITIONS]
for x in OBSTACLE_POSITIONS:
    if x[2] ==HORIZONTAL:
        x[0] += 0.3
        x[0] -= 0.3