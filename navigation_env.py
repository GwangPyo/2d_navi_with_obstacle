import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, distance)
# from Integrated_policy_learning.network import Network
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from collections import deque
import copy


class RewardWrapper(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> float:
        pass


class EasyReward(RewardWrapper):
    def __call__(self, pos, before_pos, goal_pos, reward):
        before_dist = np.linalg.norm(goal_pos - before_pos)
        cur_dist = np.linalg.norm(goal_pos - pos)
        return cur_dist - before_dist


class RerwardFunction(RewardWrapper):
    def __call__(self, reward):
        return reward








# Routing Optimization Avoiding Obstacle.

FPS = 5
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

# Drone's shape
DRONE_POLY = [
    (-11, +14), (-14, 0), (-14, -7),
    (+14, -7), (14, 0), (+11, +14)]

OBSTACLE_INIT_VEL = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1/np.sqrt(2), 1/np.sqrt(2)), (1/np.sqrt(2), -1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)),
                     (-1/np.sqrt(2), -1/np.sqrt(2))]

VIEWPORT_W = 600
VIEWPORT_H = 400

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


def normalize_position(x):
    y = np.copy(x)
    y[0] = x[0]/W
    y[1] = x[1]/H
    return y


def denormalize_position(x):
    y = np.copy(x)
    y[0] = x[0] * W
    y[1] = x[1] * H
    return y


y_positions = [0.2, 0.35, 0.5, 0.7]
y_positions = [y__ * H for y__ in copy.copy(y_positions)]
# Initial Position of Drone and Goal which of each chosen randomly among vertical ends.
DRONE_INIT_POS =[(int(np.random.randint(1, 14)), np.random.choice(y_positions))]
GOAL_POS = [ (14, 11), (11, 11)]
VERTICAL = 1
HORIZONTAL = 0

OBSTACLE_POSITIONS = [ [[0.08, 0.25], [0.65, 0.25], HORIZONTAL],
                      [[0.08, 0.4], [0.65,  0.4], HORIZONTAL],
                      [[0.08, 0.55], [0.65,  0.55], HORIZONTAL],
                      [[0.92, 0.25], [0.85, 0.25], HORIZONTAL],
                      [[0.92, 0.4], [0.85, 0.4], HORIZONTAL],
                      [[0.2,  0.9],  [0.2, 0.75], VERTICAL],
                      [[0.4,  0.9],  [0.4, 0.75], VERTICAL],
                      [[0.6,  0.9],  [0.6 ,0.75], VERTICAL],
                      [[0.8 , 0.9], [0.8,  0.75], VERTICAL]
                ]
OBSTACLE_POSITIONS = [[denormalize_position(x[0]), denormalize_position(x[1]), x[2]] for x in OBSTACLE_POSITIONS]
for x in OBSTACLE_POSITIONS:
    if x[2] ==HORIZONTAL:
        x[0] += 0.3
        x[0] -= 0.3


def rotation_4(z):
    x = z[0]
    y = z[1]
    rot = [[x, y], [-x, y], [-x, -y], [x, -y]]
    return rot


class MovingRange(object):
    def __init__(self, start, end, axis):
        assert start <= end
        self.start = start
        self.end = end
        self.axis = axis
        move_direction = np.zeros(2)
        move_direction[self.axis] = 1
        self.move_direction = move_direction

    def out_of_range(self, o):
        if o.position[self.axis] >= self.end:
            return -1
        elif o.position[self.axis] <= self.start:
            return 1
        else:
            return 0

    @classmethod
    def from_metadata(cls, meta):
        axis = meta[2]
        if meta[0][axis] > meta[1][axis]:
            start = meta[1][axis]
            end = meta[0][axis]
        else:
            start = meta[0][axis]
            end = meta[1][axis]
        return cls(start=start, end=end, axis=axis)


def to_rect(obstacle_pos):

    axis = obstacle_pos[2]
    if axis == HORIZONTAL:
        y_range = 0.6
        x_range = np.abs(obstacle_pos[0][axis] - obstacle_pos[1][axis]) + 0.6
        position = [(obstacle_pos[0][0] + obstacle_pos[1][0])/2, (obstacle_pos[0][1] + obstacle_pos[1][1])/2]
        poly = rotation_4([x_range/2, y_range/2])
    else:
        y_range = np.abs(obstacle_pos[0][axis] - obstacle_pos[1][axis]) + 0.6
        x_range = 0.6
        position = [(obstacle_pos[0][0] + obstacle_pos[1][0])/2, (obstacle_pos[0][1] + obstacle_pos[1][1])/2]
        poly = rotation_4([x_range/2, y_range/2])

    return position, poly


class LidarCallback(Box2D.b2.rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & 1) == 0:
            return 1
        self.p2 = point
        self.fraction = fraction
        return 0


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.drone == contact.fixtureA.body or self.env.drone == contact.fixtureB.body:
            # if the drone is collide to something, set game over true
            self.env.game_over = True
            # if the drone collide with the goal, success
            if self.env.goal == contact.fixtureA.body or self.env.goal == contact.fixtureB.body:
                self.env.achieve_goal = True


class NavigationEnvDefault(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    """
    dictionary representation of observation 
    it is useful handling dict space observation, 
    classifying local observation and global observation, 
    lazy evaluation of observation space; whenever we add or delete some observation information   
    """
    observation_meta_data = {
        "position":gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "goal_position":gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "lidar": gym.spaces.Box(low=0, high=1, shape=(16, )),
        "energy": gym.spaces.Box(low=0, high=1, shape=(1, )),
        "obstacle_speed": gym.spaces.Box(low=-1, high=1, shape=(len(OBSTACLE_POSITIONS), )),
        "obstacle_position": gym.spaces.Box(low=0, high=1, shape=(2 * len(OBSTACLE_POSITIONS), ))
    }

    # meta data keys. It is used to force order to get observation.
    # We may use ordered dict, but saving key with list is more economic
    # rather than create ordered dict whenever steps proceed
    observation_meta_data_keys = ["position", "goal_position", "lidar", "energy", "obstacle_speed", "obstacle_position"]

    def __init__(self, max_obs_range=3,  max_speed=5, initial_speed=2, **kwargs):
        super(EzPickle, self).__init__()
        self._ezpickle_args = ( )
        self._ezpickle_kwargs = {}
        self.np_random = 7777
        self.verbose = False
        self.scores = None
        self.episode_counter = 0

        self.seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, 0))
        self.moon = None
        self.drone = None
        self.obstacle = None
        self.disturbs = []
        self.walls = []
        self.obstacles = []
        self.goal = None
        self.obs_tracker = None
        self.obs_range_plt = None
        self.max_obs_range = max_obs_range
        self.prev_reward = None
        self.num_beams = 16
        self.lidar = None
        self.drawlist = None
        self.achieve_goal = False
        self.strike_by_obstacle = False
        self.dynamics = initial_speed
        self.energy = 1
        self.min_speed = 0.5
        self.max_speed = max_speed
        self.speed_table = None
        p1 = (0.75, 0.5)
        p2 = (W - 0.5, 0.5)
        self.sky_polys = [[p1, p2, (p2[0], H-0.5), (p1[0], H-0.5)]]
        self.position_intrinsic_reward = None
        self.reset()
        self.game_over = False
        self.prev_shaping = None

        # debug
        self.action = None
        self.obs_queue = deque(maxlen=10)

    @property
    def observation_space(self):
        size = 0
        for k in self.observation_meta_data:
            val = self.observation_meta_data[k]
            size += val.shape[0]
        return spaces.Box(-np.inf, np.inf, shape=(size, ), dtype=np.float32)

    @property
    def action_space(self):
        # Action is two floats [vertical speed, horizontal speed].
        return spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def seed(self, seed=7777):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def to_polar(x):
        r = np.linalg.norm(x)
        x_pos = x[0]
        y_pos = x[1]
        theta = np.arctan(y_pos/x_pos)
        return np.asarray([r, theta])

    def dict_observation(self):
        position = normalize_position(self.drone.position)
        goal_position = normalize_position(self.goal.position)
        lidar = [l.fraction for l in self.lidar]
        obstacle_speed = np.copy(self.speed_table)
        obstacle_position = [position - normalize_position(o.position) for o in self.obstacles]
        dict_obs = {
            "position":position,
            "goal_position": goal_position,
            "lidar": lidar,
            "energy": self.energy,
            "obstacle_speed": obstacle_speed,
            "obstacle_position":obstacle_position
        }

        return dict_obs

    def array_observation(self, dict_obs=None):
        if dict_obs is None:
            dict_obs = self.dict_observation()

        obs = []

        for k in self.observation_meta_data_keys:
            obs.append(np.asarray(dict_obs[k], dtype=np.float32).flatten())

        return np.concatenate(obs)

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.drone)
        self.drone = None
        self._clean_walls(True)
        self.world.DestroyBody(self.goal)
        self.goal = None
        self.world.DestroyBody(self.obs_range_plt)
        self.obs_range_plt = None
        self._clean_obstacles(True)

    def _observe_lidar(self, pos):
        for i in range(self.num_beams):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(i * 2 * np.pi / self.num_beams) * self.max_obs_range,
                pos[1] + math.cos(i * 2 * np.pi / self.num_beams) * self.max_obs_range)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

    def _build_wall(self):
        wall_pos =WALL_INFOS["pos"]
        wall_ver = WALL_INFOS["vertices"]

        for p, v in zip(wall_pos, wall_ver):
            wall = self.world.CreateStaticBody(
                position=p,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=v),
                    density=100.0,
                    friction=0.0,
                    categoryBits=0x001,
                    restitution=1.0,)  # 0.99 bouncy
            )
            wall.color1 = (1.0, 1.0, 1.0)
            wall.color2 = (1.0, 1.0, 1.0)
            self.walls.append(wall)

    def _build_obstacles(self):
        for i in range(len(OBSTACLE_POSITIONS)):
            pos = np.random.uniform(low=OBSTACLE_POSITIONS[i][0], high=OBSTACLE_POSITIONS[i][1])
            if self.max_speed < self.min_speed:
                vel = 0
            else:
                vel = np.random.uniform(low=self.min_speed, high=self.max_speed)
            coin = np.random.randint(low=0, high=2)
            if coin == 0:
                vel = vel * (-1)
            if OBSTACLE_POSITIONS[i][2] == HORIZONTAL:
                vel = [vel, 0]
            else:
                vel = [0, vel]
            obstacle = self.world.CreateDynamicBody(
                position=(pos[0], pos[1]),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=0.3, pos=(0, 0)),
                    density=5.0,
                    friction=0,
                    categoryBits=0x001,
                    maskBits=0x0010,
                    restitution=1.0,
                )  # 0.99 bouncy
            )
            obstacle.color1 = (0.7, 0.2, 0.2)
            obstacle.color2 = (0.7, 0.2, 0.2)

            obstacle.linearVelocity.Set(vel[0], vel[1])
            speed = np.linalg.norm(obstacle.linearVelocity)
            self.speed_table[i] = speed / 5
            if coin == 0:
                self.speed_table[i] *= -1
            range_= MovingRange.from_metadata(OBSTACLE_POSITIONS[i])
            setattr(obstacle, "moving_range", range_)
            self.obstacles.append(obstacle)

    def _clean_walls(self, all=True):
        """
        clean up objects
        :param all: legacy arg
        :return: None
        """
        while self.walls:
            self.world.DestroyBody(self.walls.pop(0))

    def _clean_obstacles(self, all=True):
        """
        clean up objects
        :param all: legacy arg
        :return: None
        """
        while self.obstacles:
            self.world.DestroyBody(self.obstacles.pop(0))

    def _get_observation(self, position):
        delta_angle = 2* np.pi/self.num_beams
        ranges = [self.world.raytrace(position,
                                      i * delta_angle,
                                      self.max_obs_range) for i in range(self.num_beams)]

        ranges = np.array(ranges)
        return ranges

    @property
    def last_score(self):
        print(len(self.scores))
        return np.mean(self.scores)

    def reset(self):
        if self.scores is None:
            self.scores = deque(maxlen=10000)
        else:
            if self.achieve_goal:
                self.scores.append(1)
            else:
                self.scores.append(0)

        self.game_over = False
        self.prev_shaping = None
        self.achieve_goal = False
        self.strike_by_obstacle = False
        self.speed_table = np.zeros(len(OBSTACLE_POSITIONS))
        # timer
        self.energy = 1
        # clean up objects in the Box 2D world
        self._destroy()
        # create lidar objects
        self.lidar = [LidarCallback() for _ in range(self.num_beams)]
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        # create new world
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        p1 = (1, 1)
        p2 = (W - 1, 1)
        self.moon.CreateEdgeFixture(
            vertices=[p1, p2],
            density=100,
            friction=0,
            restitution=1.0,
        )
        self._build_wall()
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # create obstacles
        self._build_obstacles()

        # create controller object
        while True:
            drone_pos = (int(np.random.randint(1, 10)), int(np.random.randint(1, 5)))
            self.drone = self.world.CreateDynamicBody(
                position=drone_pos,

                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                    density=5.0,
                    friction=0.1,
                    categoryBits=0x0010,
                    maskBits=0x003,  # collide all but obs range object
                    restitution=0.0)  # 0.99 bouncy
            )
            self.drone.color1 = (0.5, 0.4, 0.9)
            self.drone.color2 = (0.3, 0.3, 0.5)
            self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
            if self.game_over:
                self.world.DestroyBody(self.drone)
                self.game_over = False
            else:
                break
        # create goal
        np.random.seed(np.random.randint(low=0, high=100000))
        goal_pos = GOAL_POS[np.random.randint(len(GOAL_POS))]
        self.goal = self.world.CreateStaticBody(
            position=goal_pos,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x002,
                maskBits=0x0010,  # collide only with control device
                restitution=0.0)  # 0.99 bouncy
        )
        self.goal.color1 = (0., 0.5, 0)
        self.goal.color2 = (0., 0.5, 0)

        self.obs_range_plt = self.world.CreateKinematicBody(
            position=(self.drone.position[0], self.drone.position[1]),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=np.float64(self.max_obs_range), pos=(0, 0)),
                density=0,
                friction=0,
                categoryBits=0x0100,
                maskBits=0x000,  # collide with nothing
                restitution=0.3)
        )
        self.obs_range_plt.color1 = (0.2, 0.2, 0.4)
        self.obs_range_plt.color2 = (0.6, 0.6, 0.6)
        self.drawlist = [self.obs_range_plt, self.drone, self.goal] + self.walls + self.obstacles
        self._observe_lidar(drone_pos)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        return np.copy(self.array_observation())

    def step(self, action: np.iterable):
        action = np.asarray(action, dtype=np.float64)
        self.action = action
        self.drone.linearVelocity.Set(action[0], action[1])
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        for i in range(len(self.obstacles)):
            o = self.obstacles[i]
            moving_range = o.moving_range.out_of_range(o)
            if moving_range != 0:
                speed = np.random.uniform(low=self.min_speed, high=self.max_speed)
                next_velocity = (speed * moving_range) * o.moving_range.move_direction
                o.linearVelocity.Set(next_velocity[0], next_velocity[1])
                self.speed_table[i] = moving_range  * speed / 5

        self.energy -= 1e-3
        pos = np.array(self.drone.position)
        self._observe_lidar(pos)
        reward = 0
        done = self.game_over

        info = {}
        if self.energy <= 0:
            done = True

        if done:
            if self.achieve_goal:
                reward = 1
            else:
                reward = 0
            info = {
                'is_success':
                    self.achieve_goal,
                'energy':
                self.energy,
                'episode': {
                    "r":reward,
                    'l': (1 - self.energy) * 1000
                },
            }
        obs = np.copy(self.array_observation())
        self.obs_queue.append(obs)
        return obs, reward, done, info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        self.obs_range_plt.position.Set(self.drone.position[0], self.drone.position[1])

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        self.obs_range_plt.position.Set(self.drone.position[0], self.drone.position[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def init_position_set(self, x, y):
        self.drone.position.Set(x, y)


class NavigationEnvAcc(NavigationEnvDefault):
    observation_meta_data = {
        "position":gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "goal_position":gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "velocity": gym.spaces.Box(np.array([-3, -3]), np.array([3, 3]), dtype=np.float32),
        "lidar": gym.spaces.Box(low=0, high=1, shape=(16, )),
        "energy": gym.spaces.Box(low=0, high=1, shape=(1, )),
        "obstacle_speed": gym.spaces.Box(low=-1, high=1, shape=(len(OBSTACLE_POSITIONS), )),
        "obstacle_position": gym.spaces.Box(low=0, high=1, shape=(2 * len(OBSTACLE_POSITIONS), ))
    }
    observation_meta_data_keys = ["position", "goal_position", "velocity",
                                  "lidar", "energy", "obstacle_speed", "obstacle_position"]

    def __init__(self, max_obs_range=3,  max_speed=5, initial_speed=2, **kwargs):
        super().__init__(max_obs_range, max_speed, initial_speed, )

    def dict_observation(self):
        position = normalize_position(self.drone.position)
        goal_position = normalize_position(self.goal.position)
        lidar = [l.fraction for l in self.lidar]
        obstacle_speed = np.copy(self.speed_table)
        obstacle_position = [position - normalize_position(o.position) for o in self.obstacles]
        velocity = self.drone.linearVelocity
        dict_obs = {
            "position":position,
            "goal_position": goal_position,
            "lidar": lidar,
            "energy": self.energy,
            "obstacle_speed": obstacle_speed,
            "obstacle_position":obstacle_position,
            "velocity":velocity
        }

        return dict_obs

    def step(self, action: np.iterable):
        action = np.asarray(action, dtype=np.float64)
        before_pos = np.asarray(self.drone.position).copy()

        self.action = action
        # mass == 1 impulse == action
        self.drone.ApplyForce((action[0] * 5, action[1] * 5), self.drone.position, wake=True)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        v = self.drone.linearVelocity
        v = np.clip(v, a_min=[-3, -3], a_max=[3, 3])
        self.drone.linearVelocity.Set(v[0], v[1]) # clip velocity

        for i in range(len(self.obstacles)):
            o = self.obstacles[i]
            moving_range = o.moving_range.out_of_range(o)
            if moving_range != 0:
                speed = np.random.uniform(low=self.min_speed, high=self.max_speed)
                next_velocity = (speed * moving_range) * o.moving_range.move_direction
                o.linearVelocity.Set(next_velocity[0], next_velocity[1])
                self.speed_table[i] = moving_range  * speed / 5

        self.energy -= 1e-3
        pos = np.array(self.drone.position)
        self._observe_lidar(pos)
        reward = np.linalg.norm(self.goal.position - before_pos) - np.linalg.norm(self.goal.position - self.drone.position)

        done = self.game_over

        info = {}
        if self.energy <= 0:
            done = True

        if done:
            if self.achieve_goal:
                reward = 1
            else:
                reward = 0
            info = {
                'is_success':
                    self.achieve_goal,
                'energy':
                self.energy,
                'episode': {
                    "r":reward,
                    'l': (1 - self.energy) * 1000
                },
            }
        obs = np.copy(self.array_observation())
        self.obs_queue.append(obs)
        return obs, reward, done, info
