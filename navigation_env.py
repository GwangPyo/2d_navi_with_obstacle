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

    def __init__(self, max_obs_range=3,  max_speed=5, initial_speed=2, tail_latency=5,
                 latency_accuracy = 0.95, obs_delay=3, **kwargs):
        super(EzPickle, self).__init__()
        self._ezpickle_args = ( )
        self._ezpickle_kwargs = {}
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
        self.tail_latency= tail_latency
        self.dynamics = initial_speed
        self.energy = 1
        self.latency_error = (1 - latency_accuracy)
        self.max_delay = obs_delay
        self.min_speed = 0.5
        self.max_speed = max_speed
        self.speed_table = None
        p1 = (0.75, 0.5)
        p2 = (W - 0.5, 0.5)
        self.sky_polys = [[p1, p2, (p2[0], H-0.5), (p1[0], H-0.5)]]
        self.position_intrinsic_reward = None
        self.reset()
        self.game_over = False

        # debug
        self.action = None
        self.obs_queue = deque(maxlen=10)

    def verbosing(self):
        if len(self.scores) == 100:
            print("last 100 episode score", np.mean(self.scores))

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

        for k in dict_obs.keys():
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

    def _clean_walls(self, all):
        while self.walls:
            self.world.DestroyBody(self.walls.pop(0))

    def _clean_obstacles(self, all):
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

        if self.verbose:
            self.verbosing()

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
        np.random.seed(np.random.random_integers(low=0, high=100000))
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
        if done:
            if self.achieve_goal:
                reward = 1
        info = {}
        if self.energy <= 0:
            done = True
            info = {
                'is_success':
                    self.achieve_goal,
                'energy':
                self.energy,
                'episode': {
                    "r": reward,
                    'l': (1 - self.energy) * 1000
                },
            }
        if done and not self.achieve_goal:

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


class NavigationEnvEasy(NavigationEnvDefault):

    @staticmethod
    def distance(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.linalg.norm(x - y)

    def step(self, action: np.iterable):
        goal_position = [self.goal.position[0], self.goal.position[1]]
        before_pos = [self.drone.position[0], self.drone.position[1]]
        before_distance =  self.distance(goal_position, before_pos)

        obs, reward, done, info = super().step(action)

        after_pos = [self.drone.position[0], self.drone.position[1]]
        after_distance = self.distance(goal_position, after_pos)
        reward = before_distance - after_distance

        info['episode']['r'] = reward
        return obs, reward, done, info


class NavigationEnvLocal(NavigationEnvDefault):
    observation_meta_data = {
        "position":gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "goal_position":gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "lidar": gym.spaces.Box(low=0, high=1, shape=(16, )),
        "energy": gym.spaces.Box(low=0, high=1, shape=(1, ))
    }

    # meta data keys. It is used to force order to get observation.
    # We may use ordered dict, but saving key with list is more economic
    # rather than create ordered dict whenever steps proceed
    observation_meta_data_keys = ["position", "goal_position", "lidar", "energy"]

    def dict_observation(self):
        position = normalize_position(self.drone.position)
        goal_position = normalize_position(self.goal.position)
        lidar = [l.fraction for l in self.lidar]
        dict_obs = {
            "position":position,
            "goal_position": goal_position,
            "lidar": lidar,
            "energy": self.energy
        }

        return dict_obs

    def array_observation(self, dict_obs=None):
        if dict_obs is None:
            dict_obs = self.dict_observation()
        obs = []
        for k in NavigationEnvLocal.observation_meta_data_keys:
            obs.append(np.asarray(dict_obs[k], dtype=np.float32).flatten())
        return np.concatenate(obs)


class NavigationEnvLocalEasy(NavigationEnvLocal):
    def step(self, action: np.iterable):
        # goal_pos = np.asarray(self.goal.position)
        # pos_before = np.asarray(self.drone.position).copy()
        obs, reward, done, info = NavigationEnvLocal.step(self, action)
        # pos_now = np.asarray(self.drone.position).copy()
        # reward = np.linalg.norm(pos_before - goal_pos) - np.linalg.norm(pos_now - goal_pos)
        # reward *= 100
        if self.achieve_goal:
            # reward *= 1
            print("success!! succ" , np.mean(self.scores))
        return obs, reward, done, info


class NavigationEnvWall(NavigationEnvDefault):
    observation_meta_data = {
        "position":gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "goal_position":gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "lidar": gym.spaces.Box(low=0, high=1, shape=(16, )),
        "energy": gym.spaces.Box(low=0, high=1, shape=(1, )),
    }

    observation_meta_data_keys = ["position", "goal_position", "lidar", "energy"]
    a_map = [[25, 0], [-25, 0], [0, 25], [0, -25]]

    subgoals = [denormalize_position([0.75, 0.25]), denormalize_position([0.75, 0.65]), "goal_pos"]

    def __init__(self, *args, **kwargs):


        super().__init__(*args, **kwargs)
        self.subgoal_index = 0

        # self.position_intrinsic_reward = PositionIntrinsicReward()

    def _build_obstacles(self):
        for i in range(len(OBSTACLE_POSITIONS)):
            pos, poly = to_rect(OBSTACLE_POSITIONS[i])
            wall = self.world.CreateStaticBody(
                position=pos,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=poly),
                    density=100.0,
                    friction=0.0,
                    categoryBits=0x001,
                    restitution=1.0, )  # 0.99 bouncy
            )
            wall.color1 = (1.0, 1.0, 1.0)
            wall.color2 = (1.0, 1.0, 1.0)
            self.obstacles.append(wall)

    def reset(self):
        super().reset()
        self.subgoal_index = 0
        return np.copy(self.array_observation())

    def dict_observation(self):
        position = normalize_position(self.drone.position)
        goal_position = normalize_position(self.goal.position)
        lidar = [l.fraction for l in self.lidar]
        dict_obs = {
            "position":position,
            "goal_position": goal_position,
            "lidar":lidar,
            "energy":self.energy
        }
        return dict_obs

    def array_observation(self, dict_obs=None):
        if dict_obs is None:
            dict_obs = self.dict_observation()
        obs = []
        for k in self.observation_meta_data_keys:
            obs.append(np.asarray(dict_obs[k], dtype=np.float32).flatten())
        obs = np.concatenate(obs)
        return obs

    def reward_vector(self, position):

        position = normalize_position(position)
        goal_pos = normalize_position(self.goal.position)
        """
        x = np.concatenate([position, goal_pos])
        x = np.expand_dims(x, axis=0)
        r_vec = self.r_model.predict_on_batch(x)
        return r_vec
        """
        x_pos = position[0]
        y_pos = position[1]
        goal_pos = normalize_position(self.goal.position)

        # below
        if y_pos <= 0.4:
            if x_pos <= 0.65:
                return np.asarray([1, 0])
            elif x_pos > 0.85:
                return np.asarray([-1, 0])
            else:
                return np.asarray([0, 1])
        # middle
        elif y_pos < 0.55:
            if x_pos < 0.65:
                return np.asarray([1, 0])
            else:
                return np.asarray([0, 1])
        # high
        elif y_pos < 0.68:
            x_dir = goal_pos[0] - x_pos
            if abs(x_dir) < 0.12:
                direction = goal_pos - position
                direction = direction / np.linalg.norm(direction) * 10
                return direction
            elif x_dir > 0:
                return np.asarray([1, 0])
            else:
                return np.asarray([-1, 0])

        else:
            if goal_pos[0] <= 0.2:
                if x_pos <= 0.2:
                    direction = goal_pos - position
                    direction = direction / np.linalg.norm(direction) * 10
                    return direction
                else:
                    return np.asarray([0, -1])

            elif 0.2 < goal_pos[0] < 0.4:
                if 0.2 < x_pos < 0.4:
                    direction = goal_pos - position
                    direction = direction / np.linalg.norm(direction) * 10
                    return direction
                else:
                    return np.asarray([0, -1])

            elif 0.4 < goal_pos[0] < 0.6:
                if 0.4 < x_pos < 0.6:
                    direction = goal_pos - position
                    direction = direction / np.linalg.norm(direction) * 10
                    return direction
                else:
                    return np.asarray([0, -1])

            elif 0.6 < goal_pos[0] < 0.8:
                if 0.6 < x_pos < 0.8:
                    direction = goal_pos - position
                    direction = direction / np.linalg.norm(direction) * 10
                    return direction
                else:
                    return np.asarray([0, -1])
            else:
                if 0.8 < x_pos :
                    direction = goal_pos - position
                    direction = direction / np.linalg.norm(direction) * 10
                    return direction
                else:
                    return np.asarray([0, -1])

    @staticmethod
    def achieve_subgoal(pos, subgoal_pos):
        if np.linalg.norm(pos - subgoal_pos) < 0.5:
            return True
        else:
            return False

    def lidar_info(self, pos):
        self._observe_lidar(pos)
        return [l.fraction for l in self.lidar]

    @property
    def current_subgoals(self):
        return [self.subgoals[0], self.subgoals[1], np.asarray(self.goal.position)]

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        # reward_vector = self.reward_vector(pos)
        # reward = np.inner(reward_vector, action)
        # action = np.asarray(action, dtype=np.float64)
        self.drone.linearVelocity.Set(action[0], action[1])
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self._observe_lidar(self.drone.position)
        """
        subgoals = self.current_subgoals
        subgoal_pos = subgoals[self.subgoal_index]
        current_pos = self.drone.position

        reward = 0.25 - 0.25 * np.linalg.norm(subgoal_pos - current_pos)
        if self.achieve_subgoal(current_pos, subgoal_pos):
            reward = 50
            if self.subgoal_index < 2:
                self.subgoal_index += 1
                self.subgoal_pos = subgoals[self.subgoal_index]
        """
        reward = 0
        self.energy -= 1e-3
        done = self.game_over
        if self.energy <= 0:
            done = True

        if done and self.achieve_goal:
            reward = 1
            # print("achieved goal")
        elif done and not self.achieve_goal:
            reward = -1000

        info = {
            'success':
                self.achieve_goal,
            'energy':
            self.energy
        }
        return np.copy(self.array_observation()), reward, done, info

    def pos_step(self, pos):
        self.drone.position.Set(pos[0], pos[1])
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.energy -= 1e-3
        self._observe_lidar(pos)
        return np.copy(self.array_observation())


class NavigationEnvMeta(gym.Env, gym.utils.EzPickle):
    observation_meta_data = copy.copy(NavigationEnvDefault.observation_meta_data)
    observation_meta_data["staleness"] = gym.spaces.Box(low=0, high=1, shape=(1, ))
    observation_meta_data["network_state"] = gym.spaces.Box(low=0, high=1, shape=(1, ))
    observation_meta_data_keys = copy.copy(NavigationEnvDefault.observation_meta_data_keys)
    observation_meta_data_keys.append("network_state")
    local_observation_keys = ["position", "goal_position", "lidar", "energy"]
    global_observation_keys = ["position", "obstacle_speed", "obstacle_position",  "obstacle_speed",
                               "obstacle_position", "staleness", "network_state"]
    device_global_observation_keys = ["obstacle_speed", "obstacle_position"]
    delay_meta_data = {
        "exponential": lambda avg: np.random.exponential(scale=avg),
        "normal": lambda avg: np.random.normal(loc=avg, scale=1),
        "lognormal":lambda avg, sigma: np.random.lognormal(avg, sigma)
    }

    def __init__(self, subpolicies, network_accuracy=1, average_delay=10, global_data_decay=0, delay_function=None,
                 correction_model=None, delay_kwargs=None,
                 *args, **kwargs):
        self.state = None
        self.dummy_obs = np.zeros((28, 21))
        super(EzPickle, self).__init__()
        self.wrapped_env = NavigationEnvDefault(*args, **kwargs)
        self.aux_env = NavigationEnvWall(*args, **kwargs)
        self.subpolicies = subpolicies

        self._built_subpolicy = False

        def init_subpolicy(self):
            # to avoid thread pickle problem.
            from stable_baselines import PPO2
            for i in range(len(self.subpolicies)):
                if type(self.subpolicies[i]) is str:
                    self.subpolicies[i] = PPO2.load(self.subpolicies[i])
            return None

        self.subpolicy_initializer = lambda: init_subpolicy(self)

        """
         network error = 1  <=> no net 
        """
        self.network_error = 1 - network_accuracy
        self.average_delay = average_delay
        self.global_data_decay = global_data_decay
        self.correction_model = correction_model
        self.delay = 0
        self.observation_stack = None
        self.current_subpolicy = 0
        self.action_histogram = None
        self.delay_function = delay_function(**delay_kwargs)
        self.step_cnt = 0
        self.stale_global_obs = None
        self.stale_local_obs = None

    @property
    def action_space(self):
        return gym.spaces.Discrete(n=len(self.subpolicies))

    @property
    def achieve_goal(self):
        return self.wrapped_env.achieve_goal

    @property
    def last_score(self):
        return self.wrapped_env.last_score

    @staticmethod
    def array_observation(dict_obs: dict, obs_key: list):
        obs = []
        for k in obs_key:
            obs.append(np.asarray(dict_obs[k], dtype=np.float32).flatten())
        obs = np.concatenate(obs)
        return obs

    @property
    def server_observation(self):
        fresh_obs = self.wrapped_env.dict_observation()
        global_obs = self.array_observation(fresh_obs, self.device_global_observation_keys)
        delta = global_obs - self.stale_global_obs
        network_state = [self.step_cnt/self.delay, self.delay]
        return np.copy(np.concatenate([fresh_obs["position"],  global_obs, delta, np.asarray(network_state)]))

    @property
    def local_only_device_observation(self):
        fresh_obs = self.wrapped_env.dict_observation()
        return np.copy(self.array_observation(fresh_obs, self.local_observation_keys))

    @property
    def full_global_device_observation(self):
        fresh_obs = self.wrapped_env.dict_observation()
        local_obs = self.array_observation(fresh_obs, self.local_observation_keys)
        return np.copy(np.concatenate((local_obs, self.stale_global_obs)))

    def reset(self):
        if self._built_subpolicy:
            pass
        else:
            self.subpolicy_initializer()
        self.wrapped_env.reset()
        self.network_state()
        self.step_cnt = 0
        fresh_obs = self.wrapped_env.dict_observation()
        fresh_obs["network_state"] = self.delay
        if 1 > self.network_error > 0:
            fresh_obs["network_state"] += np.random.normal(0, self.network_error)
        elif self.network_error == 1:
            fresh_obs["network_state"] = 0
        fresh_obs["staleness"] = self.step_cnt/self.delay
        local_obs = self.array_observation(fresh_obs, self.local_observation_keys)
        self.stale_global_obs = np.copy(self.array_observation(fresh_obs, self.device_global_observation_keys))
        self.stale_local_obs = np.copy(local_obs)
        self.action_histogram = [0, 0]
        self.state = None
        obs = self.server_observation
        return obs

    def render(self):
        if self.current_subpolicy == 0:
            self.wrapped_env.goal.color1 = [0., 0.5, 0.5 ]
            self.wrapped_env.goal.color2 = [0., 0.5, 0.5 ]
        else:
            self.wrapped_env.goal.color1 = [0.5, 0., 0. ]
            self.wrapped_env.goal.color2 = [0.5, 0., 0. ]
        return self.wrapped_env.render()

    @property
    def observation_space(self):
        size = 0
        for k in self.global_observation_keys:
            shape = self.observation_meta_data[k].shape[0]
            size += shape
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(size, ))

    def network_state(self):
        self.delay = self.delay_function()
        if self.delay <= 1:
            self.delay = 1
        return self.delay

    def dict_observation(self):
        dict_obs = self.wrapped_env.dict_observation()
        net_stat = self.delay
        net_stat *= 0.1
        dict_obs["network_state"] = net_stat
        return dict_obs

    @staticmethod
    def net_state_to_delay(network_state):
        network_state = int(round(network_state))
        return network_state

    def update_obs(self):
        # delay is over update
        last_obs = self.wrapped_env.dict_observation()
        local_obs = self.array_observation(last_obs, self.local_observation_keys)
        global_obs = self.array_observation(last_obs, self.device_global_observation_keys)

        self.network_state()
        self.stale_global_obs = np.copy(global_obs)
        self.stale_local_obs = np.copy(local_obs)

    def step(self, action):
        steps = self.net_state_to_delay(self.delay)
        subpolicy = self.subpolicies[action]
        self.current_subpolicy = action
        self.action_histogram[action] += 1
        reward = 0
        done = False
        if action == 0:
            for _ in range(steps):
                device_obs = self.full_global_device_observation
                device_action, _ = subpolicy.predict(device_obs)
                _, __, done, info = self.wrapped_env.step(device_action)
                if done:
                    break

        elif action == 1:

            for _ in range(steps):
                device_obs = self.local_only_device_observation
                self.dummy_obs[0] = device_obs
                dummy_action_vector, self.state = subpolicy.predict(self.dummy_obs, self.state)
                device_action = dummy_action_vector[0]
                _, __, done, info = self.wrapped_env.step(device_action)
                if done:
                    break

        else:
            raise ValueError

        self.update_obs()
        self.state = None
        if self.wrapped_env.energy <= 0:
            done = True

        if done and self.wrapped_env.achieve_goal:
            reward = 1

        elif done and not self.wrapped_env.achieve_goal:
            reward = 0

        obs = self.server_observation
        if done and self.achieve_goal:
            info = {"is_success":True}
        else:
            info = {"is_success":False}

        return obs, reward, done, info

    def timer_heuristic(self, episodes=1000, thresh_hold=6)->list:
        scores = []
        for epi in range(episodes):
            self.reset()
            done = False
            reward = 0
            while not done:
                steps = self.net_state_to_delay(self.delay)
                subpolicy = self.subpolicies[0]
                global_steps = min(thresh_hold, steps)
                local_steps = max(0, steps-global_steps)
                self.update_obs()
                done = False
                for _ in range(global_steps):
                    device_obs = self.full_global_device_observation
                    device_action, _ = subpolicy.predict(device_obs)
                    _, reward, done, info = self.wrapped_env.step(device_action)
                    if done:
                        break
                if local_steps > 0 and not done:
                    subpolicy = self.subpolicies[1]
                    for _ in range(local_steps):

                        device_obs = self.local_only_device_observation

                        device_action, state = subpolicy.predict(device_obs, self.state)
                        if self.state is None:
                            self.state = np.zeros((28, 512))
                            self.state[0] = state
                        _, reward, done, info = self.wrapped_env.step(device_action)
                        if done:
                            break
                self.update_obs()
                self.network_state()
            if reward > 0:
                scores.append(1)
            else:
                scores.append(0)
            if epi % 100 == 0:
                print(epi ,"/", episodes)
                print(np.mean(scores))
        return scores


class NavigationEnvHeuristic(NavigationEnvMeta):
    key = ["staleness"]

    def reset(self):
        super().reset()
        obs = np.asarray([self.delay])
        return obs

    @property
    def observation_space(self):
        return gym.spaces.Box(np.asarray([0]), np.asarray([np.inf]), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(np.asarray([0]), np.asarray([2]), dtype=np.float32)

    def step(self, action):
        steps = self.net_state_to_delay(self.delay)
        subpolicy = self.subpolicies[0]
        thresh_hold =  int(np.round(action * 10))
        global_steps = min(thresh_hold, steps)
        local_steps = max(0, steps - global_steps)
        self.update_obs()
        done = False
        reward = 0
        info = {}
        for _ in range(global_steps):
            device_obs = self.full_global_device_observation
            device_action, _ = subpolicy.predict(device_obs)
            _, ___, done, info = self.wrapped_env.step(device_action)
            if done:
                break
        if local_steps > 0 and not done:
            subpolicy = self.subpolicies[1]
            for _ in range(local_steps):
                device_obs = self.local_only_device_observation
                device_action, _ = subpolicy.predict(device_obs)
                _, __, done, info = self.wrapped_env.step(device_action)
                if done:
                    break
        if done and self.wrapped_env.achieve_goal:
            reward = 1
        elif done:
            reward = -1

        self.update_obs()
        self.network_state()
        obs = np.asarray([self.delay])
        return obs, reward, done, info


class NavigationEnvNetTest(NavigationEnvMeta):
    def step(self, action):
        steps = self.net_state_to_delay(self.delay)
        subpolicy = self.subpolicies[action]
        self.current_subpolicy = action
        self.action_histogram[action] += 1
        # dodge moving subpolicy
        # obs_queue = deque(maxlen=self.max_delay)
        if action == 0:
            wrapped_env_pos = self.wrapped_env.drone.position
            lidar = self.aux_env.lidar_info(pos=wrapped_env_pos)
            dict_obs = self.wrapped_env.dict_observation()
            dict_obs["lidar"] = np.asarray(lidar, dtype=np.float32)
            position_info = dict_obs["position"]
            del dict_obs["obstacle_speed"]
            current_obs = self.aux_env.array_observation(dict_obs)
            for _ in range(steps):
                action, _ = subpolicy.predict(current_obs)
                _, reward, done, info = self.wrapped_env.step(action)
                if done:
                     break
                wrapped_env_pos = self.wrapped_env.drone.position
                lidar = self.aux_env.lidar_info(pos=wrapped_env_pos)
                dict_obs = self.wrapped_env.dict_observation()
                dict_obs["lidar"] = np.asarray(lidar, dtype=np.float32)
                dict_obs["position"] = position_info
                current_obs = self.aux_env.array_observation(dict_obs)

        net_state = self.network_state()
        obs = self.array_observation()
        done = self.wrapped_env.game_over
        reward = 0
        if self.wrapped_env.energy <= 0:
            done = True

        if done and self.wrapped_env.achieve_goal:
            reward = 1

        return np.copy(obs), reward, done, {}


class NavigationEnvPartialObs(NavigationEnvMeta):
    def __init__(self, **kwargs):
        kwargs["subpolicies"] = []
        super().__init__(**kwargs)
        self.wrapped_env = NavigationEnvDefault(**kwargs)
        self.obs_before = None
        self.verbose = False

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    @property
    def observation_space(self):
        return self.wrapped_env.observation_space

    def reset(self):
        self.obs_before = np.copy(self.wrapped_env.reset())
        self.step_cnt = 0
        self.network_state()
        self.update_obs()
        return np.copy(self.obs_before)

    def step(self, action):
        steps = self.net_state_to_delay(self.delay)
        self.step_cnt += 1
        obs, reward, done, info = self.wrapped_env.step(action)
        # obs = self.next_state_network.predict(self.obs_before
        obs_real = np.copy(obs)
        if self.step_cnt >= steps:
            self.step_cnt = 0
            self.network_state()
            self.update_obs()
        else:
            obs = self.full_global_device_observation

        if self.verbose:
                print("state real")
                print(obs_real)
                print("observation")
                print(obs)
        self.obs_before = obs
        return np.copy(obs), reward, done, info


class NavigationEnvMetaNoNet(NavigationEnvMeta):
    observation_meta_data = copy.copy(NavigationEnvDefault.observation_meta_data)
    observation_meta_data_keys = copy.copy(NavigationEnvDefault.observation_meta_data_keys)

    def dict_observation(self):
        dict_obs = self.wrapped_env.dict_observation()
        return dict_obs


class AdversarialRLEnv(NavigationEnvMeta):
    P = 0
    A = 1
    protagonist = None
    antagonist = None

    class DummyEnv(gym.Env):
        def __init__(self, action_space, obs_space):
            self.action_space = action_space
            self.observation_space = obs_space

    def __init__(self, *args, **kwargs):
        self.wrapped_env = NavigationEnvDefault(**kwargs)

        subpolicies = [self.protagonist, self.antagonist]
        super().__init__(subpolicies=subpolicies, *args, **kwargs)

        self.step = None
        self.current_obs = None
        self.cursor = AdversarialRLEnv.P
        self.cursor = AdversarialRLEnv.A

    @classmethod
    def get_models(cls, obs_space, p_action_space, a_action_space):
        protagonist_env = cls.DummyEnv(obs_space=obs_space, action_space=p_action_space)
        antagonist_env = cls.DummyEnv(obs_space=obs_space, action_space=a_action_space)

        from stable_baselines import PPO2
        from stable_baselines.common.vec_env import DummyVecEnv
        protagonist = PPO2(env=DummyVecEnv([lambda:protagonist_env]), policy='MlpPolicy')
        antagonist = PPO2(env=DummyVecEnv([lambda:antagonist_env]), policy='MlpPolicy')
        return protagonist, antagonist

    def render(self, mode="human"):
        return self.wrapped_env.render()

    @classmethod
    def set_protagonist(cls, protagonist):
        cls.protagonist = protagonist

    @classmethod
    def set_antagonist(cls, antagonist):
        cls.antagonist = antagonist

    def switch(self, cursor):

        if cursor == AdversarialRLEnv.A:
            self.cursor = AdversarialRLEnv.A
            self.step = self._step_antagonist
        else:
            self.cursor = AdversarialRLEnv.P
            self.step = self._step_protagonist

    def learn_protagonist(self, steps):
        self.switch(cursor=self.P)
        self.protagonist.learn(steps)

    def learn_antagonist(self, steps):
        self.switch(cursor=self.A)
        self.antagonist.learn(steps)

    @property
    def observation_space(self):
        return self.wrapped_env.observation_space

    @property
    def protagonist_action_space(self):
        return self.wrapped_env.action_space

    @property
    def antagonist_action_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(2, ))

    @property
    def action_space(self):
        if self.cursor == AdversarialRLEnv.P:
            return self.protagonist_action_space

        elif self.cursor == AdversarialRLEnv.A:
            return self.antagonist_action_space
        else:
            raise ValueError("Env cursor must be 0 (Protagonist) or 1 (Antagonist)!")

    def reset(self):
        self.current_obs = self.wrapped_env.reset()
        return self.current_obs.copy()

    def network_state(self, action):
        self.delay = self.delay_function()
        if self.delay <= 1:
            self.delay = 1
        return self.delay

    def _step_antagonist(self, action):
        self.delay = action
        steps = self.net_state_to_delay(self.delay)
        subpolicy = self.subpolicies[0]
        self.current_subpolicy = action
        self.action_histogram[action] += 1
        reward = 0
        done = False
        for _ in range(steps):
            device_obs = self.full_global_device_observation
            device_action, _ = subpolicy.predict(device_obs)
            _, __, done, info = self.wrapped_env.step(device_action)
            if done:
                break

        self.update_obs()
        if self.wrapped_env.energy <= 0:
            done = True

        if done and self.wrapped_env.achieve_goal:
            reward = 1

        elif done and not self.wrapped_env.achieve_goal:
            reward = -1

        obs = self.server_observation
        return obs, -reward, done, {}

    def _step_protagonist(self, action):
        delay, _ = self.antagonist.predict(self.server_observation)
        self.delay = delay
        steps = self.net_state_to_delay(self.delay)
        subpolicy = self.subpolicies[0]
        reward = 0
        done = False
        for _ in range(steps):
            device_obs = self.full_global_device_observation
            device_action, _ = subpolicy.predict(device_obs)
            _, __, done, info = self.wrapped_env.step(device_action)
            if done:
                break

        self.update_obs()
        if self.wrapped_env.energy <= 0:
            done = True

        if done and self.wrapped_env.achieve_goal:
            reward = 1

        elif done and not self.wrapped_env.achieve_goal:
            reward = -1

        obs = self.server_observation
        return obs, reward, done, {}

if __name__ == '__main__':
    import os
    from default_config import config
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    AdversarialRLEnv(**config)