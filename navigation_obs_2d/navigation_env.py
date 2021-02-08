import math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, distance)
# from Integrated_policy_learning.network import Network
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from collections import deque
from navigation_obs_2d.config import *
from navigation_obs_2d.util import *
from navigation_obs_2d.objects import Obstacles

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

    '''
    dictionary representation of observation 
    it is useful handling dict space observation, 
    classifying local observation and global observation, 
    lazy evaluation of observation space; whenever we add or delete some observation information   
    '''
    observation_meta_data = {
        'position':gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        'goal_position':gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        'lidar': gym.spaces.Box(low=0, high=1, shape=(16, )),
        'energy': gym.spaces.Box(low=0, high=1, shape=(1, )),
        'obstacle_speed': gym.spaces.Box(low=-1, high=1, shape=(len(OBSTACLE_POSITIONS), )),
        'obstacle_position': gym.spaces.Box(low=0, high=1, shape=(2 * len(OBSTACLE_POSITIONS), ))
    }

    # meta data keys. It is used to force order to get observation.
    # We may use ordered dict, but saving key with list is more economic
    # rather than create ordered dict whenever steps proceed
    observation_meta_data_keys = ['position', 'goal_position', 'lidar', 'energy', 'obstacle_speed', 'obstacle_position']

    def __init__(self, max_obs_range=3,  max_speed=2, initial_speed=2, **kwargs):
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
        p1 = (0.75, 0.5)
        p2 = (W - 0.5, 0.5)
        self.sky_polys = [[p1, p2, (p2[0], H-0.5), (p1[0], H-0.5)]]
        self.position_intrinsic_reward = None
        self.obstacles = Obstacles(self.world, max_speed)

        self.reset()
        self.game_over = False
        self.prev_shaping = None

        # debug
        self.action = None
        self.obs_queue = deque(maxlen=10)

    @property
    def drone_start_pos(self):
        return

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
        position = normalize_position(self.drone.position, W, H)
        goal_position = normalize_position(self.goal.position, W, H)
        lidar = [l.fraction for l in self.lidar]
        obstacle_speed = self.obstacles.speeds
        obstacle_position = self.obstacles.positions(self.drone.position)
        dict_obs = {
            'position':position,
            'goal_position': goal_position,
            'lidar': lidar,
            'energy': self.energy,
            'obstacle_speed': obstacle_speed,
            'obstacle_position':obstacle_position
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
        self._clean_walls()
        self.world.DestroyBody(self.goal)
        self.goal = None
        self.world.DestroyBody(self.obs_range_plt)
        self.obs_range_plt = None
        self.obstacles.clean_obstacles()

    def _observe_lidar(self, pos):
        for i in range(self.num_beams):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(i * 2 * np.pi / self.num_beams) * self.max_obs_range,
                pos[1] + math.cos(i * 2 * np.pi / self.num_beams) * self.max_obs_range)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

    def _build_wall(self):
        wall_pos = WALL_INFOS['pos']
        wall_ver = WALL_INFOS['vertices']

        for p, v in zip(wall_pos, wall_ver):
            wall = self.world.CreateStaticBody(position=p, angle=0.0,
                                                fixtures=fixtureDef(shape=polygonShape(vertices=v), density=100.0,
                                                friction=0.0, categoryBits=0x001, restitution=1.0,))
            wall.color1 = (1.0, 1.0, 1.0)
            wall.color2 = (1.0, 1.0, 1.0)
            self.walls.append(wall)

    def _build_drone(self):
        # create controller object
        while True:
            drone_pos = (int(np.random.randint(1, 10)), int(np.random.randint(1, 5)))
            self.drone = self.world.CreateDynamicBody(position=drone_pos, angle=0.0,
                                                      fixtures=fixtureDef(shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                                                                          density=5.0, friction=0.1, categoryBits=0x0010,
                                                                          maskBits=0x003, restitution=0.0))
            self.drone.color1 = (0.5, 0.4, 0.9)
            self.drone.color2 = (0.3, 0.3, 0.5)
            self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
            if self.game_over:
                self.world.DestroyBody(self.drone)
                self.game_over = False
            else:
                break
        return drone_pos

    def _build_goal(self):
        goal_pos = GOAL_POS[np.random.randint(len(GOAL_POS))]
        self.goal = self.world.CreateStaticBody(position=goal_pos, angle=0.0,
                                                fixtures=fixtureDef(shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                                                                    density=5.0, friction=0.1, categoryBits=0x002,
                                                                    maskBits=0x0010, restitution=0.0))
        self.goal.color1 = (0., 0.5, 0)
        self.goal.color2 = (0., 0.5, 0)

    def _build_obs_range(self):
        self.obs_range_plt = self.world.CreateKinematicBody(position=(self.drone.position[0], self.drone.position[1]), angle=0.0,
                                                            fixtures=fixtureDef(shape=circleShape(radius=np.float64(self.max_obs_range), pos=(0, 0)),
                                                                                density=0, friction=0, categoryBits=0x0100,
                                                                                maskBits=0x000, restitution=0.3))
        self.obs_range_plt.color1 = (0.2, 0.2, 0.4)
        self.obs_range_plt.color2 = (0.6, 0.6, 0.6)

    def _clean_walls(self):
        while self.walls:
            self.world.DestroyBody(self.walls.pop(0))

    def _get_observation(self, position):
        delta_angle = 2* np.pi/self.num_beams
        ranges = [self.world.raytrace(position, i * delta_angle, self.max_obs_range) for i in range(self.num_beams)]
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
        self.moon.CreateEdgeFixture(vertices=[p1, p2], density=100, friction=0, restitution=1.0)
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        self._build_wall()
        # create obstacles
        print("?????????")
        self.obstacles.build_obstacles()
        drone_pos = self._build_drone()
        # create goal
        np.random.seed(np.random.randint(low=0, high=100000))
        self._build_goal()
        self._build_obs_range()

        self.drawlist = [self.obs_range_plt, self.drone, self.goal] + self.walls + self.obstacles.dynamic_bodies
        self._observe_lidar(drone_pos)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        return np.copy(self.array_observation())

    def step(self, action: np.iterable):
        action = np.asarray(action, dtype=np.float64)
        self.action = action
        self.drone.linearVelocity.Set(action[0], action[1])
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.obstacles.step()

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
            info['is_success'] = self.achieve_goal
            info['energy'] = self.energy
            info['episode'] = {'r': reward, 'l': (1 - self.energy) * 1000}
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
        'position': gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        'goal_position': gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        'lidar': gym.spaces.Box(low=0, high=1, shape=(16,)),
        'energy': gym.spaces.Box(low=0, high=1, shape=(1,)),
        'obstacle_speed': gym.spaces.Box(low=-1, high=1, shape=(len(OBSTACLE_POSITIONS),)),
        'obstacle_position': gym.spaces.Box(low=0, high=1, shape=(2 * len(OBSTACLE_POSITIONS),)),
        'velocity' : gym.spaces.Box(np.array([-3, -3]), np.array([3, 3]), dtype=np.float32)
    }

    # meta data keys. It is used to force order to get observation.
    # We may use ordered dict, but saving key with list is more economic
    # rather than create ordered dict whenever steps proceed
    observation_meta_data_keys = ['position', 'goal_position', 'lidar', 'energy', 'obstacle_speed', 'obstacle_position',
                                  'velocity']

    def __init__(self, max_obs_range=3,  max_speed=5, initial_speed=2, **kwargs):
        super().__init__(max_obs_range, max_speed, initial_speed, )

    def dict_observation(self):
        dict_obs = super().dict_observation()
        velocity = self.drone.linearVelocity
        dict_obs['velocity'] = velocity
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

        self.obstacles.step()
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
            info['is_success'] = self.achieve_goal
            info['energy'] = self.energy
            info['episode'] = {'r': reward, 'l': (1 - self.energy) * 1000}

        obs = np.copy(self.array_observation())
        self.obs_queue.append(obs)
        return obs, reward, done, info


if __name__ == '__main__':
    env = NavigationEnvAcc()
    env.reset()
    done = False
    while not done:
        _, _, done,_ = env.step(np.zeros(2))
        env.render()