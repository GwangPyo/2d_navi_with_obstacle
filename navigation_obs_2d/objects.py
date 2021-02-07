from navigation_obs_2d.config import *
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, distance)

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

class Obstacles(object):
    def __init__(self, world, max_speed):
        self.world = world
        self.speed_table = np.zeros(len(OBSTACLE_POSITIONS))
        self.obstacles = []
        self.max_speed = max_speed
        self.min_speed = 0.5

    def build_obstacles(self):
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
            obstacle = self.world.CreateDynamicBody(position=(pos[0], pos[1]), angle=0.0,
                                                    fixtures=fixtureDef(shape=circleShape(radius=0.3, pos=(0, 0)),
                                                                        density=5.0, friction=0, categoryBits=0x001,
                                                                        maskBits=0x0010, restitution=1.0,))
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

    def clean_obstacles(self):
        while self.obstacles:
            self.world.DestroyBody(self.obstacles.pop(0))

    def step(self):
        for i, o in enumerate(self.obstacles):
            moving_range = o.moving_range.out_of_range(o)
            if moving_range != 0:
                speed = np.random.uniform(low=self.min_speed, high=self.max_speed)
                next_velocity = (speed * moving_range) * o.moving_range.move_direction
                o.linearVelocity.Set(next_velocity[0], next_velocity[1])
                self.speed_table[i] = moving_range  * speed / 5

    def reset(self):
        self.speed_table = np.zeros(len(OBSTACLE_POSITIONS))

    def positions(self, drone_position):
        position = normalize_position(drone_position, W, H)
        obstacle_position = [position - normalize_position(o.position, W, H) for o in self.obstacles]
        return obstacle_position

    def speeds(self):
        return np.copy(self.speed_table)