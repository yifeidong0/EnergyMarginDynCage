from OpenGL.GL import *
from .geometric import *
from ..spaces.objective import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..planners.problem import PlanningProblem
from ..bullet.forwardsimulator import *
from ..structures.toolfunc import *
import math

def rotate_vector(vector, angle):
    """
    Rotate a 2D vector by a given angle.

    Parameters:
    vector: A tuple or list representing the vector (vx, vy).
    angle: The rotation angle in radians.
    
    Returns:
    The rotated vector (vx_new, vy_new).
    """
    rotation_matrix = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle),  np.cos(angle)]
    ])

    return np.dot(rotation_matrix, vector)

def calculate_new_velocity(vx, vy, omega, theta1, theta2):
    """
    Calculate the new velocity of the object after time T.

    Parameters:
    vx, vy: Initial velocity components.
    omega: Angular velocity.
    T: Time duration.
    
    Returns:
    New velocity components (vx_new, vy_new).
    """
    # Angular displacement
    angular_displacement = theta2 - theta1

    # Correct for wrapping
    if angular_displacement > np.pi:
        angular_displacement -= 2 * np.pi
    elif angular_displacement < -np.pi:
        angular_displacement += 2 * np.pi

    # Time duration for the change in orientation
    T = angular_displacement / omega if omega != 0 else 0

    # Angular displacement
    theta = omega * T

    # Rotate the velocity vector
    vx_new, vy_new = rotate_vector([vx, vy], theta)

    return vx_new, vy_new

class WaterSwingControlSpace(ControlSpace):
    def __init__(self, cage):
        self.cage = cage
        self.dynamics_sim = cage.dynamics_sim
        self.dynamics_sim.set_params(cage.params)
        self.dynamics_sim.create_shapes()
        self.obstacles = self.cage.obstacles
        self.is_water_swing = True

    def configurationSpace(self):
        return self.cage.configurationSpace()
    
    def controlSet(self, x):
        return MultiSet(TimeBiasSet(self.cage.time_range,self.cage.controlSet()),self.cage.controlSet())
    
    def nextState(self, x, u):
        return self.eval(x, u, 1.0)
    
    def eval(self, x, u, amount, print_via_points=False):
        # x_i, y_i, theta_i, vx_i, vy_i, alpha_i, xr_i, yr_i, thetar_i = x # state space, 9D (3+3: cage, 3: robot gripper)
        t, ax, ay, omega = u # control space
        tc = t * amount
        mu = [tc, ax, ay, omega]

        # xaug = x + self.cage.gripper_vel
        vxg_init, vyg_init = self.cage.gripper_vel[:2]
        omegag = self.cage.gripper_vel_theta
        vxg, vyg = calculate_new_velocity(vxg_init, vyg_init, omegag, self.cage.start_state[8], x[8])
        xaug = x + [vxg, vyg, omegag]

        self.dynamics_sim.reset_states(xaug)
        x_new, xo_via_points = self.dynamics_sim.run_forward_sim(mu, 1)

        # Make theta fall in [-pi, pi]
        x_new[2] = limit_angle_to_pi(x_new[2])
        x_new[8] = limit_angle_to_pi(x_new[8])

        if print_via_points: # TODO
            self.xo_via_points = [[q[0], q[1]] for q in xo_via_points]

        return x_new[:9]
    
    def interpolator(self, x, u, xnext=None):
        return LambdaInterpolator(lambda s:self.eval(x,u,s), self.configurationSpace(), 10, xnext=xnext)

class WaterSwing:
    def __init__(self, data, dynamics_sim):
        self.dynamics_sim = dynamics_sim
        self.x_range = 10
        self.y_range = 10
        self.offset = 8.0 # extend the landscape
        self.max_velocity = 10
        self.max_ang_velocity = 5
        self.max_acceleration = 20
        self.max_ang_acceleration = 3
        self.mass_object = .05
        self.mass_gripper = 10 # has to be the SAME as the 4face-bottle.urdf file!
        self.moment_object = 1e-3 # moment of inertia.  Solid - (I) = (1/12) * m * (a^2 + b^2) - a=b=0.3
        self.moment_gripper = 1 # has to be the SAME as the 4face-bottle.urdf file! Hollow - (I) = (1/2) * m * R^2 - R ~= 0.5
        self.params = [self.mass_object, self.moment_object, self.mass_gripper, self.moment_gripper]

        # Gripper moving velocity (constant)
        self.gripper_vel = data[9:]
        self.gripper_vel_x = data[9]
        self.gripper_vel_y = data[10]
        self.gripper_vel_theta = data[11]

        self.start_state = data[:9]
        self.goal_state = [0, data[1]-self.offset, 0, 0, 0, 0, 0, 0, 0] # varying goal region # TODO
        self.goal_radius = .2 # MPPI goal radius
        self.goal_half_extent = 1.5 # AO-xxx goal region
        self.time_range = .2

        self.obstacles = []
        self.gravity = -9.81

        self.cspace_bound = [[0, self.x_range], 
                             [0, self.y_range], 
                             [-math.pi, math.pi], 
                             [-self.max_velocity, self.max_velocity],
                             [-self.max_velocity, self.max_velocity],
                             [-self.max_ang_velocity, self.max_ang_velocity],
                             [-self.x_range, 2*self.x_range], 
                             [-self.y_range, 2*self.y_range], 
                             [-math.pi, math.pi], 
                             ]
        
    # def checkStartFeasibility(self): # TODO; bullet collision checking
    #     gripper = AxisNotAlignedBox(self.obstacles[0][:3], self.obstacles[0][3:])
    #     contains = gripper.contains(self.start_state[:2])
    #     return contains

    def controlSet(self):
        return BoxSet([-self.max_acceleration, -self.max_acceleration, -self.max_ang_acceleration], 
                      [self.max_acceleration, self.max_acceleration, self.max_ang_acceleration])

    def controlSpace(self):
        # System dynamics
        return WaterSwingControlSpace(self)

    def workspace(self):
        # For visualization
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [-self.offset,-self.offset]
        wspace.box.bmax = [self.x_range+self.offset, self.y_range+self.offset]
        return wspace
    
    def configurationSpace(self):
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [-self.offset,-self.offset]
        wspace.box.bmax = [self.x_range+self.offset, self.y_range+self.offset]
        wspace.addObstacleParam(self.obstacles)
        res =  MultiConfigurationSpace(wspace,
                                       BoxConfigurationSpace([-math.pi],[math.pi]),
                                       BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]), 
                                       BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]),
                                       BoxConfigurationSpace([-self.max_ang_velocity],[self.max_ang_velocity]),
                                       BoxConfigurationSpace([-2.5*self.x_range],[2.5*self.x_range]),
                                       BoxConfigurationSpace([-2.5*self.y_range],[2.5*self.y_range]),
                                       BoxConfigurationSpace([-math.pi],[math.pi])
                                       ) # this c-space has to cover the state constraint in MPPI, better with some more margins
        return res

    def startState(self):
        return self.start_state

    def goalSet(self):
        return BoxSet([-self.offset, -self.offset, -math.pi,
                       -self.max_velocity, -self.max_velocity, -self.max_ang_velocity,
                       -2.5*self.x_range, -2.5*self.y_range, -math.pi],
                      [self.x_range+self.offset, self.goal_state[1], math.pi,
                       self.max_velocity, self.max_velocity, self.max_ang_velocity,
                       2.5*self.x_range, 2.5*self.y_range, math.pi])


class WaterSwingObjectiveFunction(ObjectiveFunction):
    """Given a function pointwise(x,u), produces the incremental cost
    by incrementing over the interpolator's length.
    """
    def __init__(self, cage, timestep=0.2):
        self.cage = cage
        self.space = cage.controlSpace()
        self.timestep = timestep
        self.xnext = None

    def incremental(self, x, u, uparent=None):
        m = self.cage.mass_object
        I = self.cage.moment_object
        g = self.cage.gravity

        xnext = self.space.nextState(x,u)
        self.xnext = xnext

        # Energy
        # E = 0.5*m*(x[3]**2+x[4]**2) + 0.5*I*x[5]**2 + m*g*x[1]
        # Enext = 0.5*m*(xnext[3]**2+xnext[4]**2) + 0.5*I*xnext[5]**2 + m*g*xnext[1]
        # # c = max((Enext-E), 1e-3) + (2e-2)*(abs(u[0]) + abs(u[1]) + abs(u[2]))
        # c = max((Enext-E), 1e-5)

        # Work (applied force, torque and friction)
        delta_alpha = xnext[2] - x[2]
        if delta_alpha < -math.pi:
            delta_alpha = 2*math.pi + delta_alpha
        if delta_alpha > math.pi:
            delta_alpha = -2*math.pi + delta_alpha
        W = m*u[1]*(xnext[0]-x[0]) + m*(u[2]-g)*(xnext[1]-x[1]) + I*u[3]*delta_alpha
        c = max(W, 1e-5)

        return c


def waterSwingTest(dynamics_sim,
                   data = [3.0, 5.5, 0.0, 0.0, 1.0, 0.0,
                           3.0, 4.3, 0.0, 1.0, 1.0, -np.pi/2]):
    p = WaterSwing(data, dynamics_sim)

    # if p.checkStartFeasibility():
    #     print('In collision!')
    #     return False
    objective = WaterSwingObjectiveFunction(p)
    return PlanningProblem(objective.space,p.startState(),p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean = True)


