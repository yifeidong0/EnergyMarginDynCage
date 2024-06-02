from OpenGL.GL import *
from scipy.spatial.transform import Rotation as R
import numpy as np
from .geometric import *
from ..spaces.objective import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..planners.problem import PlanningProblem
from ..bullet.forwardsimulator import *
from ..structures.toolfunc import *

class ShufflingControlSpace(ControlSpace):
    def __init__(self, cage):
        self.cage = cage
        self.dynamics_sim = cage.dynamics_sim
        self.dynamics_sim.set_params(cage.params)
        self.dynamics_sim.create_shapes()
        self.obstacles = self.cage.obstacles
        self.is_gripper = True

    def configurationSpace(self):
        return self.cage.configurationSpace()
    
    def controlSet(self, x):
        return MultiSet(TimeBiasSet(self.cage.time_range, self.cage.controlSet()),
                        self.cage.controlSet())
    
    def nextState(self, x, u):
        return self.eval(x, u, 1.0)
    
    def eval(self, x, u, amount, print_via_points=False):
        t, alpha0, alpha1, alpha2, alpha3 = u
        tc = t * amount
        mu = [tc, alpha0, alpha1, alpha2, alpha3]

        xaug = x + self.cage.gripper_vel
        self.dynamics_sim.reset_states(xaug)
        x_new, xo_via_points = self.dynamics_sim.run_forward_sim(mu, 1)

        if print_via_points:
            self.xo_via_points = xo_via_points

        return x_new
    
    def interpolator(self, x, u, xnext=None):
        return LambdaInterpolator(lambda s:self.eval(x,u,s), self.configurationSpace(), 10, xnext=xnext)

class Shuffling:
    def __init__(self, data, dynamics_sim,):
        self.dynamics_sim = dynamics_sim
        # self.movable_joints = movable_joints
        self.num_joints = 4
        self.dim_workspace = 3
        self.dim_se3 = 6
        self.dim_gripper = 1
        self.dim_state = 2*(self.dim_se3 + self.num_joints) + self.dim_gripper # 21
        self.y_range = 12
        self.z_range = 5
        self.x_range = 2
        self.offset = 0.0 # extend the landscape
        self.max_velocity = 5
        self.max_ang_velocity = 10
        self.max_joint_torque = .5

        # ShufflingControlSpace moving velocity (constant)
        self.gripper_vel = data[self.dim_state:] # list[1,]
        self.start_state = data[:self.dim_state] # list[2*6+2*8+1,] gripper joints velocity
        self.goal_state = [0,] * self.dim_state # varying goal region
        self.time_range = 2

        self.params = []
        self.obstacles = []
        self.gravity = -9.81

    def controlSet(self):
        return BoxSet([-self.max_joint_torque,]*self.num_joints, # 4D
                      [self.max_joint_torque,]*self.num_joints, )

    def controlSpace(self):
        # System dynamics
        return ShufflingControlSpace(self)

    def workspace(self):
        # For visualization. Order of x,z,y for better OpenGL visualization
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [-2-(self.y_range+self.offset)/2, 0.0]
        wspace.box.bmax = [-2+(self.y_range+self.offset)/2, self.z_range+self.offset]
        return wspace
    
    def configurationSpace(self):
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [-2-(self.y_range+self.offset)/2, 0.0]
        wspace.box.bmax = [-2+(self.y_range+self.offset)/2, self.z_range+self.offset]
        wspace.addObstacleParam(self.obstacles)

        res =  MultiConfigurationSpace(wspace, 
                                       BoxConfigurationSpace([-(self.x_range+self.offset)/2],[(self.x_range+self.offset)/2]), # 3
                                       *[BoxConfigurationSpace([-math.pi],[math.pi]),]*self.dim_workspace, # 6
                                       *[BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]),]*self.dim_workspace, # 9
                                       *[BoxConfigurationSpace([-self.max_ang_velocity],[self.max_ang_velocity]),]*self.dim_workspace, # 12
                                       *[BoxConfigurationSpace([-.6],[.6]),]*self.num_joints, # 16
                                       *[BoxConfigurationSpace([-self.max_ang_velocity],[self.max_ang_velocity]),]*self.num_joints, # 20
                                       BoxConfigurationSpace([0.0],[2*self.z_range+self.offset]), # 21
                                       )
        return res

    def startState(self):
        return self.start_state

    def goalSet(self):
        return BoxSet([-2-(self.y_range+self.offset)/2, 0.0, -(self.x_range+self.offset)/2,
                       -math.pi, -math.pi, -math.pi,
                       -self.max_velocity, -self.max_velocity, -self.max_velocity,
                       -self.max_ang_velocity, -self.max_ang_velocity, -self.max_ang_velocity,
                       *[-1.5e-1,]*self.num_joints,
                       *[-self.max_ang_velocity,]*self.num_joints,
                       0.0,
                       ],
                      [-2+(self.y_range+self.offset)/2, self.z_range+self.offset, (self.x_range+self.offset)/2,
                       math.pi, math.pi, math.pi,
                       self.max_velocity, self.max_velocity, self.max_velocity,
                       self.max_ang_velocity, self.max_ang_velocity, self.max_ang_velocity,
                       *[1.5e-1,]*self.num_joints,
                       *[self.max_ang_velocity,]*self.num_joints,
                       2*self.z_range+self.offset,
                       ])
    
class ShufflingObjectiveFunction(ObjectiveFunction):
    """Given a function pointwise(x,u), produces the incremental cost
    by incrementing over the interpolator's length.
    """
    def __init__(self, cage, timestep=0.2):
        self.cage = cage
        self.space = cage.controlSpace()
        self.timestep = timestep
        self.xnext = None

    def incremental(self, x, u, uparent=None):
        """Only the elasticity of the card stack is considered as the cost function. 
        The mass of the card is assumed to be 0."""
        torques = u[1:]

        # Roll out
        xnext = self.space.nextState(x,u)
        self.xnext = xnext
        joint_pos = x[2*self.cage.dim_se3:2*self.cage.dim_se3+self.cage.num_joints]
        joint_pos_next = xnext[2*self.cage.dim_se3:2*self.cage.dim_se3+self.cage.num_joints]

        # Calculate the work done by the applied joint torques
        W_joint = sum([tau*(jnext-j) for tau,j,jnext in zip(torques, joint_pos, joint_pos_next)])

        # Energy
        # E = 0.5*m*(x[3]**2+x[4]**2) + 0.5*I*x[5]**2 + m*g*x[1]
        # Enext = 0.5*m*(xnext[3]**2+xnext[4]**2) + 0.5*I*xnext[5]**2 + m*g*xnext[1]
        # # c = max((Enext-E), 1e-3) + (2e-2)*(abs(u[0]) + abs(u[1]) + abs(u[2]))
        # c = max((Enext-E), 1e-5)

        # Work (applied force, torque and friction)
        # W = W_R3 + W_SO3
        c = max(W_joint, 1e-5)

        return c

def ShufflingTest(dynamics_sim, data=[0,]*22):
    p = Shuffling(data, dynamics_sim)

    objective = ShufflingObjectiveFunction(p)
    return PlanningProblem(objective.space,
                           p.startState(),
                           p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean=True)


