from OpenGL.GL import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import csv
from .geometric import *
from ..spaces.objective import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..planners.problem import PlanningProblem
from ..bullet.forwardsimulator import *
from ..structures.toolfunc import *

def calculate_work(alpha, I, x, xnext):
    """
    Calculate the work done by applying torque to a rigid object.

    Parameters:
    alpha (np.array): The angular acceleration applied (3 elements).
    I (np.array): The moment of inertia of the object (3x3 matrix).
    x (np.array): The initial Euler angles of the object (3 elements).
    xnext (np.array): The final Euler angles of the object (3 elements).

    Returns:
    float: The work done on the object.
    """
    # calculate the torque
    torque = np.dot(I, alpha)

    # Convert Euler angles to quaternions
    q1 = R.from_euler('xyz', x).as_quat()
    q2 = R.from_euler('xyz', xnext).as_quat()

    # Calculate the relative rotation quaternion
    dq = R.from_quat(q2) * R.from_quat(q1).inv()

    # Convert the relative rotation to an angular displacement vector
    theta = dq.as_rotvec()

    # Calculate the work done as the dot product of torque and angular displacement
    work = np.dot(torque, theta)

    return work

class GripperControlSpace(ControlSpace):
    def __init__(self, cage):
        self.cage = cage
        self.dynamics_sim = cage.dynamics_sim
        self.dynamics_sim.set_params(cage.params)
        self.dynamics_sim.create_shapes()
        self.obstacles = self.cage.obstacles
        self.cost_inv_coef = self.cage.cost_inv_coef
        self.is_gripper = True

    def configurationSpace(self):
        return self.cage.configurationSpace()
    
    def controlSet(self, x):
        return MultiSet(TimeBiasSet(self.cage.time_range, self.cage.controlSet()),
                        self.cage.controlSet())
    
    def nextState(self, x, u):
        return self.eval(x, u, 1.0)
    
    def eval(self, x, u, amount, print_via_points=False):
        t, ax, ay, az, alphax, alphay, alphaz = u # control space
        tc = t * amount
        mu = [tc, ax, ay, az, alphax, alphay, alphaz]

        xaug = x + self.cage.gripper_vel
        self.dynamics_sim.reset_states(xaug)
        x_new, xo_via_points = self.dynamics_sim.run_forward_sim(mu, 1)

        if print_via_points:
            self.xo_via_points = xo_via_points

        return x_new
    
    def interpolator(self, x, u, xnext=None):
        return LambdaInterpolator(lambda s:self.eval(x,u,s), self.configurationSpace(), 10, xnext=xnext)

class Gripper:
    def __init__(self, data, dynamics_sim, save_hyperparams=1, quasistatic_motion=0, lateral_friction_coef=0.1, mass_object=3.0):
        self.dynamics_sim = dynamics_sim
        self.movable_joints = [1,2,3,5,6,7,9,10,11]
        self.num_joints = len(self.movable_joints) + 1 # 9+1, plus z-axis movement of the table
        self.dim_workspace = 3
        self.dim_object = 6 + 6
        self.dim_state = self.dim_object + self.num_joints # 6+6+9+1
        self.x_range = 4
        self.z_range = 4
        self.y_range = 4
        self.offset = 0.0 # extend the landscape
        self.max_velocity = 100
        self.max_ang_velocity = 100
        self.max_acceleration = 3
        self.max_ang_acceleration = .3
        self.length_object = .3
        self.mass_object = mass_object
        self.moment_object = [(1/12) * self.mass_object * (self.length_object**2 + self.length_object**2),]*self.dim_workspace
        self.lateral_friction_coef = lateral_friction_coef

        # Gripper joints velocity and table z-axis velocity
        if quasistatic_motion:
            self.gripper_vel = [0.0,] * self.num_joints # list[9+1,]
        else:
            self.gripper_vel = data[self.dim_state:] # list[9+1,]

        self.start_state = data[:self.dim_state] # list[6+6+9+1,]
        self.start_gripper_pos = self.start_state[-self.num_joints:] # list[9+1,]
        self.time_range = 1e-1
        self.params = [self.mass_object, self.moment_object, self.length_object, self.movable_joints, self.start_gripper_pos, self.lateral_friction_coef]
        self.obstacles = []
        self.gravity = -9.81

        self.success_vz_thres = -0.5
        self.success_z_thres = 0.25
        self.cost_inv_coef = -3e0
        self.hyperparams = [self.x_range, self.y_range, self.offset, self.max_velocity, self.max_ang_velocity, self.max_acceleration, 
                            self.max_ang_acceleration, self.time_range, self.gravity, self.success_vz_thres, self.success_z_thres,
                            self.cost_inv_coef] + self.params
        self.hyperparams_header = ['x_range', 'y_range', 'offset', 'max_velocity', 'max_ang_velocity', 'max_acceleration',
                                   'max_ang_acceleration', 'time_range', 'gravity', 'success_vz_thres', 'success_z_thres', 
                                   'cost_inv_coef',
                                   'mass_object', 'moment_object', 'length_object', 'movable_joints', 'start_gripper_pos', 'lateral_friction_coef']
        if save_hyperparams:
            self.saveHyperparams()

        self.bmin = [-(self.x_range+self.offset)/2, -(self.z_range+self.offset)/2, -(self.y_range+self.offset)/2,
                    -math.pi, -math.pi, -math.pi,
                    -self.max_velocity, -self.max_velocity, -self.max_velocity,
                    -self.max_ang_velocity, -self.max_ang_velocity, -self.max_ang_velocity,
                    *[-5e-2,]*(self.num_joints-1), -1e3,
                    ]
        self.bmax = [(self.x_range+self.offset)/2, (self.z_range+self.offset)/2, (self.y_range+self.offset)/2,
                    math.pi, math.pi, math.pi,
                    self.max_velocity, self.max_velocity, self.max_velocity,
                    self.max_ang_velocity, self.max_ang_velocity, self.max_ang_velocity,
                    *[math.pi/2,]*(self.num_joints-1), 1,
                    ]

    def saveHyperparams(self, filename='gripper_hyperparams.csv'):
        with open(filename, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for header, data in zip(self.hyperparams_header, self.hyperparams):
                csv_writer.writerow([header, data])

    def controlSet(self):
        return BoxSet([-self.max_acceleration,-0.*self.max_acceleration,-self.max_acceleration,] + [-0.0*self.max_ang_acceleration,]*self.dim_workspace, 
                      [self.max_acceleration,0.*self.max_acceleration,+self.max_acceleration,] + [0.0*self.max_ang_acceleration,]*self.dim_workspace)

    def controlSpace(self):
        # System dynamics
        return GripperControlSpace(self)

    def workspace(self):
        # For visualization. Order of x,z,y for better OpenGL visualization
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [-(self.x_range+self.offset)/2, -(self.z_range+self.offset)/2]
        wspace.box.bmax = [(self.x_range+self.offset)/2, (self.z_range+self.offset)/2]
        return wspace
    
    def configurationSpace(self):
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [-(self.x_range+self.offset)/2, -(self.z_range+self.offset)/2]
        wspace.box.bmax = [(self.x_range+self.offset)/2, (self.z_range+self.offset)/2]
        wspace.addObstacleParam(self.obstacles)

        res =  MultiConfigurationSpace(wspace, 
                                       BoxConfigurationSpace([-(self.y_range+self.offset)/2],[(self.y_range+self.offset)/2]), 
                                       *[BoxConfigurationSpace([-math.pi],[math.pi]),]*self.dim_workspace,
                                       *[BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]),]*self.dim_workspace,
                                       *[BoxConfigurationSpace([-self.max_ang_velocity],[self.max_ang_velocity]),]*self.dim_workspace,
                                       *[BoxConfigurationSpace([-5e-2],[math.pi/2]),]*(self.num_joints-1),
                                       BoxConfigurationSpace([-1e3],[1]), 
                                       )
        return res

    def startState(self):
        return self.start_state

    def successSet(self):
        return GripperSuccessSet(self.bmin, self.bmax, self.success_z_thres, self.success_vz_thres)
    
    def complementCaptureSet(self):
        return GripperNonCaptureSet(self.bmin, self.bmax, self.success_z_thres, self.success_vz_thres)
    
    def goalSet(self):
        return BoxSet([-(self.x_range+self.offset)/2, -(self.z_range+self.offset)/2, -(self.y_range+self.offset)/2,
                       -math.pi, -math.pi, -math.pi,
                       -self.max_velocity, -self.max_velocity, -self.max_velocity,
                       -self.max_ang_velocity, -self.max_ang_velocity, -self.max_ang_velocity,
                       *[-5e-2,]*(self.num_joints-1), -1e3,
                       ],
                      [(self.x_range+self.offset)/2, 1.-(self.z_range+self.offset)/2, (self.y_range+self.offset)/2,
                       math.pi, math.pi, math.pi,
                       self.max_velocity, self.max_velocity, self.max_velocity,
                       self.max_ang_velocity, self.max_ang_velocity, self.max_ang_velocity,
                       *[math.pi/2,]*(self.num_joints-1), 1,
                       ])
        # return self.successSet()
    
class GripperObjectiveFunction(ObjectiveFunction):
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
        I = np.diag(self.cage.moment_object)
        g = self.cage.gravity

        # Roll out
        xnext = self.space.nextState(x,u)
        self.xnext = xnext

        # Calculate the work done by the applied force and torque
        alpha = np.array(u[4:])
        W_SO3 = calculate_work(alpha, I, np.array(x[3:6]), np.array(xnext[3:6]))
        W_R3 = m*u[1]*(xnext[0]-x[0]) + m*u[2]*(xnext[2]-x[2]) + m*(u[3])*(xnext[1]-x[1]) # u[1:4] - x,y,z, x[:3] - x,z,y, TODO: g???
        W_R3 = m*u[1]*(xnext[0]-x[0]) + m*u[2]*(xnext[2]-x[2]) + m*(u[3]-g)*(xnext[1]-x[1]) # u[1:4] - x,y,z, x[:3] - x,z,y

        # Work (applied force, torque and friction)
        W = W_R3 + W_SO3
        c = max(abs(W), 1e-5)
        return c

def GripperTest(dynamics_sim, 
                data=[0.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [0.0,]*6 + [math.pi/12]*9 + [0.0,] + [0.0]*9 + [0.0,],
                save_hyperparams=False,
                lateral_friction_coef=0.1,
                mass_object=3.0,
                ):
    p = Gripper(data, 
                dynamics_sim, 
                save_hyperparams=save_hyperparams,
                lateral_friction_coef=lateral_friction_coef,
                mass_object=mass_object,
                )
    objective = GripperObjectiveFunction(p)
    return PlanningProblem(objective.space,
                           p.startState(),
                           p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean=True,
                           successSet=p.successSet(),
                           complementCaptureSet=p.complementCaptureSet(),
                           )


