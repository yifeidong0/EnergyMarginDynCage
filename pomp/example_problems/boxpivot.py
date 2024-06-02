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
import csv

class BoxPivotControlSpace(ControlSpace):
    def __init__(self, cage):
        self.cage = cage
        self.dynamics_sim = cage.dynamics_sim
        self.dynamics_sim.set_params(cage.params)
        self.dynamics_sim.create_shapes()
        self.obstacles = self.cage.obstacles
        self.cost_inv_coef = cage.cost_inv_coef
        self.is_box_pivot = True

    def configurationSpace(self):
        return self.cage.configurationSpace()
    
    def controlSet(self, x):
        return MultiSet(TimeBiasSet(self.cage.time_range,self.cage.controlSet()),self.cage.controlSet())
    
    def nextState(self, x, u):
        return self.eval(x, u, 1.0)
    
    def eval(self, x, u, amount, print_via_points=False):
        t, ax, omega = u # control space
        tc = t * amount
        mu = [tc, ax, omega]
        vxg1, vxg2 = self.cage.gripper_vel[:2]
        xaug = x + [vxg1, vxg2]

        self.dynamics_sim.reset_states(xaug)
        x_new, xo_via_points = self.dynamics_sim.run_forward_sim(mu, 1)

        if print_via_points: # TODO
            self.xo_via_points = [[q[0], q[1]] for q in xo_via_points]

        return x_new[:self.cage.num_state]
    
    def interpolator(self, x, u, xnext=None):
        return LambdaInterpolator(lambda s:self.eval(x,u,s), self.configurationSpace(), 10, xnext=xnext)

class BoxPivot:
    def __init__(self, data, dynamics_sim, save_hyperparams=1, lateral_friction_coef=0.8):
        self.dynamics_sim = dynamics_sim
        self.num_state = 8 # box 6 + spring 2
        self.num_input = 2
        self.x_range = 12
        self.y_range = 12
        self.offset = 100.0 # extend the landscape
        self.max_velocity = 100
        self.max_ang_velocity = 100
        self.max_acceleration = 50 # 1 for prob-aoest
        self.max_ang_acceleration = 50 # 1 for prob-aoest
        self.task_goal_margin = 0.1 * math.pi/6
        self.maneuver_goal_margin = 1/3 * math.pi/6
        self.maneuver_goal_tmax = 1.5
        self.cost_inv_coef = -3e0

        self.mass_object = .3
        self.mass_gripper = 1e-1
        self.moment_object = 8e-1 # moment of inertia.  Solid - (I) = (1/12) * m * (a^2 + b^2) - a=b=4
        self.moment_gripper = 3e-3 # (2/5) * mass_ball * radius_ball**2
        self.lateral_friction_coef = lateral_friction_coef
        self.for_paper_vis = 1
        if self.for_paper_vis:
            self.rest_length = 1
            self.k = 5 # Spring constant
            self.height_spring = 0.9
        else:
            self.rest_length = 3
            self.k = 2
            self.height_spring = 3.7
        self.quasistatic_motion = False
        self.params = [self.mass_object, self.moment_object, self.mass_gripper, self.moment_gripper, self.lateral_friction_coef,
                       self.rest_length, self.k, self.height_spring, self.quasistatic_motion]
        
        self.gripper_vel = data[self.num_state:] # spring 1,2 velocity
        self.start_state = data[:self.num_state]
        self.goal_state = [0, 0, 0, 0, 0, 0, 0, 0] # varying goal region # TODO
        self.time_range = .1
        self.obstacles = []
        self.gravity = -9.81

        # Gripper moving velocity (constant)
        if self.quasistatic_motion:
            self.gripper_vel = [0, 0]
        self.hyperparams = [self.x_range, self.y_range, self.offset, self.max_velocity, self.max_ang_velocity, self.max_acceleration, 
                            self.max_ang_acceleration, self.time_range, self.gravity, self.task_goal_margin, self.maneuver_goal_margin,
                            self.maneuver_goal_tmax, self.cost_inv_coef] + self.params
        self.hyperparams_header = ['x_range', 'y_range', 'offset', 'max_velocity', 'max_ang_velocity', 'max_acceleration',
                                   'max_ang_acceleration', 'time_range', 'gravity', 'task_goal_margin', 'maneuver_goal_margin', 'maneuver_goal_tmax',
                                   'cost_inv_coef', 
                                   'mass_object', 'moment_object', 'mass_gripper', 'moment_gripper', 'lateral_friction_coef', 'rest_length',
                                   'k', 'height_spring', 'quasistatic_motion']
        if save_hyperparams:
            self.saveHyperparams()

    def saveHyperparams(self, filename='box_pivot_hyperparams.csv'):
        with open(filename, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for header, data in zip(self.hyperparams_header, self.hyperparams):
                csv_writer.writerow([header, data])

    def controlSet(self):
        return BoxSet([-self.max_acceleration, -1.0*self.max_ang_acceleration], 
                      [self.max_acceleration, 1.0*self.max_ang_acceleration])

    def controlSpace(self):
        # System dynamics
        return BoxPivotControlSpace(self)

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
                                       BoxConfigurationSpace([-self.offset],[self.x_range+self.offset]), # spring positions
                                       BoxConfigurationSpace([-self.offset],[self.y_range+self.offset]),
                                       ) # this c-space has to cover the state constraint in MPPI, better with some more margins
        return res

    def startState(self):
        return self.start_state

    def goalSet(self):
        # workspaceset = BoxSet([-self.offset, -self.offset], [self.x_range+self.offset, self.y_range+self.offset])
        # thetaset = UnionBoxSet([[-math.pi,],[math.pi-0.1]],[[-math.pi+0.1,],[math.pi]])
        # restset = BoxSet([-self.max_velocity, -self.max_velocity, -self.max_ang_velocity, -2.5*self.x_range, -2.5*self.x_range],
        #                  [self.max_velocity, self.max_velocity, self.max_ang_velocity, 2.5*self.x_range, 2.5*self.x_range])
        # return MultiSet(workspaceset, thetaset, restset)
        return self.complementCaptureSet()
    
    def successSet(self):
        return BoxSet([-self.offset, -self.offset, math.pi/2-self.task_goal_margin,
                       -self.max_velocity, -self.max_velocity, -self.max_ang_velocity,
                       -self.x_range-self.offset, -self.x_range-self.offset],
                      [self.x_range+self.offset, self.y_range+self.offset, math.pi,
                       self.max_velocity, self.max_velocity, self.max_ang_velocity,
                       self.x_range+self.offset, self.x_range+self.offset])
    
    def complementCaptureSet(self):
        """The object is maneuverable if the pivot point velocity is not too large."""
        return BoxPivotNonCaptureSet([-self.offset, -self.offset, 0.0,
                       -self.max_velocity, -self.max_velocity, -self.max_ang_velocity,
                       -self.x_range-self.offset, -self.x_range-self.offset],
                    #    -2.5*self.x_range, -2.5*self.x_range],
                      [self.x_range+self.offset, self.y_range+self.offset, math.pi/2,
                       self.max_velocity, self.max_velocity, self.max_ang_velocity,
                       self.x_range+self.offset, self.x_range+self.offset])
                    #    2.5*self.x_range, 2.5*self.x_range])
    
    
class BoxPivotObjectiveFunction(ObjectiveFunction):
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
        I = self.cage.moment_object # TODO
        xnext = self.space.nextState(x,u)
        self.xnext = xnext

        # Work (applied force, torque and friction)
        W = m*u[1]*(xnext[0]-x[0]) + I*u[2]*(xnext[2]-x[2])
        c = max(abs(W), 1e-5)
        return c


def BoxPivotTest(dynamics_sim,
                data=[6.499079904444838,2.397549737084281,0.22622950186326807,1.8207181866422655,1.1495647514494098,0.7609057703831511,
                      1.6157827276435577,4.439045449428635,0.20465756939330715,2.192905112691566],
                save_hyperparams=False,
                lateral_friction_coef=0.8,
                 ):
    p = BoxPivot(data, dynamics_sim, save_hyperparams, lateral_friction_coef)
    objective = BoxPivotObjectiveFunction(p)
    return PlanningProblem(objective.space,
                           p.startState(),
                           p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean=True,
                           successSet=p.successSet(),
                           complementCaptureSet=p.complementCaptureSet(),
                           )


