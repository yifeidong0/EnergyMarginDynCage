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
"""Planar push problem with a gripper and multiple objects."""

class PlanePushMultiControlSpace(ControlSpace):
    def __init__(self, cage):
        self.cage = cage
        self.num_objects = cage.num_objects
        self.dynamics_sim = cage.dynamics_sim
        self.dynamics_sim.set_params(cage.params)
        self.dynamics_sim.create_shapes()
        self.obstacles = self.cage.obstacles
        self.cost_inv_coef = self.cage.cost_inv_coef
        self.is_plane_push_multi = True

    def configurationSpace(self):
        return self.cage.configurationSpace()
    
    def controlSet(self, x):
        return MultiSet(TimeBiasSet(self.cage.time_range,self.cage.controlSet()),self.cage.controlSet())
    
    def nextState(self, x, u, is_planner=False):
        return self.eval(x, u, 1.0, is_planner=is_planner)
    
    def eval(self, x, u, amount, print_via_points=False, is_planner=False):
        t = u[0] # control space
        tc = t * amount
        mu = [tc,] + u[1:]

        xaug = x if is_planner else (x + self.cage.gripper_vel)
        self.dynamics_sim.reset_states(xaug)
        x_new, xo_via_points = self.dynamics_sim.run_forward_sim(mu, 1)

        if print_via_points:
            self.xo_via_points = [[q[0], q[1]] for q in xo_via_points]

        if is_planner:
            return x_new
        else:
            return x_new[:-3]
    
    def interpolator(self, x, u, xnext=None):
        return LambdaInterpolator(lambda s:self.eval(x,u,s), self.configurationSpace(), 10, xnext=xnext)

class PlanePushMulti:
    def __init__(self, data, dynamics_sim, save_hyperparams=False, lateral_friction_coef=0.3, quasistatic_motion=0, paper_version=1):
        self.dim_se2 = 3
        self.num_objects = int(len(data) / (2*self.dim_se2)) - 1
        self.nx = len(data) - self.dim_se2 # state space dimension
        self.nu = 1 + self.num_objects*self.dim_se2 # control space dimension
        self.dynamics_sim = dynamics_sim
        self.max_velocity = 100
        self.max_ang_velocity = 10 # 2
        self.max_ang_acceleration = .5 * lateral_friction_coef/0.3
        self.angle_slope = 0.0 * math.pi  # equivalent to on a slope
        self.lateral_friction_coef = lateral_friction_coef
        self.task_goal_margin = 0.1
        self.capture_goal_margin = .4

        if paper_version:
            self.x_range = 2
            self.y_range = 2
            # lateral_friction_coef = 0.5
            self.max_acceleration = 1 * lateral_friction_coef/0.2
            self.y_obstacle = 1.6 # the lower rim y_pos of the obstacle
            self.maneuver_goal_tmax = 1
            self.offset = 0.1 # extend the landscape
            self.time_range = .1
            self.cost_inv_coef = -5e0
            self.maneuver_goal_tmax = 1
        else:
            self.x_range = 10
            self.y_range = 10
            self.max_acceleration = 2 * lateral_friction_coef/0.3
            self.y_obstacle = 9 # the lower rim y_pos of the obstacle
            self.maneuver_goal_tmax = 1.5
            self.offset = 2.0
            self.time_range = .5
            self.cost_inv_coef = -1e0
            self.maneuver_goal_tmax = 1

        self.obstacle_borderline = [[-self.offset, self.y_obstacle+0.1], [self.x_range+self.offset, self.y_obstacle+0.1]] # for OpenGL vis, 0.1 for offset

        self.object_name = 'cylinder' # 'box', 'cylinder'
        self.gripper_name = 'box' # 'box', 'cylinder', 'bowl'
        self.mass_object = .2
        self.mass_gripper = 4
        factor_object = 1e-1 if self.object_name == 'box' else 1e-3
        factor_gripper = 1e-1 if (self.gripper_name == 'box' or self.gripper_name == 'bowl') else 1e-3
        self.moment_object = self.mass_object * factor_object # moment of inertia
        self.moment_gripper = self.mass_gripper * factor_gripper
        self.half_extents_object = [0.6, 0.2,] # for box

        self.params = [self.mass_object, self.moment_object, self.mass_gripper, self.moment_gripper, self.y_obstacle, self.angle_slope,
                       self.object_name, self.gripper_name, self.lateral_friction_coef, self.num_objects]
        self.c_space_boundary = [[0, self.x_range], [0, self.y_range], [-0.8*math.pi, 0.8*math.pi], 
                                 [-0.8*self.max_velocity, 0.8*self.max_velocity], [-0.8*self.max_velocity, 0.8*self.max_velocity], [-0.8*self.max_ang_velocity, 0.8*self.max_ang_velocity], 
                                 [-self.x_range, self.x_range], [-self.y_range, self.y_range], [-0.8*math.pi, 0.8*math.pi],
                                 [-0.8*self.max_velocity, 0.8*self.max_velocity], [-0.8*self.max_velocity, 0.8*self.max_velocity], [-0.8*self.max_ang_velocity, 0.8*self.max_ang_velocity], 
                                 ] # shrink the c-space a bit for MPPI
        self.c_space_boundary = self.c_space_boundary[:2*self.dim_se2] * self.num_objects + self.c_space_boundary[2*self.dim_se2:]

        # Gripper moving velocity (constant)
        if not quasistatic_motion:
            self.gripper_vel = data[-3:]
            self.gripper_vel_x = data[-3]
            self.gripper_vel_y = data[-2]
            self.gripper_vel_theta = data[-1]
        else:
            self.gripper_vel = [0, 0, 0]
            self.gripper_vel_x = 0
            self.gripper_vel_y = 0
            self.gripper_vel_theta = 0

        self.start_state = data[:-3]
        self.obstacles = []
        self.gravity = -9.81

        self.hyperparams = [self.x_range, self.y_range, self.offset, self.max_velocity, self.max_ang_velocity, self.max_acceleration, 
                            self.max_ang_acceleration, self.time_range, self.gravity, self.task_goal_margin, self.capture_goal_margin,
                            self.maneuver_goal_tmax, self.cost_inv_coef] + self.params
        self.hyperparams_header = ['x_range', 'y_range', 'offset', 'max_velocity', 'max_ang_velocity', 'max_acceleration',
                                   'max_ang_acceleration', 'time_range', 'gravity', 'task_goal_margin', 'capture_goal_margin', 'maneuver_goal_tmax',
                                   'cost_inv_coef',
                                   'mass_object', 'moment_object', 'mass_gripper', 'moment_gripper', 'y_obstacle', 'angle_slope',
                                   'object_name', 'gripper_name', 'lateral_friction_coef']
        if save_hyperparams:
            self.saveHyperparams()

    def saveHyperparams(self, filename='push_press_hyperparams.csv'):
        with open(filename, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for header, data in zip(self.hyperparams_header, self.hyperparams):
                csv_writer.writerow([header, data])

    def controlSet(self):
        return BoxSet([-self.max_acceleration, -self.max_acceleration, -self.max_ang_acceleration] * self.num_objects, 
                      [self.max_acceleration, self.max_acceleration, self.max_ang_acceleration] * self.num_objects)

    def controlSpace(self):
        # System dynamics
        return PlanePushMultiControlSpace(self)

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
        res =  MultiConfigurationSpace(
                                       *[wspace,
                                       BoxConfigurationSpace([-math.pi],[math.pi]),
                                       BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]), 
                                       BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]),
                                       BoxConfigurationSpace([-self.max_ang_velocity],[self.max_ang_velocity]),]*self.num_objects,
                                       BoxConfigurationSpace([-2.5*self.x_range],[2.5*self.x_range]),
                                       BoxConfigurationSpace([-2.5*self.y_range],[2.5*self.y_range]),
                                       BoxConfigurationSpace([-math.pi],[math.pi])
                                       ) # this c-space has to cover the state constraint in MPPI, better with some more margins
        return res

    def startState(self):
        return self.start_state

    def successSet(self, ball_rad=0.05):
        wsbmin = [-self.offset, self.y_obstacle-ball_rad-self.task_goal_margin,]
        wsbmax = [self.x_range+self.offset, self.y_obstacle-ball_rad+self.task_goal_margin,]
        bmin = [-math.pi, -self.max_velocity, -self.max_velocity, -self.max_ang_velocity,]
        bmax = [math.pi, self.max_velocity, self.max_velocity, self.max_ang_velocity,]
        bmin_ee = [-2.5*self.x_range, -2.5*self.y_range, -math.pi,]
        bmax_ee = [2.5*self.x_range, 2.5*self.y_range, math.pi,]
        return MultiSet(*[BoxSet(wsbmin+bmin, wsbmax+bmax),]*self.num_objects, 
                        BoxSet(bmin_ee, bmax_ee)) # multiset: consistent with other sets

    def complementSuccessSet(self, ball_rad=0.05):
        wsbmin = [-self.offset, -self.offset,]
        wsbmax = [self.x_range+self.offset, self.y_obstacle-ball_rad-self.task_goal_margin,]
        bmin = [-math.pi, -self.max_velocity, -self.max_velocity, -self.max_ang_velocity,]
        bmax = [math.pi, self.max_velocity, self.max_velocity, self.max_ang_velocity,]
        bmin_ee = [-2.5*self.x_range, -2.5*self.y_range, -math.pi,]
        bmax_ee = [2.5*self.x_range, 2.5*self.y_range, math.pi,]
        return MultiSet(*[BoxSet(wsbmin+bmin, wsbmax+bmax),]*self.num_objects, 
                        BoxSet(bmin_ee, bmax_ee)) # multiset: consistent with other sets
    
    def goalSet(self):
        bmin = [-math.pi, -self.max_velocity, -self.max_velocity, -self.max_ang_velocity,]
        bmax = [math.pi, self.max_velocity, self.max_velocity, self.max_ang_velocity,]
        bmin_ee = [-2.5*self.x_range, -2.5*self.y_range, -math.pi,]
        bmax_ee = [2.5*self.x_range, 2.5*self.y_range, math.pi,]
        return MultiSet(*[RingSet([self.start_state[0], self.start_state[1]], 1.0*self.x_range, 1.0*self.x_range+self.offset), # arbitrarily far away
                          BoxSet(bmin, bmax),]*self.num_objects,
                        BoxSet(bmin_ee, bmax_ee)
                        )
        # return self.complementCaptureSet()

    def captureSet(self, default_radius=1000):
        bmin = [-math.pi, -self.max_velocity, -self.max_velocity, -self.max_ang_velocity,]
        bmax = [math.pi, self.max_velocity, self.max_velocity, self.max_ang_velocity,]
        bmin_ee = [-2.5*self.x_range, -2.5*self.y_range, -math.pi,]
        bmax_ee = [2.5*self.x_range, 2.5*self.y_range, math.pi,]
        arcbmin = [-self.offset, -self.offset]
        arcbmax = [self.x_range+self.offset, self.y_range+self.offset]

        # Calculate arc center and radius
        gripper_velocity = np.linalg.norm(np.array([self.gripper_vel_x, self.gripper_vel_y]))
        arc_radius = gripper_velocity / abs(self.gripper_vel_theta) if self.gripper_vel_theta != 0 else default_radius
        if gripper_velocity >= 3e-1:
            # Calculate perpendicular direction to velocity
            perp_direction = [-self.gripper_vel_y, self.gripper_vel_x]
            perp_direction = perp_direction / np.linalg.norm(perp_direction)

            # Calculate arc center
            if self.gripper_vel_theta >= 0:
                arc_center = self.start_state[-3:-1] + perp_direction * arc_radius
            else:
                arc_center = self.start_state[-3:-1] - perp_direction * arc_radius
        else:
            # Handle linear motion case or set a default large radius
            arc_center = self.start_state[-3:-1]
            arc_radius = 0.5 * self.capture_goal_margin
            arc_angle_range = [0.0, 2*np.pi-1e-9]
            return MultiSet(*[CaptureSetClass([arcbmin, arcbmax], self.capture_goal_margin, arc_center, arc_radius, arc_angle_range),
                              BoxSet(bmin, bmax),]*self.num_objects,
                            BoxSet(bmin_ee, bmax_ee),)
        
        # Calculate arc angle range
        angle_to_point = np.arctan2(self.start_state[-2] - arc_center[1], self.start_state[-3] - arc_center[0])

        # Normalize the angle_to_point within the range [0, 2π)
        initial_pos_angle = (angle_to_point + 2 * np.pi) % (2 * np.pi)

        offset = 0.1*self.capture_goal_margin/arc_radius
        if self.gripper_vel_theta > 0:
            # If the gripper is rotating counterclockwise, the arc angle range is [initial_pos_angle, initial_pos_angle + π]
            arc_angle_range = [(initial_pos_angle - offset) % (2*np.pi), 
                               (initial_pos_angle + offset + self.gripper_vel_theta*self.maneuver_goal_tmax) % (2*np.pi)]
        elif self.gripper_vel_theta < 0:
            # If the gripper is rotating clockwise, the arc angle range is [initial_pos_angle - π, initial_pos_angle]
            arc_angle_range = [(initial_pos_angle - offset + self.gripper_vel_theta*self.maneuver_goal_tmax) % (2*np.pi), 
                               (initial_pos_angle + offset) % (2*np.pi),]
        else:
            # If the gripper is not rotating
            arc_angle_range = [(initial_pos_angle-0.15*self.capture_goal_margin/default_radius) % (2*np.pi), 
                               (initial_pos_angle+gripper_velocity*self.maneuver_goal_tmax/default_radius) % (2*np.pi)]

        return MultiSet(*[CaptureSetClass([arcbmin, arcbmax], self.capture_goal_margin, arc_center, arc_radius, arc_angle_range),
                          BoxSet(bmin, bmax),]*self.num_objects,
                        BoxSet(bmin_ee, bmax_ee),)

    def complementCaptureSet(self, default_radius=1000):
        bmin = [-math.pi, -self.max_velocity, -self.max_velocity, -self.max_ang_velocity,]
        bmax = [math.pi, self.max_velocity, self.max_velocity, self.max_ang_velocity,]
        bmin_ee = [-2.5*self.x_range, -2.5*self.y_range, -math.pi,]
        bmax_ee = [2.5*self.x_range, 2.5*self.y_range, math.pi,]
        arcbmin = [-self.offset, -self.offset]
        arcbmax = [self.x_range+self.offset, self.y_range+self.offset]

        # Calculate arc center and radius
        gripper_velocity = np.linalg.norm(np.array([self.gripper_vel_x, self.gripper_vel_y]))
        arc_radius = gripper_velocity / abs(self.gripper_vel_theta) if self.gripper_vel_theta != 0 else default_radius
        if gripper_velocity >= 3e-1:
            # Calculate perpendicular direction to velocity
            perp_direction = [-self.gripper_vel_y, self.gripper_vel_x]
            perp_direction = perp_direction / np.linalg.norm(perp_direction)

            # Calculate arc center
            if self.gripper_vel_theta >= 0:
                arc_center = self.start_state[-3:-1] + perp_direction * arc_radius
            else:
                arc_center = self.start_state[-3:-1] - perp_direction * arc_radius
        else:
            # Handle linear motion case or set a default large radius
            arc_center = self.start_state[-3:-1]
            arc_radius = 0.5 * self.capture_goal_margin
            arc_angle_range = [0.0, 2*np.pi-1e-9]
            return MultiSet(*[ComplementCaptureSetClass([arcbmin, arcbmax], self.capture_goal_margin, arc_center, arc_radius, arc_angle_range),
                              BoxSet(bmin, bmax),]*self.num_objects,
                            BoxSet(bmin_ee, bmax_ee),)
        
        # Calculate arc angle range
        angle_to_point = np.arctan2(self.start_state[-2] - arc_center[1], self.start_state[-3] - arc_center[0])

        # Normalize the angle_to_point within the range [0, 2π)
        initial_pos_angle = (angle_to_point + 2 * np.pi) % (2 * np.pi)

        offset = 0.1*self.capture_goal_margin/arc_radius
        if self.gripper_vel_theta > 0:
            # If the gripper is rotating counterclockwise, the arc angle range is [initial_pos_angle, initial_pos_angle + π]
            arc_angle_range = [(initial_pos_angle - offset) % (2*np.pi), 
                               (initial_pos_angle + offset + self.gripper_vel_theta*self.maneuver_goal_tmax) % (2*np.pi)]
        elif self.gripper_vel_theta < 0:
            # If the gripper is rotating clockwise, the arc angle range is [initial_pos_angle - π, initial_pos_angle]
            arc_angle_range = [(initial_pos_angle - offset + self.gripper_vel_theta*self.maneuver_goal_tmax) % (2*np.pi), 
                               (initial_pos_angle + offset) % (2*np.pi),]
        else:
            # If the gripper is not rotating
            arc_angle_range = [(initial_pos_angle-0.15*self.capture_goal_margin/default_radius) % (2*np.pi), 
                               (initial_pos_angle+gripper_velocity*self.maneuver_goal_tmax/default_radius) % (2*np.pi)]

        return MultiSet(*[ComplementCaptureSetClass([arcbmin, arcbmax], self.capture_goal_margin, arc_center, arc_radius, arc_angle_range),
                          BoxSet(bmin, bmax),]*self.num_objects,
                        BoxSet(bmin_ee, bmax_ee),)

class PlanePushMultiObjectiveFunction(ObjectiveFunction):
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

        xnext = self.space.nextState(x,u)
        self.xnext = xnext

        W = 0
        for i in range(self.cage.num_objects):
            # Calculating change in position and orientation
            delta_x = xnext[2*self.cage.dim_se2*i] - x[2*self.cage.dim_se2*i]
            delta_y = xnext[2*self.cage.dim_se2*i+1] - x[2*self.cage.dim_se2*i+1]
            delta_theta = math.atan2(math.sin(xnext[2*self.cage.dim_se2*i+2] - x[2*self.cage.dim_se2*i+2]), 
                                     math.cos(xnext[2*self.cage.dim_se2*i+2] - x[2*self.cage.dim_se2*i+2]))
            W += abs(m * u[3*i+1] * delta_x + m * u[3*i+2] * delta_y + I * u[3*i+3] * delta_theta)
        
        # Include a small positive value to avoid zero cost in cases where it's needed
        return max(W, 1e-9)


def PlanePushMultiTest(dynamics_sim, 
                data = [1.0, 0.4, 0.0, 0.0, 0.0, 0.0, # object 1, 2,..
                        1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 
                        1.4, 0.7, 0.0, 0.0, 0.0, 0.0, 
                        # 1.4, 0.5, 0.0, 0.0, 0.0, 0.0, 
                        # 1.4, 0.3, 0.0, 0.0, 0.0, 0.0, 
                        # 1.4, 0.9, 0.0, 0.0, 0.0, 0.0, 
                        # 1.4, 1.3, 0.0, 0.0, 0.0, 0.0, 
                        1.0, 0.1, 0.0, 0.0, 1.0, 0.2,], # gripper
                save_hyperparams=False,
                lateral_friction_coef=0.3,
                ):
    p = PlanePushMulti(data, dynamics_sim, save_hyperparams, lateral_friction_coef)
    objective = PlanePushMultiObjectiveFunction(p)

    return PlanningProblem(objective.space,
                           p.startState(),
                           p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean=True,
                           successSet=p.successSet(),
                           complementSuccessSet=p.complementSuccessSet(),
                           captureSet=p.captureSet(),
                           complementCaptureSet=p.complementCaptureSet(),
                           )


