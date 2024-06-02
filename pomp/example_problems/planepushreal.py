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

class PlanePushRealControlSpace(ControlSpace):
    def __init__(self, cage):
        self.cage = cage
        self.dynamics_sim = cage.dynamics_sim
        self.dynamics_sim.set_params(cage.params)
        self.dynamics_sim.create_shapes()
        self.obstacles = self.cage.obstacles
        self.cost_inv_coef = self.cage.cost_inv_coef
        self.is_plane_push = True

    def configurationSpace(self):
        return self.cage.configurationSpace()
    
    def controlSet(self, x):
        return MultiSet(TimeBiasSet(self.cage.time_range,self.cage.controlSet()),self.cage.controlSet())
    
    def nextState(self, x, u, is_planner=False):
        return self.eval(x, u, 1.0, is_planner=is_planner)
    
    def eval(self, x, u, amount, print_via_points=False, is_planner=False):
        # x_i, y_i, theta_i, vx_i, vy_i, alpha_i, xr_i, yr_i, thetar_i = x # state space, 9D (3+3: cage, 3: robot gripper)
        t, ax, ay, omega = u # control space
        tc = t * amount
        mu = [tc, ax, ay, omega]

        xaug = x if is_planner else (x + self.cage.gripper_vel)
        self.dynamics_sim.reset_states(xaug)
        x_new, xo_via_points = self.dynamics_sim.run_forward_sim(mu, 1)

        # Make theta fall in [-pi, pi]
        x_new[2] = limit_angle_to_pi(x_new[2])
        x_new[8] = limit_angle_to_pi(x_new[8])

        if print_via_points: # TODO
            self.xo_via_points = [[q[0], q[1]] for q in xo_via_points]

        if is_planner:
            return x_new
        else:
            return x_new[:9]
    
    def interpolator(self, x, u, xnext=None):
        return LambdaInterpolator(lambda s:self.eval(x,u,s), self.configurationSpace(), 10, xnext=xnext)

class PlanePushReal:
    def __init__(self, data, dynamics_sim, save_hyperparams=False, quasistatic_motion=0,):
        self.nx = 9 # state space dimension
        self.nu = 4 # control space dimension
        self.dynamics_sim = dynamics_sim
        self.x_range = 10
        self.y_range = 10
        self.offset = 5 # extend the landscape
        self.scale_factor = 10
        self.max_velocity = 10
        self.max_ang_velocity = 10 # 2
        self.lateral_friction_coef = 27/68 # mu_static
        self.max_acceleration = 2.5
        self.max_ang_acceleration = .5
        self.y_obstacle = -0.15 * self.scale_factor # the lower rim y_pos of the obstacle
        self.obstacle_borderline = [[-self.offset, self.y_obstacle], [self.x_range+self.offset, self.y_obstacle]]
        self.angle_slope = 0.0 * math.pi  # equivalent to on a slope
        self.cost_inv_coef = -5e-2

        self.object_name = 'convex' # 'rectangle', 'convex', 'concave', 'irregular', 'triangle'
        self.gripper_name = 'jaw' # 'circle', 'jaw'
        self.mass_bodies = {'rectangle': 0.048, 'convex': 0.042, 'concave': 0.043, 'irregular': 0.024, 'triangle': 0.028, 'circle': 0.026, 'jaw': 0.026}
        self.mass_object = self.mass_bodies[self.object_name] * self.scale_factor**3
        self.mass_gripper = self.mass_bodies[self.gripper_name] * self.scale_factor**3 + 100 # robot arm weight added
        factor_object = 1e-1 # approximately
        factor_gripper = 1e-1
        self.moment_object = self.mass_object * factor_object # moment of inertia
        self.moment_gripper = self.mass_gripper * factor_gripper

        self.task_goal_margins = {'rectangle': 0.05, 'convex': 0.05, 'concave': 0.05, 'irregular': 0.15, 'triangle': 0.10} # after 10-time scaling
        self.task_goal_margin = self.task_goal_margins[self.object_name]
        self.object_thicknesses = {'rectangle': 0.35, 'convex': 0.20, 'concave': 0.20, 'irregular': 0.20, 'triangle': 0.20} # after 10-time scaling
        self.object_thickness = self.object_thicknesses[self.object_name]
        self.maneuver_goal_margin = .5
        self.maneuver_goal_tmax = 1.5

        self.params = [self.mass_object, self.moment_object, self.mass_gripper, self.moment_gripper, self.y_obstacle, self.angle_slope,
                       self.object_name, self.gripper_name, self.lateral_friction_coef, self.scale_factor]
        self.c_space_boundary = [[0, self.x_range], [0, self.y_range], [-0.8*math.pi, 0.8*math.pi], 
                                 [-0.8*self.max_velocity, 0.8*self.max_velocity], [-0.8*self.max_velocity, 0.8*self.max_velocity], [-0.8*self.max_ang_velocity, 0.8*self.max_ang_velocity], 
                                 [-self.x_range, self.x_range], [-self.y_range, self.y_range], [-0.8*math.pi, 0.8*math.pi],
                                 [-0.8*self.max_velocity, 0.8*self.max_velocity], [-0.8*self.max_velocity, 0.8*self.max_velocity], [-0.8*self.max_ang_velocity, 0.8*self.max_ang_velocity], 
                                 ] # shrink the c-space a bit for MPPI
        
        # Gripper moving velocity (constant)
        if not quasistatic_motion:
            self.gripper_vel = data[9:]
            self.gripper_vel_x = data[9]
            self.gripper_vel_y = data[10]
            self.gripper_vel_theta = data[11]
        else:
            self.gripper_vel = [0, 0, 0]
            self.gripper_vel_x = 0
            self.gripper_vel_y = 0
            self.gripper_vel_theta = 0

        self.start_state = data[:9]
        self.time_range = .3
        self.obstacles = []
        self.gravity = -9.81

        self.hyperparams = [self.x_range, self.y_range, self.offset, self.max_velocity, self.max_ang_velocity, self.max_acceleration, 
                            self.max_ang_acceleration, self.time_range, self.gravity, self.task_goal_margin, self.maneuver_goal_margin,
                            self.maneuver_goal_tmax, self.cost_inv_coef, self.object_thickness] + self.params
        self.hyperparams_header = ['x_range', 'y_range', 'offset', 'max_velocity', 'max_ang_velocity', 'max_acceleration',
                                   'max_ang_acceleration', 'time_range', 'gravity', 'task_goal_margin', 'maneuver_goal_margin', 'maneuver_goal_tmax',
                                   'cost_inv_coef', 'object_thickness',
                                   'mass_object', 'moment_object', 'mass_gripper', 'moment_gripper', 'y_obstacle', 'angle_slope',
                                   'object_name', 'gripper_name', 'lateral_friction_coef']
        if save_hyperparams:
            self.saveHyperparams()

    def saveHyperparams(self, filename='push_realworld_hyperparams.csv'):
        with open(filename, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for header, data in zip(self.hyperparams_header, self.hyperparams):
                csv_writer.writerow([header, data])

    def controlSet(self):
        return BoxSet([-self.max_acceleration, -self.max_acceleration, -self.max_ang_acceleration], 
                      [self.max_acceleration, self.max_acceleration, self.max_ang_acceleration])

    def controlSpace(self):
        # System dynamics
        return PlanePushRealControlSpace(self)

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

    def successSet(self): # TODO
        wsbmin = [-self.offset, self.y_obstacle+self.object_thickness-self.task_goal_margin,]
        wsbmax = [self.x_range+self.offset, self.y_obstacle+self.object_thickness+self.task_goal_margin,]
        bmin = [-math.pi, -self.max_velocity, -self.max_velocity, -self.max_ang_velocity, -2.5*self.x_range, -2.5*self.y_range, -math.pi]
        bmax = [math.pi, self.max_velocity, self.max_velocity, self.max_ang_velocity, 2.5*self.x_range, 2.5*self.y_range, math.pi]
        return MultiSet(BoxSet(wsbmin, wsbmax), BoxSet(bmin, bmax)) # multiset: consistent with other sets

    def goalSet(self):
        bmin = [-math.pi, -self.max_velocity, -self.max_velocity, -self.max_ang_velocity, -2.5*self.x_range, -2.5*self.y_range, -math.pi]
        bmax = [math.pi, self.max_velocity, self.max_velocity, self.max_ang_velocity, 2.5*self.x_range, 2.5*self.y_range, math.pi]
        return MultiSet(RingSet([self.start_state[0], self.start_state[1]], 1.0*self.x_range, 1.0*self.x_range+self.offset), # arbitrarily far away
                        BoxSet(bmin, bmax))
        # return self.complementCaptureSet()
    
    def complementCaptureSet(self, default_radius=1000):
        bmin = [-math.pi, -self.max_velocity, -self.max_velocity, -self.max_ang_velocity, -2.5*self.x_range, -2.5*self.y_range, -math.pi]
        bmax = [math.pi, self.max_velocity, self.max_velocity, self.max_ang_velocity, 2.5*self.x_range, 2.5*self.y_range, math.pi]
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
                arc_center = self.start_state[6:8] + perp_direction * arc_radius
            else:
                arc_center = self.start_state[6:8] - perp_direction * arc_radius
        else:
            # Handle linear motion case or set a default large radius
            arc_center = self.start_state[6:8]
            arc_radius = 1 * self.maneuver_goal_margin
            arc_angle_range = [0.0, 2*np.pi-1e-9]
            return MultiSet(ComplementCaptureSetClass([arcbmin, arcbmax], self.maneuver_goal_margin, arc_center, arc_radius, arc_angle_range),
                            BoxSet(bmin, bmax))
        
        # Calculate arc angle range
        angle_to_point = np.arctan2(self.start_state[7] - arc_center[1], self.start_state[6] - arc_center[0])

        # Normalize the angle_to_point within the range [0, 2π)
        initial_pos_angle = (angle_to_point + 2 * np.pi) % (2 * np.pi)

        offset = 0.1*self.maneuver_goal_margin/arc_radius
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
            arc_angle_range = [(initial_pos_angle-0.15*self.maneuver_goal_margin/default_radius) % (2*np.pi), 
                               (initial_pos_angle+gripper_velocity*self.maneuver_goal_tmax/default_radius) % (2*np.pi)]

        return MultiSet(ComplementCaptureSetClass([arcbmin, arcbmax], self.maneuver_goal_margin, arc_center, arc_radius, arc_angle_range),
                        BoxSet(bmin, bmax))


class PlanePushRealObjectiveFunction(ObjectiveFunction):
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

        # Calculating change in position and orientation
        delta_x = xnext[0] - x[0]
        delta_y = xnext[1] - x[1]
        delta_theta = math.atan2(math.sin(xnext[2] - x[2]), math.cos(xnext[2] - x[2]))  # More robust angle difference

        # Work done by forces and torque
        W = m * u[1] * delta_x + m * u[2] * delta_y + I * u[3] * delta_theta

        # Considering both positive and negative work
        c = abs(W)
        
        # Include a small positive value to avoid zero cost in cases where it's needed
        return max(c, 1e-5)


def PlanePushRealTest(dynamics_sim, 
                data = [5.0, 4.3, 0.0, 0.0, 0.0, 0, # point gripper with cylinder/box object
                        5.0, 4, 0.0, 0.0, 1.0, 0.0],
                save_hyperparams=False,
                ):
    p = PlanePushReal(data, dynamics_sim, save_hyperparams)
    objective = PlanePushRealObjectiveFunction(p)
    return PlanningProblem(objective.space,
                           p.startState(),
                           p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean=True,
                           successSet=p.successSet(),
                           complementCaptureSet=p.complementCaptureSet(),
                           )


