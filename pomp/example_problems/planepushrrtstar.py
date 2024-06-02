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

class PlanePushRrtstar:
    def __init__(self, data, dynamics_sim, save_hyperparams=0):
        self.dynamics_sim = dynamics_sim
        self.x_range = 10
        self.y_range = 10
        self.offset = 2.0 # extend the landscape
        self.y_obstacle = 9
        self.obstacle_borderline = [[-self.offset,self.y_obstacle], [self.x_range+self.offset, self.y_obstacle]]
        self.start_state = data[:3]
        self.gripper_pose = data[6:9]
        self.gripper_vel = data[9:]
        self.gripper_vel_x = data[9]
        self.gripper_vel_y = data[10]
        self.gripper_vel_theta = data[11]
        self.lateral_friction_coef = 0.2
        self.cost_inv_coef = 3

        self.object_name = 'box' # 'box', 'cylinder'
        self.gripper_name = 'cylinder' # 'box', 'cylinder', 'bowl'
        self.mass_object = 1
        self.mass_gripper = 4
        factor_object = 1e-1 if self.object_name == 'box' else 1e-3
        factor_gripper = 1e-1 if (self.gripper_name == 'box' or self.gripper_name == 'bowl') else 1e-3
        self.moment_object = self.mass_object * factor_object # moment of inertia
        self.moment_gripper = self.mass_gripper * factor_gripper

        self.angle_slope = 0 * math.pi  # equivalent to on a slope
        self.params = [self.mass_object, self.moment_object, self.mass_gripper, self.moment_gripper, self.y_obstacle, self.angle_slope,
                       self.object_name, self.gripper_name, self.lateral_friction_coef]
        self.maneuver_goal_margin = .7
        self.maneuver_goal_tmax = 3.0
        self.dynamics_sim.set_params(self.params)
        self.dynamics_sim.create_shapes()
        self.dynamics_sim.set_gripper_pos(self.gripper_pose)

        self.obstacles = []
        self.gravity = -9.81

        self.hyperparams = [self.x_range, self.y_range, self.offset, self.gravity, self.maneuver_goal_margin,
                            self.maneuver_goal_tmax] + self.params
        self.hyperparams_header = ['x_range', 'y_range', 'offset', 'gravity', 'maneuver_goal_margin', 'maneuver_goal_tmax',
                                   'mass_object', 'moment_object', 'mass_gripper', 'moment_gripper', 'y_obstacle', 'angle_slope',
                                   'object_name', 'gripper_name', 'lateral_friction_coef']
        
        if save_hyperparams:
            self.saveHyperparams()

    def saveHyperparams(self, filename='push_press_hyperparams.csv'):
        with open(filename, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for header, data in zip(self.hyperparams_header, self.hyperparams):
                csv_writer.writerow([header, data])

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
        res = RRTstarMultiConfigurationSpace((wspace, BoxConfigurationSpace([-math.pi],[math.pi])), self.dynamics_sim)
        res.setObstacle(self.obstacle_borderline)
        return res

    def startState(self):
        return self.start_state

    def goalSet(self, default_radius=1000):
        bmin = [-math.pi,]
        bmax = [math.pi,]
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
                arc_center = self.gripper_pose[:2] + perp_direction * arc_radius
            else:
                arc_center = self.gripper_pose[:2] - perp_direction * arc_radius
        else:
            # Handle linear motion case or set a default large radius
            arc_center = self.gripper_pose[:2]
            arc_radius = self.maneuver_goal_margin
            arc_angle_range = [0.0, 2*np.pi-1e-9]
            return MultiSet(ComplementCaptureSetClass([arcbmin, arcbmax], self.maneuver_goal_margin, arc_center, arc_radius, arc_angle_range),
                            BoxSet(bmin, bmax))
        
        # Calculate arc angle range
        angle_to_point = np.arctan2(self.gripper_pose[1] - arc_center[1], self.gripper_pose[0] - arc_center[0])

        # Normalize the angle_to_point within the range [0, 2π)
        initial_pos_angle = (angle_to_point + 2 * np.pi) % (2 * np.pi)

        if self.gripper_vel_theta > 0:
            # If the gripper is rotating counterclockwise
            arc_angle_range = [(initial_pos_angle - self.maneuver_goal_margin/arc_radius) % (2*np.pi), 
                               (initial_pos_angle + self.gripper_vel_theta*self.maneuver_goal_tmax) % (2*np.pi)]
        elif self.gripper_vel_theta < 0:
            # If the gripper is rotating clockwise
            arc_angle_range = [(initial_pos_angle + self.gripper_vel_theta*self.maneuver_goal_tmax) % (2*np.pi), 
                               (initial_pos_angle + self.maneuver_goal_margin/arc_radius) % (2*np.pi),]
        else:
            # If the gripper is not rotating
            arc_angle_range = [(initial_pos_angle-self.maneuver_goal_margin/default_radius) % (2*np.pi), 
                               (initial_pos_angle+gripper_velocity*self.maneuver_goal_tmax/default_radius) % (2*np.pi)]

        return MultiSet(ComplementCaptureSetClass([arcbmin, arcbmax], self.maneuver_goal_margin, arc_center, arc_radius, arc_angle_range),
                        BoxSet(bmin, bmax))
        # return MultiSet(BoxSet([0,0], [10,1]),
        #                 BoxSet(bmin, bmax))
    
    def potentialMetric(self, a, b, is_potential_cost=1):
        """Potential-based metric for RRT* - edge cost from a to b"""
        if is_potential_cost:
            c = (self.mass_object * abs(self.gravity) * math.sin(self.angle_slope) * (b[1]-a[1]) 
                 + euclideanMetric(a, b) * self.lateral_friction_coef * abs(self.gravity) * self.mass_object * math.cos(self.angle_slope))
        else:
            c = euclideanMetric(a, b) # path length cost
        return max(c, 1e-5)

def PlanePushRrtstarTest(dynamics_sim, 
                        data = [5.0, 4.3, 0.0, 0.0, 0.0, 0, # point gripper with cylinder/box object
                                5.0, 4, 0.0, 0.0, 1.0, 0.0],
                        # data = [5.0, 4, 0.0, 0.0, 0, 0, # bowl gripper with cylinder object
                        #         5.0, 4, 0.0, 0.0, 1, 0.0],
                        save_hyperparams=0,
                        ):
    p = PlanePushRrtstar(data, dynamics_sim, save_hyperparams)
    return PlanningProblem(p.configurationSpace(), # for geometric planner - rrt*
                           p.startState(),
                           p.goalSet(),
                           objective=p.potentialMetric,
                           visualizer=p.workspace(),
                           euclidean=True)


