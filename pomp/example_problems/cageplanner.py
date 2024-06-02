from OpenGL.GL import *
from .geometric import *
from ..spaces.objective import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..planners.problem import PlanningProblem
from ..bullet.forwardsimulator import *
import math
import joblib
# from tensorflow.keras.models import load_model

# TODO: 1. Discontinuity of movement in Pybullet replay
# 2. Cage-based robustness penalty term in the objective function
# 3. Continuous visualization in OpenGL
# 4. scattered nodes and edges in the graph of OpenGL

class CagePlannerControlSpace(ControlSpace):
    def __init__(self,cage):
        self.cage = cage
        self.dynamics_sim = forwardSimulation(cage.params, gui=0)
        self.is_cage_planner = True
        self.half_extents_gripper = cage.half_extents_gripper # [x,z]
        self.obstacle_pose = cage.start_state[4:7] # xc, zc, theta
        self.xo_via_points = None

    def configurationSpace(self):
        return self.cage.configurationSpace()
    
    def controlSet(self,x):
        return MultiSet(TimeBiasSet(self.cage.time_range,self.cage.controlSet()),self.cage.controlSet())
    def nextState(self,x,u):
        return self.eval(x,u,1.0)
    
    def toBulletStateInput(self, x, u=None):
        # OpenGL (O) Cartesian coordiantes are different from Bullet (B)
        # O--->---------
        # |             | 
        # \/     *      | 
        # |    =====    | 
        # /\            | 
        #  |            | 
        # B--->---------
        q = [x[0],self.cage.y_range-x[1],
             x[2],-x[3],
             x[4],self.cage.y_range-x[5],-x[6],
             x[7],-x[8],-x[9]]
        if u is not None:
            mu = [u[0],u[1],-u[2],-u[3]]
        else:
            mu = None
        return q, mu
    
    def toOpenglStateInput(self, q):
        x = [q[0],self.cage.y_range-q[1],
             q[2],-q[3],
             q[4],self.cage.y_range-q[5],-q[6],
             q[7],-q[8],-q[9]]
        return x
        
    def check_state_feasibility(self, x, max_distance=-0.007):
        """Check if the state indicates a collision between the object and the gripper."""
        q, _ = self.toBulletStateInput(x)
        self.dynamics_sim.reset_states(q)
        obj = self.dynamics_sim.objectUid
        grip = self.dynamics_sim.gripperUid
        is_feasible = (len(p.getClosestPoints(bodyA=obj, bodyB=grip, distance=max_distance)) == 0)

        return is_feasible

    def eval(self,x,u,amount,print_via_points=False):
        """amount: float within [0,1], scale the duration for interpolator."""
        # xo,yo,vox,voy,xg,yg,thetag,vgx,vgy,omegag = x # state space, 10D (4: cage, 6: gripper)
        t,thrust_x,thrust_y,alpha = u # control space, 4D
        tc = t*amount
        u = [tc,thrust_x,thrust_y,alpha]
        q, mu = self.toBulletStateInput(x, u)
        self.dynamics_sim.reset_states(q)
        q_new, qo_via_points = self.dynamics_sim.run_forward_sim(mu, print_via_points)
        x_new = self.toOpenglStateInput(q_new)

        if print_via_points:
            self.xo_via_points = [[q[0], self.cage.y_range-q[1]] for q in qo_via_points]

        return x_new
    
    def interpolator(self,x,u):
        return LambdaInterpolator(lambda s:self.eval(x,u,s),self.configurationSpace(),10)

class CagePlanner:
    def __init__(self):
        self.x_range = 10
        self.y_range = 10
        self.max_velocity = 10
        self.max_acceleration = 10

        # Parameters passing to Pybullet
        self.mass_object = 1
        self.mass_gripper = 10
        self.moment_gripper = 1 # moment of inertia
        self.half_extents_gripper = [.7, .4] # movement on x-z plane
        self.radius_object = 0.01
        self.params = [self.mass_object, self.mass_gripper, self.moment_gripper, 
                       self.half_extents_gripper, self.radius_object]
        
        xo_init = 8
        yo_init = 2
        xo_goal = 3
        yo_goal = 7
        self.start_state = [xo_init,yo_init,0,0,xo_init,yo_init+self.radius_object+self.half_extents_gripper[1],0,0,0,0]
        self.goal_state = [xo_goal,yo_goal,0,0,0,0,0,0,0,0]
        self.goal_radius = 1
        self.time_range = 1

        self.obstacles = []
        self.gravity = 9.81 # downward in openGL vis

        self.c_space_boundary = [[0.0,self.x_range], [0.0,self.y_range],
                                 [-3.0,3.0], [-3.0,3.0],
                                 [0.0,self.x_range], [0.0,self.x_range], [-math.pi/2,math.pi/2],
                                 [-3.0,3.0], [-3.0,3.0],[-math.pi/9,math.pi/9]]
        # self.u_boundary = [[0.0, -0.3, -self.gravity-5.0, -math.pi/27], 
        #                    [self.time_range, 0.3, -self.gravity+1.0, math.pi/27]] # parameter for initial dataset generation
        self.u_boundary = [[0.0, -0.9, -self.gravity-3.0, -math.pi/6], 
                           [self.time_range, 0.9, -self.gravity+3.0, math.pi/6]]

    def controlSet(self):
        return BoxSet(self.u_boundary[0][1:], 
                      self.u_boundary[1][1:])

    def controlSpace(self):
        # System dynamics
        return CagePlannerControlSpace(self)

    def workspace(self):
        # For visualization
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [0,0]
        wspace.box.bmax = [self.x_range,self.y_range]
        wspace.addObstacleParam(self.obstacles)
        for o in self.obstacles:
            wspace.addObstacle(Box(o[0],o[1],o[0]+o[2],o[1]+o[3]))
        return wspace
    
    def configurationSpace(self):
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [0,0]
        wspace.box.bmax = [self.x_range,self.y_range]
        wspace.addObstacleParam(self.obstacles)
        for o in self.obstacles:
            wspace.addObstacle(Box(o[0],o[1],o[0]+o[2],o[1]+o[3]))
        cbd = self.c_space_boundary
        res =  MultiConfigurationSpace(wspace,
                                       BoxConfigurationSpace([cbd[2][0]],[cbd[2][1]]), 
                                       BoxConfigurationSpace([cbd[3][0]],[cbd[3][1]]),
                                       BoxConfigurationSpace([cbd[4][0]],[cbd[4][1]]),
                                       BoxConfigurationSpace([cbd[5][0]],[cbd[5][1]]),
                                       BoxConfigurationSpace([cbd[6][0]],[cbd[6][1]]), 
                                       BoxConfigurationSpace([cbd[7][0]],[cbd[7][1]]), 
                                       BoxConfigurationSpace([cbd[8][0]],[cbd[8][1]]), 
                                       BoxConfigurationSpace([cbd[9][0]],[cbd[9][1]]), 
                                       )
        return res

    def startState(self):
        return self.start_state

    def goalSet(self):
        r = self.goal_radius
        return BoxSet([self.goal_state[0]-r, self.goal_state[1]-r,
                       -self.max_velocity, -self.max_velocity, 
                       0.0, 0.0, -math.pi,
                       -self.max_velocity, -self.max_velocity, -self.max_velocity],
                      [self.goal_state[0]+r, self.goal_state[1]+r,
                       self.max_velocity, self.max_velocity, 
                       self.x_range, self.y_range, math.pi,
                       self.max_velocity, self.max_velocity, self.max_velocity])


class CagePlannerObjectiveFunction(ObjectiveFunction):
    """Given a function pointwise(x,u), produces the incremental cost
    by incrementing over the interpolator's length.
    """
    def __init__(self,cage,timestep=0.2):
        self.cage = cage
        self.space = cage.controlSpace()
        self.timestep = timestep
        self.masso = cage.params[0]
        self.massg = cage.params[1]
        self.momentg = cage.params[2]
        # self.model = load_model('data/planar-gripper-dynamic-cage-dataset/model_varyingGoal_cutoffLabels.h5')
        # self.scaler = joblib.load('data/planar-gripper-dynamic-cage-dataset/scaler_minmax.pkl')
        
    def incremental(self,x,u):
        xnext = self.space.nextState(x,u)
        # g = self.cage.gravity

        # Instability cost (inverse of escape energy)
        # Normalize the data
        x_tran = self.scaler.transform([xnext])
        cage_metric = self.model.predict(x_tran, verbose=0)
        c0 = 1/(1+max(cage_metric,1e-4))

        # Distance from goal region
        xo_goal = self.cage.goal_state[:2]
        xo = x[:2]
        xo_next = xnext[:2]
        # dis = math.sqrt(sum([(xo_goal[i]-xo[i])**2 for i in range(len(xo))]))
        dis_next = math.sqrt(sum([(xo_goal[i]-xo_next[i])**2 for i in range(len(xo))]))
        # c1 = max(dis_next-dis, 0.01)
        c1 = dis_next

        # # Object and gripper total energy (kinetic and potential)
        # E_o = self.masso * (g*(self.cage.y_range-x[1]) + 0.5*(x[2]**2+x[3]**2))
        # Enext_o = self.masso * (g*(self.cage.y_range-xnext[1]) + 0.5*(xnext[2]**2+xnext[3]**2))
        # E_g = g*self.massg*(self.cage.y_range-x[5]) + 0.5*(self.massg*(x[7]**2+x[8]**2)+self.momentg*(x[9]**2))
        # Enext_g = g*self.massg*(self.cage.y_range-xnext[5]) + 0.5*(self.massg*(xnext[7]**2+xnext[8]**2)+self.momentg*(x[9]**2))
        # c2 = max((Enext_g+Enext_o-E_o-E_g), 0.0)

        # Time penalty
        # return 10*c1 + 0.001*c2 + u[0]
        # return c1 + 0.001*u[0]
        # return c1 + 0.1*u[0]
        return c1 + 10.0*c0


def cagePlannerTest():
    p = CagePlanner()
    objective = CagePlannerObjectiveFunction(p)
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean = True)


