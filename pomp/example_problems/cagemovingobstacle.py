from OpenGL.GL import *
from .geometric import *
from ..spaces.objective import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..planners.problem import PlanningProblem

class CageMOControlSpace(ControlSpace):
    def __init__(self,cage):
        self.cage = cage
        self.is_moving_obstacle = True
    def configurationSpace(self):
        return self.cage.configurationSpace()
    def controlSet(self,x):
        return MultiSet(TimeBiasSet(self.cage.time_range,self.cage.controlSet()),self.cage.controlSet())
    def nextState(self,x,u):
        return self.eval(x,u,1.0)
    def eval(self,x,u,amount):
        x_i,y_i,vx_i,vy_i,xr_i,yr_i = x # state space, 6D (4: cage, 2: robot gripper)
        t,thrust_x,thrust_y = u # control space
        tc = t*amount
        net_acceler_x = thrust_x
        net_acceler_y = self.cage.gravity + thrust_y
        return [x_i+vx_i*tc+0.5*net_acceler_x*(tc**2), 
                y_i+vy_i*tc+0.5*net_acceler_y*(tc**2), 
                vx_i+net_acceler_x*tc,
                vy_i+net_acceler_y*tc,
                xr_i+self.cage.gripper_vel_x*tc,
                yr_i+self.cage.gripper_vel_y*tc,
                ]
    
    def interpolator(self,x,u):
        return LambdaInterpolator(lambda s:self.eval(x,u,s),self.configurationSpace(),10)

class CageMO:
    def __init__(self):
        self.x_range = 1000
        self.y_range = 1000
        self.max_velocity = 40
        self.max_acceleration = 20

        # Gripper moving velocity (constant)
        self.gripper_vel_x = 0.0
        self.gripper_vel_y = -6.0

        self.start_state = [450, 390, 0, self.gripper_vel_y, 0, 0]
        self.goal_state = [950, 100, 0, 0, 0, 0]
        self.goal_radius = 50
        self.time_range = 10

        self.obstacles = [
            #  (375, 200, 50, 200), 
            #  (575, 200, 50, 200), 
             (375, 395, 120, 70), 
             ]
        self.gravity = -9.81

    def controlSet(self):
        return BoxSet([-self.max_acceleration, -self.max_acceleration], 
                      [self.max_acceleration, self.max_acceleration])

    def controlSpace(self):
        # System dynamics
        return CageMOControlSpace(self)

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
        res =  MultiConfigurationSpace(wspace,
                                       BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]), 
                                       BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]),
                                       BoxConfigurationSpace([-self.x_range/2],[self.x_range/2]),
                                       BoxConfigurationSpace([-self.y_range/2],[self.y_range/2])
                                       )
        return res

    def startState(self):
        return self.start_state

    def goalSet(self):
        r = self.goal_radius
        return BoxSet([self.goal_state[0]-r, self.goal_state[1]-r,
                       self.goal_state[2]-self.max_velocity, self.goal_state[3]-self.max_velocity, 
                       self.goal_state[4]-self.x_range/2, self.goal_state[5]-self.y_range/2],
                      [self.goal_state[0]+r, self.goal_state[1]+r,
                       self.goal_state[2]+self.max_velocity, self.goal_state[3]+self.max_velocity, 
                       self.goal_state[4]+self.x_range/2, self.goal_state[5]+self.y_range/2])


class CageMOObjectiveFunction(ObjectiveFunction):
    """Given a function pointwise(x,u), produces the incremental cost
    by incrementing over the interpolator's length.
    """
    def __init__(self,cage,timestep=0.2):
        self.cage = cage
        self.space = cage.controlSpace()
        self.timestep = timestep

    def incremental(self,x,u):
        # Energy E_k+E_g total increase cost (BUG: root node is asked to be pruned without max)
        xnext = self.space.nextState(x,u)
        E = abs(self.cage.gravity)*(self.cage.y_range-x[1]) + 0.5*(x[2]**2+x[3]**2)
        Enext = abs(self.cage.gravity)*(self.cage.y_range-xnext[1]) + 0.5*(xnext[2]**2+xnext[3]**2)
        c = max((Enext-E), 0.001)

        return c


def cageMOTest():
    p = CageMO()
    objective = CageMOObjectiveFunction(p)
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean = True)


