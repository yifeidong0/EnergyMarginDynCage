from OpenGL.GL import *
from .geometric import *
from ..spaces.objective import *
from ..spaces.statespace import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..planners.problem import PlanningProblem

class CageControlSpace(ControlSpace):
    def __init__(self,cage):
        self.cage = cage
    def configurationSpace(self):
        return self.cage.configurationSpace()
    def controlSet(self,x):
        return MultiSet(TimeBiasSet(self.cage.time_range,self.cage.controlSet()),self.cage.controlSet())
        #return MultiSet(BoxSet([0],[self.cage.time_range]),self.cage.controlSet())
    def nextState(self,x,u):
        return self.eval(x,u,1.0)
    def eval(self,x,u,amount):
        x_i,y_i,vx_i,vy_i = x # state space
        t,thrust_x,thrust_y = u # control space
        tc = t*amount
        net_acceler_x = thrust_x
        net_acceler_y = self.cage.gravity + thrust_y
        return [x_i+vx_i*tc+0.5*net_acceler_x*(tc**2), 
                y_i+vy_i*tc+0.5*net_acceler_y*(tc**2), 
                vx_i+net_acceler_x*tc,
                vy_i+net_acceler_y*tc,
                ]
    
    def interpolator(self,x,u):
        return LambdaInterpolator(lambda s:self.eval(x,u,s),self.configurationSpace(),10)

class Cage:
    def __init__(self):
        self.x_range = 1000
        self.y_range = 1000
        # self.min_altitude = 300
        self.max_velocity = 40
        self.max_acceleration = 20

        self.start_state = [450, 350, 0, 0]
        self.goal_state = [950, 900, 0, 0]
        self.goal_radius = 50
        self.time_range = 10
        #u = lambda:round(random.random())

        self.obstacles = []
        self.obstacles = [
            #  (175, 450, 50, 100), (175, 0, 50, 100), (175, 150, 50, 200), 
             (375, 200, 50, 200), 
             (575, 200, 50, 200), 
             (375, 400, 250, 50), 
            #  (775, 200, 50, 400)
             ]
        self.v_x = 5
        self.gravity = -9.8
        # self.thrust = 4

    def controlSet(self):
        # return FiniteSet([[0],[1]])
        return BoxSet([-self.max_acceleration, -self.max_acceleration], 
                      [self.max_acceleration, self.max_acceleration])

    def controlSpace(self):
        # System dynamics
        return CageControlSpace(self)

    def workspace(self): 
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [0,0]
        wspace.box.bmax = [self.x_range,self.y_range]
        for o in self.obstacles:
            wspace.addObstacle(Box(o[0],o[1],o[0]+o[2],o[1]+o[3]))
        return wspace
    
    def configurationSpace(self):
        wspace = Geometric2DCSpace()
        wspace.box.bmin = [0,0]
        wspace.box.bmax = [self.x_range,self.y_range]
        for o in self.obstacles:
            wspace.addObstacle(Box(o[0],o[1],o[0]+o[2],o[1]+o[3]))
        res =  MultiConfigurationSpace(wspace,
                                       BoxConfigurationSpace([-self.max_velocity],[self.max_velocity]), 
                                       BoxConfigurationSpace([-self.max_velocity],[self.max_velocity])
                                       )
        return res

    def startState(self):
        return self.start_state

    def goalSet(self):
        r = self.goal_radius
        return BoxSet([self.goal_state[0]-r,self.goal_state[1]-r,-self.max_velocity,-self.max_velocity],
                      [self.goal_state[0]+r,self.goal_state[1]+r,self.max_velocity,self.max_velocity])


class CageObjectiveFunction(ObjectiveFunction):
    """Given a function pointwise(x,u), produces the incremental cost
    by incrementing over the interpolator's length.
    """
    def __init__(self,cage,timestep=0.2):
        self.cage = cage
        self.space = cage.controlSpace()
        self.timestep = timestep
    def incremental(self,x,u):
        e = self.space.interpolator(x,u)
        tmax = u[0]
        t = 0
        c = 0
        
        # Energy E_k+E_g total increase cost (BUG: root node is asked to be pruned without max)
        xnext = self.space.nextState(x,u)
        E = -self.cage.gravity*(self.cage.y_range-x[1]) + 0.5*(x[2]**2+x[3]**2)
        Enext = -self.cage.gravity*(self.cage.y_range-xnext[1]) + 0.5*(xnext[2]**2+xnext[3]**2)
        c = max((Enext-E), 0.0)

        # Energy Fs cost
        # xnext = self.space.nextState(x,u)
        # c = (xnext[0]-x[0])*u[1] + (xnext[1]-x[1])*u[2]

        # # Distance cost
        # while t < tmax:
        #     t = min(tmax,t+self.timestep)
        #     xnext = e.eval(t / tmax)
        #     c += vectorops.distance(x,xnext)
        #     x = xnext
        #     print(x)

        # # delte_v quadratic cost
        # c = sum([(u[i]*u[0])**2 for i in range(1, len(u))]) # (v1x-v2x)^2+(v1y-v2y)^2

        return c


def cageTest():
    p = Cage()
    objective = CageObjectiveFunction(p)
    return PlanningProblem(p.controlSpace(),p.startState(),p.goalSet(),
                           objective=objective,
                           visualizer=p.workspace(),
                           euclidean = True)


