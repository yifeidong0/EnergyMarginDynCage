from OpenGL.GL import *
from ..klampt import gldraw
from ..spaces.objectives import *
from ..spaces.configurationspace import *
from ..spaces.edgechecker import *
from ..spaces.metric import *
from ..spaces.configurationspace import NeighborhoodSubset,SingletonSubset
from ..planners.problem import PlanningProblem

class Circle:
    def __init__(self,x=0,y=0,radius=1):
        self.center = (x,y)
        self.radius = radius
        
    def contains(self,point):
        return (vectorops.distance(point,self.center) <= self.radius)

    def signedDistance(self,point):
        return (vectorops.distance(point,self.center) - self.radius)
    
    def signedDistance_gradient(self,point):
        d = vectorops.sub(point,self.center)
        return vectorops.div(d,vectorops.norm(d))

    def drawGL(self,res=0.01):
        numdivs = int(math.ceil(self.radius*math.pi*2/res))
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(*self.center)
        for i in range(numdivs+1):
            u = float(i)/float(numdivs)*math.pi*2
            glVertex2f(self.center[0]+self.radius*math.cos(u),self.center[1]+self.radius*math.sin(u))
        glEnd()


class Box(BoxSet):
    def __init__(self,x1=0,y1=0,x2=0,y2=0):
        BoxSet.__init__(self,[min(x1,x2),min(y1,y2)],[max(x1,x2),max(y1,y2)])
        
    def drawGL(self):
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set color with alpha for transparency (RGBA)
        glColor4f(0, 1, 0, 0.2)  # White color with 50% transparency

        glBegin(GL_QUADS)
        glVertex2f(*self.bmin)
        glVertex2f(self.bmax[0],self.bmin[1])
        glVertex2f(*self.bmax)
        glVertex2f(self.bmin[0],self.bmax[1])
        glEnd()


class AxisNotAlignedBox:
    def __init__(self, gripperPose, halfExtent):
        self.center_x = gripperPose[0]
        self.center_y = gripperPose[1]
        self.theta = gripperPose[2]
        self.half_length = halfExtent[0] # x
        self.half_height = halfExtent[1] # z

    def drawGL(self):
        # Calculate the corner points of the rotated box
        cos_theta = math.cos(-self.theta)
        sin_theta = math.sin(-self.theta)

        p1 = (self.center_x - cos_theta * self.half_length + sin_theta * self.half_height,
              self.center_y + sin_theta * self.half_length + cos_theta * self.half_height)
        
        p2 = (self.center_x + cos_theta * self.half_length + sin_theta * self.half_height,
              self.center_y - sin_theta * self.half_length + cos_theta * self.half_height)

        p3 = (self.center_x + cos_theta * self.half_length - sin_theta * self.half_height,
              self.center_y - sin_theta * self.half_length - cos_theta * self.half_height)

        p4 = (self.center_x - cos_theta * self.half_length - sin_theta * self.half_height,
              self.center_y + sin_theta * self.half_length - cos_theta * self.half_height)

        glBegin(GL_QUADS)
        glVertex2f(*p1)
        glVertex2f(*p2)
        glVertex2f(*p3)
        glVertex2f(*p4)
        glEnd()

    def contains(self, point, tolerance=0.03):
        """
            point: list[2], the point-mass object.
            tolerance: a margin that the object can violate the collision constraint with the gripper.
        """
        # Translate the point to the box's coordinate system
        translated_x = point[0] - self.center_x
        translated_y = point[1] - self.center_y

        # Rotate the point in the opposite direction of the box
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        rotated_x = translated_x * cos_theta - translated_y * sin_theta
        rotated_y = translated_x * sin_theta + translated_y * cos_theta

        # Check if the rotated point is within the axis-aligned box
        return (abs(rotated_x) <= self.half_length-tolerance) and (abs(rotated_y) <= self.half_height-tolerance)

class Geometric2DCSpace(BoxConfigurationSpace):
    def __init__(self):
        BoxConfigurationSpace.__init__(self,[0,0],[1,1])
        self.obstacles = []
        self.obstacleParams = None

    def addObstacleParam(self,params):
        self.obstacleParams = params

    def addObstacle(self,obs):
        self.obstacles.append(obs) # initial obstacle params

    def feasible(self,x):
        """Collision detection.
            input - x: list[2], position of object. 
        """
        if not BoxConfigurationSpace.feasible(self,x): return False
        return True

        # TODO: applied for the features in the main branch
        # if self.obstacleParams is None: # static obstacles [Cage]
        #     for o in self.obstacles:
        #         if o.contains(x): return False
        #     return True
        # elif self.obstacleParams is not None:
        #     if len(self.obstacleParams)==0: # [CagePlanner]
        #         return True
        #     if len(self.obstacleParams[0])==4: # moving obstacles with horizontal gripper [CageMovingObstacle]
        #         for i in range(len(self.obstacles)):
        #             topLeftX = self.obstacleParams[i][0] + obstaclePos[0]
        #             topLeftY = self.obstacleParams[i][1] + obstaclePos[1]
        #             o = Box(topLeftX,
        #                     topLeftY,
        #                     topLeftX+self.obstacleParams[i][2],
        #                     topLeftY+self.obstacleParams[i][3]
        #                     )
        #             if o.contains(x): return False
        #         return True
        #     else: # len(self.obstacleParams[0])==5, tilting gripper, [BalanceGrasp]
        #         gripperPose = obstaclePos # xc, yc, theta
        #         halfExtent = self.obstacleParams[0][3:]
        #         o = AxisNotAlignedBox(gripperPose, halfExtent)
        #         if o.contains(x): return False # collision. TODO: pybullet
        #         return True

    def toScreen(self,q):
        return (q[0]-self.box.bmin[0])/(self.box.bmax[0]-self.box.bmin[0]),(q[1]-self.box.bmin[1])/(self.box.bmax[1]-self.box.bmin[1])

    def toState(self,x,y):
        return (self.box.bmin[0]+x*(self.box.bmax[0]-self.box.bmin[0]),
                self.box.bmin[1]+y*(self.box.bmax[1]-self.box.bmin[1]))

    def beginDraw(self):
        if self.box.bmin != [0,0] or self.box.bmin != [1,1]:
            glPushMatrix()
            glScalef(1.0/(self.box.bmax[0]-self.box.bmin[0]),1.0/(self.box.bmax[1]-self.box.bmin[1]),1)
            glTranslatef(-self.box.bmin[0],-self.box.bmin[1],0)

    def endDraw(self):
        if self.box.bmin != [0,0] or self.box.bmin != [1,1]:
            glPopMatrix()

    def drawObstaclesGL(self, obstaclePos=None):
        self.beginDraw()
        if obstaclePos is None:
            # glColor3f(0.2,0.2,0.2)
            glColor3f(0, 0.65, 0)
            for o in self.obstacles:
                o.drawGL()
        else: # moving obstacles
            for i in range(len(self.obstacles)):
                topLeftX = self.obstacleParams[i][0] + obstaclePos[0]
                topLeftY = self.obstacleParams[i][1] + obstaclePos[1]
                o = Box(topLeftX,
                        topLeftY,
                        topLeftX+self.obstacleParams[i][2],
                        topLeftY+self.obstacleParams[i][3]
                        )
                o.drawGL()
        self.endDraw()

    def drawGripperGL(self, gripperPose, halfExtent, color=[1.0, 0.65, 0.0, 1]):
        """For cagePlanner and cageEL"""
        self.beginDraw()
        glColor4f(1,1,1,1)
        gripper = AxisNotAlignedBox(gripperPose, halfExtent)
        glColor4f(*color)
        gripper = AxisNotAlignedBox(gripperPose, halfExtent)
        gripper.drawGL()
        self.endDraw()

    def drawVerticesGL(self, qs):
        # Determine the range of costs
        min_cost = min(q[-1] for q in qs)
        max_cost = max(q[-1] for q in qs)
        cost_range = max_cost - min_cost if max_cost != min_cost else 1  # prevent division by zero

        self.beginDraw()
        glBegin(GL_POINTS)
        for q in qs:
            # Normalize the cost to [0,1]
            normalized_cost = (q[-1] - min_cost) / cost_range

            # Interpolate between blue (low cost) and red (high cost)
            red = normalized_cost  # Higher cost -> More red
            blue = 1 - normalized_cost  # Lower cost -> More blue
            green = 0  # Keeping green constant, but you can adjust it as needed

            # Set the color for the current point
            glColor4f(red, green, blue, 1)  # Adjust the alpha as needed

            # Draw the point
            glVertex2f(q[0],q[1])
        glEnd()
        self.endDraw()

    def drawLineGL(self, a, b):
        self.beginDraw()
        # glLineWidth(8)
        # glBegin(GL_LINES)  # Use GL_LINES to draw lines

        # # Set the color for the line (change as needed)
        # glColor4f(0.3, .3, 0.3, 1)  # Green color

        # # Draw the line from a to b
        # glVertex2f(*a)  # Starting point of the line
        # glVertex2f(*b)  # Ending point of the line

        # glEnd()

        glColor4f(0,0,0,.8)
        gripper = AxisNotAlignedBox([1.75,a[1],0], [2,0.1])
        gripper.drawGL()
        self.endDraw()

    def drawObjectGL(self,q):
        glColor3f(0,0,1)
        glPointSize(7.0)
        self.drawVerticesGL([q])

    def drawRobotGL(self,q, color=[0,.6,0,1]):
        glColor4f(*color)
        glPointSize(90.0)
        self.beginDraw()
        glBegin(GL_POINTS)
        glVertex2f(q[0],q[1])
        glEnd()
        self.endDraw()
        # self.drawVerticesGL([q])

    def drawGoalGL(self, goal, example_name=None, color='escapeGoal'):
        # Set the colors for the goals
        if color == 'escapeGoal':
            c = [1, 0, 0, 0.2]
        elif color == 'complementCaptureSet':
            c = [255/255, 220/255, 255/255, 1]
        elif color == 'successSet':
            c = [105/255, 130/255, 126/255, .4]

        self.beginDraw()
        if isinstance(goal,NeighborhoodSubset):
            q = goal.c
            glColor3f(.2,.4,.6)
            gldraw.circle(q,goal.r,filled=False)
            glPointSize(7.0)
            glBegin(GL_POINTS)
            glVertex2f(q[0],q[1])
            glEnd()
        elif isinstance(goal,SingletonSubset):
            q = goal.x
            glColor3f(.2,.4,.6)
            glPointSize(7.0)
            glBegin(GL_POINTS)
            glVertex2f(q[0],q[1])
            glEnd()
        else:
            if example_name in ["is_plane_push", "is_plane_push_rrtstar", "is_balance_grasp",]:
                goal.components[0].drawGL([1,1,1,1]) # erase set, union box set, ring set
                goal.components[0].drawGL(c) # erase set, union box set, ring set
            else:
                glColor4f(0,1,0,0.5)
                glPointSize(4.0)
                glBegin(GL_POINTS)
                for i in range(500):
                    q = goal.sample()
                    glVertex2f(q[0],q[1])
                glEnd()
        self.endDraw()

    def drawInterpolatorGL(self,interpolator):
        self.beginDraw()
        if isinstance(interpolator,LinearInterpolator):
            #straight line paths
            glBegin(GL_LINES)
            glVertex2f(interpolator.a[0],interpolator.a[1])
            glVertex2f(interpolator.b[0],interpolator.b[1])
            glEnd()
        elif isinstance(interpolator,PiecewiseLinearInterpolator):
            glBegin(GL_LINE_STRIP)
            for x in interpolator.path:
                glVertex2f(x[0],x[1])
            glEnd()
        else:
            glBegin(GL_LINE_STRIP)
            for s in range(10):
                u = float(s) / (9.0)
                x = interpolator.eval(u)
                glVertex2f(x[0],x[1])
            glEnd()
        self.endDraw()

    def clearance(self,x):
        res = BoxConfigurationSpace.clearance(self,x)
        for o in self.obstacles:
            res.append(o.signedDistance(x))
        return res
    
    def clearance_gradient(self,x):
        res = [BoxConfigurationSpace.clearance_gradient(self,x)]
        for o in self.obstacles:
            res.append(o.signedDistance_gradient(x))
        return np.vstack(res)


def circleTest():
    space = Geometric2DCSpace()
    space.addObstacle(Circle(0.5,0.4,0.39))
    start=[0.06,0.25]
    goal=[0.94,0.25]
    objective = PathLengthObjectiveFunction()
    goalRadius = 0.1
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)


def rrtChallengeTest():
    w = 0.03
    eps = 0.01
    space = Geometric2DCSpace()
    space.box = BoxSet([0,0],[1,w*2+eps])
    space.addObstacle(Box(0,w*2+eps,1,1))
    space.addObstacle(Box(w,w,1,w+eps))
    start=[1-w*0.5,w+eps+w*0.5]
    goal=[1-w*0.5,w*0.5]
    goalRadius = w*0.5
    objective = PathLengthObjectiveFunction()
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)

def kinkTest():
    space = Geometric2DCSpace()
    w = 0.02
    space.addObstacle(Box(0.3,0,0.5+w*0.5,0.2-w*0.5))
    space.addObstacle(Box(0.5+w*0.5,0,0.7,0.3-w*0.5))
    space.addObstacle(Box(0.3,0.2+w*0.5,0.5-w*0.5,0.7))
    space.addObstacle(Box(0.5-w*0.5,0.3+w*0.5,0.7,0.7))
    start=[0.06,0.25]
    goal=[0.94,0.25]
    goalRadius = 0.1
    objective = PathLengthObjectiveFunction()
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)

def bugtrapTest():
    space = Geometric2DCSpace()
    w = 0.1
    """
    space.addObstacle(Box(0.55,0.25,0.6,0.75))
    space.addObstacle(Box(0.15,0.25,0.55,0.3))
    space.addObstacle(Box(0.15,0.7,0.55,0.75))
    space.addObstacle(Box(0.15,0.25,0.2,0.5-w*0.5))
    space.addObstacle(Box(0.15,0.5+w*0.5,0.2,0.75))
    start=[0.5,0.5]
    goal=[0.65,0.5]
    """
    start=[0.1,0.5]
    goal=[0.9,0.5]
    space.addObstacle(Box(0.3,0.3,0.7,0.7))
    goalRadius = 0.1
    objective = PathLengthObjectiveFunction()
    return PlanningProblem(space,start,goal,
                           objective=objective,
                           visualizer=space,
                           costLowerBound = vectorops.distance,
                           goalRadius = goalRadius,
                           euclidean = True)

