from __future__ import print_function,division
from builtins import range
from six import iteritems

from .klampt.glprogram import GLProgram
from .klampt import vectorops,so2,gldraw
from .spaces import metric
from .planners import allplanners
from .planners.kinodynamicplanner import EST
from .planners.optimization import iLQR
from .spaces.objectives import *
from OpenGL.GL import *
import os,errno
import time
import random
from OpenGL.GLUT import *
from OpenGL.GLU import *

def mkdir_p(path):
    """Quiet path making"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class PlanVisualizationProgram(GLProgram):
    """Attributes:
        problem (PlanningProblem): the overall planning problem
        planner (multiple...): the planner selected for testing
        plannerFilePrefix (str): where to save testing data
    
    Internal attributes:
        path (Trajectory): if a solution is found, the best trajectory
        G (pair (V,E)): the roadmap for the planner.
        painted, save_movie, movie_frame: used for drawing / saving movies.
    """
    def __init__(self,problem,planner,plannerFilePrefix):
        GLProgram.__init__(self)
        self.problem = problem
        self.planner = planner
        self.plannerFilePrefix = plannerFilePrefix
        self.path = None
        self.G = None
        self.painted = False
        self.save_movie = False
        self.movie_frame = 0
        self.prev_path_x = None
        self.display_new_path = False
        self.just_animated = False

    def mousefunc(self,button,state,x,y):
        if state == 0:
            if hasattr(self.planner,'nextSampleList'):
                self.planner.nextSampleList.append(self.problem.visualizer.toState(float(x)/self.width,float(y)/self.height))
                print(self.planner.nextSampleList)
            self.refresh()

    def idlefunc(self):
        if self.save_movie:
            if self.painted:
                try:
                    mkdir_p(self.plannerFilePrefix)
                    if self.just_animated:
                        self.just_animated = False
                    else:
                        # self.save_screen("/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/plane-push-sim/analysis-traj-id-1-and-2/escape/traj-2/state-6/image%04d.ppm"%(self.movie_frame))
                        self.save_screen("%s/image%04d.ppm"%(self.plannerFilePrefix,self.movie_frame))
                        self.movie_frame += 1
                except Exception as e:
                    ex_type, ex, tb = sys.exc_info()
                    traceback.print_tb(tb)
                    self.save_movie = False
                    raise e

                self.planner.planMore(100)
                self.path = self.planner.getPath()
                self.G = self.planner.getRoadmap()
                self.planner.getScores()
                self.refresh()
                self.painted = False
               
    def keyboardfunc(self,key,x,y):
        key = key.decode("utf-8") 
        print("Key",key,"pressed at",x,y)
        # print('self.height',self.height) # 640
        # print('self.width',self.width) # 640
        if key==' ':
            print("Planning 1...")
            self.planner.planMore(1)
            self.path = self.planner.getPath()
            self.G = self.planner.getRoadmap()
            self.planner.stats.pretty_print()
            self.refresh()
        elif key=='h':
            print("Planning 100...")
            self.planner.planMore(100)
            self.path = self.planner.getPath()
            self.G = self.planner.getRoadmap()
            self.planner.getScores()
            self.planner.stats.pretty_print()
            self.refresh()
        elif key=='p':
            print("Planning 1000...")
            self.planner.planMore(1000)
            self.path = self.planner.getPath()
            self.G = self.planner.getRoadmap()
            self.planner.getScores()
            self.planner.stats.pretty_print()
            self.refresh()
        elif key=='o':
            if self.path is not None:
                costWeightInit = 0.1
                xinit,uinit = self.path
            else:
                costWeightInit = 0.1
                xinit,uinit,cost = self.planner.getBestPath(self.problem.objective + SetDistanceObjectiveFunction(self.problem.goal) + StepCountObjectiveFunction()*(-0.1))
            print("Optimizing 10...")
            objective = self.problem.objective
            if isinstance(objective,PathLengthObjectiveFunction):
                objective = EnergyObjectiveFunction()
            if self.problem.controlSpace is None:
                from .spaces.controlspace import ControlSpaceAdaptor
                optimizer = iLQR(ControlSpaceAdaptor(self.problem.configurationSpace),objective,self.problem.goal,'log',costWeightInit)
            else:
                optimizer = iLQR(self.problem.controlSpace,objective,self.problem.goal,'log',costWeightInit)
            converged,reason = optimizer.run(xinit,uinit,20)
            print("Converged?",converged,"Termination reason",reason)
            print("Endpoints",optimizer.xref[0],optimizer.xref[-1])
            self.path = optimizer.xref,optimizer.uref
            self.refresh()
        elif key=='r':
            print("Resetting planner")
            self.planner.reset()
            self.path = None
            self.G = self.planner.getRoadmap()
            self.refresh()
        elif key=='m':
            self.save_movie = not self.save_movie
            if self.save_movie:
                self.refresh()
        elif key=='t':
            maxTime = 30
            # testPlanner(self.planner,10,maxTime,self.plannerFilePrefix+'.csv')

    def drawText(self, text, x, y):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glColor3f(0, 0, 0)  # Red color
        glRasterPos2f(x, y)
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(char))

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
    def display(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # glOrtho(0,1,1,0,-1,1)
        glOrtho(0,1,0,1,-1,1) # origin: bottom left
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_POINT_SMOOTH)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Draw initial manipulator position and goal region
        if hasattr(self.problem.controlSpace, "is_energy_labeler"):
            gripperPose = self.problem.controlSpace.obstacles[:3]
            halfExtent = self.problem.controlSpace.obstacles[3:]
            glColor3f(0, 0.65, 0)
            self.problem.visualizer.drawGripperGL(gripperPose, halfExtent)
            self.problem.visualizer.drawGoalGL(self.problem.goal)
        elif hasattr(self.problem.controlSpace, "is_plane_push"):
            self.problem.visualizer.drawGripperGL([2,2,0], [10,10], [79/255.0, 198/255.0, 1, 1]) # draw background color (safe set)
            self.problem.visualizer.drawGoalGL(self.problem.goal, example_name="is_plane_push", color='escapeGoal')
            self.problem.visualizer.drawGoalGL(self.problem.captureSet, example_name="is_plane_push", color='complementCaptureSet')
            self.problem.visualizer.drawGoalGL(self.problem.successSet, example_name="is_plane_push", color='successSet')
            self.problem.visualizer.drawRobotGL(self.problem.controlSpace.cage.start_state[6:8])
            self.problem.visualizer.drawGripperGL(self.problem.controlSpace.cage.start_state[:3], [.6, .2]) # draw rectangluar object
            self.problem.visualizer.drawLineGL(*self.problem.controlSpace.cage.obstacle_borderline) # obstacle border represented by a line
        elif hasattr(self.problem.controlSpace, "is_plane_push_multi"):
            self.problem.visualizer.drawGripperGL([2,2,0], [10,10], [79/255.0, 198/255.0, 1, 1]) # draw background color (safe set)
            self.problem.visualizer.drawGoalGL(self.problem.goal, example_name="is_plane_push", color='escapeGoal')
            self.problem.visualizer.drawGoalGL(self.problem.complementCaptureSet, example_name="is_plane_push", color='complementCaptureSet')
            self.problem.visualizer.drawGoalGL(self.problem.successSet, example_name="is_plane_push", color='successSet')
            for i in range(self.problem.controlSpace.num_objects):
                self.problem.visualizer.drawRobotGL(self.problem.controlSpace.cage.start_state[6*i:6*i+2]) # draw circular objects
            self.problem.visualizer.drawGripperGL(self.problem.controlSpace.cage.start_state[-3:], [.3, .1]) # draw rectangular manipulator
            self.problem.visualizer.drawLineGL(*self.problem.controlSpace.cage.obstacle_borderline) # obstacle border represented by a line
        elif hasattr(self.problem.space, "is_plane_push_rrtstar"):
            self.problem.visualizer.drawGoalGL(self.problem.goal, example_name="is_plane_push_rrtstar", color='complementCaptureSet')
            self.problem.visualizer.drawLineGL(*self.problem.space.obstacle_borderline) # obstacle border represented by a line
        elif hasattr(self.problem.controlSpace, "is_balance_grasp"):
            self.problem.visualizer.drawRobotGL(self.problem.controlSpace.cage.start_state[6:8])
            self.problem.visualizer.drawGoalGL(self.problem.goal, example_name="is_balance_grasp", color='escapeGoal')
            self.problem.visualizer.drawGoalGL(self.problem.complementCaptureSet, example_name="is_balance_grasp", color='complementCaptureSet')
            self.problem.visualizer.drawGoalGL(self.problem.successSet, example_name="is_balance_grasp", color='successSet')
            self.problem.visualizer.drawLineGL(*self.problem.controlSpace.cage.obstacle_borderline) # obstacle border represented by a line
        elif hasattr(self.problem.controlSpace, "is_box_pivot"):
            self.problem.visualizer.drawObjectGL(self.problem.controlSpace.cage.start_state[:2])
        elif hasattr(self.problem.controlSpace, "is_herding"):
            self.problem.visualizer.drawGoalGL(self.problem.complementCaptureSet, example_name="is_herding", color='complementCaptureSet')
            self.problem.visualizer.drawGoalGL(self.problem.goal, example_name="is_herding", color='escapeGoal')
        else:
            self.problem.visualizer.drawGoalGL(self.problem.goal)

        if hasattr(self.planner,'nextSampleList'):
            for p in self.planner.nextSampleList:
                self.problem.visualizer.drawObjectGL(p)

        # Draw start and goal region
        self.problem.visualizer.drawObjectGL(self.problem.start)

        # Draw roadmap
        self.draw_graph()

        # Draw static/animated path found by the planner
        self.draw_solution_path()
        # TODO: print the cost - self.planner.bestPathCost - on the window

        #Debug extension cache
        est = None
        if isinstance(self.planner,EST): est = self.planner
        elif hasattr(self.planner,'est'): est = self.planner.est
        if est and hasattr(est,'extensionCache'):
            glLineWidth(3.0)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
            sumw = sum(est.extensionCache[0])
            for (w,(n,u,e)) in zip(*est.extensionCache):
                glColor4f(1,0,1,(0.1+0.9*w/sumw))
                self.problem.visualizer.drawInterpolatorGL(e)
                #self.problem.visualizer.drawObjectGL(e.end())
            glDisable(GL_BLEND)
            glLineWidth(1.0)

        # Debug density
        """
        if est:
            for n in est.nodes:
                if est.density(n.x) <= 1.0:
                    self.problem.visualizer.drawObjectGL(n.x)
        """

    def draw_graph(self, draw_edge=1):
        """Draw the roadmap."""
        if self.G:
            V,E = self.G

            # Draw random gripper and rectangular object from the nodes
            if hasattr(self.problem.controlSpace, "is_plane_push"):
                total_num = len(V)
                # rand_num  = int(total_num * 0.03)
                rand_num  = 1
                for i in random.sample(range(total_num), rand_num):
                    self.problem.visualizer.drawGripperGL(V[i][:3], [.6, .2], [1.0, 0.65, 0.0, 0.5*max(0,(7-V[i][-1])/7)]) # draw rectangluar object
                    self.problem.visualizer.drawRobotGL(V[i][6:8], [0,.6,0,0.5*max(0,(7-V[i][-1])/7)]) # draw robot gripper

            # Draw static obstacle
            if hasattr(self.problem.controlSpace, "is_herding"):
                glLineWidth(1)
                num_robot = self.problem.controlSpace.cage.num_robot
                for i in range(num_robot):
                    self.problem.visualizer.drawRobotGL(V[0][4+2*i:6+2*i])

            # Draw edges
            if draw_edge:
                glColor4f(0.5,0.5,0.5,0.5)
                glLineWidth(3)
                for (i,j,u) in E: # (i,j,e): parent index, child_index, parent_u
                    x_new = self.problem.controlSpace.eval(V[i][:-1], u, 1, print_via_points=True)
                    xo_via_points = self.problem.controlSpace.xo_via_points
                    self.problem.visualizer.beginDraw() # draw edges
                    glBegin(GL_LINE_STRIP)
                    for p in xo_via_points:
                        glVertex2f(p[0],p[1])
                    glEnd()
                    self.problem.visualizer.endDraw()
                glDisable(GL_BLEND)

            # Draw nodes
            glLineWidth(1)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0,0,0,0.5)
            glPointSize(12.0)
            self.problem.visualizer.drawVerticesGL(V)

            # Draw text box of metrics at last frame
            # if self.movie_frame == 29:
            #     success_metric, maneuver_metric = self.planner.getScores()
            #     success_metric = round(success_metric, 2)
            #     maneuver_metric = round(maneuver_metric, 2)
            #     self.drawText("("+str(success_metric)+", "+str(maneuver_metric)+")", self.width / 1.1 - 70, self.height / 1.1 + 10)

            # # Draw text box of escape cost
            # escape_cost = self.planner.getBestPathCost()
            # escape_cost = "inf" if escape_cost is None else str(round(escape_cost, 3))
            # self.drawText("Escape cost: "+escape_cost, self.width / 1.1 - 110, self.height / 1.1 + 10)

    def draw_solution_path(self):
        if self.path is not None:
            # print("!!!!x",self.path[0])
            # print("!!!!u",self.path[1])
            if not are_nested_lists_equal(self.prev_path_x, self.path[0]):
                # self.display_new_path = True
                self.display_new_path = False
                self.prev_path_x = self.path[0]
        
        if not self.display_new_path: # draw static path without animation if no path update
            self.draw_path_static()
        else: # in animation mode
            self.problem.visualizer.drawObjectGL(self.problem.start)
            self.problem.visualizer.drawGoalGL(self.problem.goal)

            if self.path and len(self.path[0]) > 1:
                for k in range(len(self.path[0]) - 1):
                    if k >= len(self.path[0])-1:
                        break
                    x1, u = self.path[0][k], self.path[1][k]
                    x2 = self.path[0][k+1]

                    if hasattr(self.problem.controlSpace, "is_energy_labeler"):
                        # Clear the previous obstacle by drawing a background color (e.g., white)
                        glColor3f(1, 1, 1)
                        self.problem.visualizer.drawGripperGL(x1[4:7], self.problem.controlSpace.half_extents_gripper)

                        # Draw the new obstacle at x2
                        glColor3f(0, 0.65, 0)
                        self.problem.visualizer.drawGripperGL(x2[4:7], self.problem.controlSpace.half_extents_gripper)

                    # Draw the graph again
                    self.draw_graph()

                    for m in range(k+1):
                        x1, u = self.path[0][m], self.path[1][m]
                        x2 = self.path[0][k+1]

                        # Interpolate and draw all previous line segments along object trajectory
                        glColor3f(0, 0.75, 0)
                        glLineWidth(7.0)
                        interpolator = self.problem.space.interpolator(x1, u)
                        self.problem.visualizer.drawInterpolatorGL(interpolator)
                        
                        # Draw the node x2 along object trajectory
                        glLineWidth(1)
                        self.problem.visualizer.drawObjectGL(x2[:2])

                        # Indicate robot gripper's center moving position
                        if hasattr(self.problem.controlSpace, "is_water_swing"):
                            glLineWidth(1)
                            self.problem.visualizer.drawRobotGL(x2[6:8])
                        if hasattr(self.problem.controlSpace, "is_plane_push"):
                            glLineWidth(1)
                            self.problem.visualizer.drawRobotGL(x2[6:8])
                        if hasattr(self.problem.controlSpace, "is_box_pivot"):
                            glLineWidth(1)
                            height_spring = 3.7
                            self.problem.visualizer.drawRobotGL([x2[7], height_spring])
                        if hasattr(self.problem.controlSpace, "is_herding"):
                            glLineWidth(1)
                            num_robot = self.problem.controlSpace.cage.num_robot
                            for i in range(num_robot):
                                self.problem.visualizer.drawRobotGL(x2[4+2*i:6+2*i])

                    # Process events to update the display
                    glutSwapBuffers()
                    glutMainLoopEvent()

                    # Pause and save screen
                    if self.display_new_path:
                        time.sleep(0.03*u[0]) # <0.05*u[0] might cause frame chaos in the generated movies
                    if self.save_movie:
                        self.save_screen("%s/image%04d.ppm"%(self.plannerFilePrefix,self.movie_frame))
                        self.movie_frame += 1
                    time.sleep(0.1)

        self.display_new_path = False

    def draw_path_static(self):
        """Draw the solution path without animation."""
        if self.path:
            # Plot path edges
            glColor3f(0,0.75,0)
            glLineWidth(4.0)
            for q,u in zip(self.path[0][:-1],self.path[1]):
                interpolator = self.problem.space.interpolator(q,u)
                self.problem.visualizer.drawInterpolatorGL(interpolator)

            # Plot path nodes
            # glLineWidth(1)
            # for q in self.path[0]:
            #     self.problem.visualizer.drawObjectGL(q)

    def displayfunc(self):
        GLProgram.displayfunc(self)
        self.painted = True

def are_nested_lists_equal(list1, list2):
    # Check if the lengths of the lists are equal
    if list1 is None or list2 is None:
        return False
    if len(list1) != len(list2):
        return False

    # Iterate through the elements of the lists
    for i in range(len(list1)):
        if isinstance(list1[i], list) and isinstance(list2[i], list):
            if not are_nested_lists_equal(list1[i], list2[i]):
                return False
        else:
            if list1[i] != list2[i]:
                return False

    return True

def runVisualizer(problem,**plannerParams):   
    if 'type' in plannerParams:
        plannerType = plannerParams['type']
        del plannerParams['type']
    else:
        plannerType = 'r-rrt'

    planner = problem.planner(plannerType,**plannerParams)
    mkdir_p("data")
    program = PlanVisualizationProgram(problem,planner,os.path.join("data",allplanners.filename[plannerType]))
    program.width = program.height = 960
    program.run()


