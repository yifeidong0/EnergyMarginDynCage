from ..spaces.controlspace import *
from ..spaces.configurationspace import SingletonSubset,NeighborhoodSubset
from ..spaces import metric
from . import allplanners

class PlanningProblem:
    def __init__(self,space,
                 start=None,
                 goal=None,
                 objective=None,
                 visualizer=None,
                 heuristic=None,
                 costLowerBound=None,
                 goalRadius=None,
                 euclidean=False,
                 successSet=None,
                 complementSuccessSet=None,
                 captureSet=None,
                 complementCaptureSet=None,
                 ):
        self.space = space
        if isinstance(space,ControlSpace):
            self.controlSpace = space
            self.configurationSpace = space.configurationSpace()
        else:
            self.controlSpace = None
            self.configurationSpace = space
        if goalRadius != None and isinstance(goal,(list,tuple)):
            goal = NeighborhoodSubset(self.configurationSpace,goal,goalRadius)
        self.start = start
        self.goal = goal
        self.objective = objective
        self.visualizer = visualizer
        self.heuristic = heuristic
        self.costLowerBound = costLowerBound
        self.euclidean = euclidean
        self.successSet = successSet
        self.complementSuccessSet = complementSuccessSet
        self.captureSet = captureSet
        self.complementCaptureSet = complementCaptureSet

    def cartesian(self):
        return self.euclidean
    def pointToPoint(self):
        return isinstance(self.goal,SingletonSubset) or isinstance(self.goal, (list,tuple))
    def differentiallyConstrained(self):
        return self.controlSpace != None
    def planner(self,type,**params):
        d = metric.euclideanMetric if self.euclidean else self.configurationSpace.distance
        return allplanners.makePlanner(type,
                                       space=self.space,
                                       start=self.start,
                                       goal=self.goal,
                                       objective=self.objective,
                                       heuristic=self.heuristic,
                                       metric=d,
                                       costLowerBound=self.costLowerBound,
                                       successSet=self.successSet,
                                       complementSuccessSet=self.complementSuccessSet,
                                       captureSet=self.captureSet,
                                       complementCaptureSet=self.complementCaptureSet,
                                       **params)
