from __future__ import print_function,division
from six import iteritems

from pomp.planners import allplanners
from pomp.planners import test
from pomp.example_problems import *
from pomp.spaces.objectives import *
from pomp.bullet.forwardsimulator import *
import time
import copy
import sys
import os,errno

numTrials = 1

def mkdir_p(path):
    """Quiet path making"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def testPlannerDefault(problem, problemName, maxTime, maxIters, plannerType,**plannerParams):
    global numTrials
    print("Planning with",plannerType,'on problem',problemName)
    planner = problem.planner(plannerType,**plannerParams)
    folder = os.path.join("data",problemName)
    mkdir_p(folder)
    data_id = 0
    if 'data_id' in plannerParams: # TODO
        data_id = plannerParams['data_id']

    # from datetime import datetime
    # # Get the current timestamp as a string
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = allplanners.filename[plannerType]

    # Concatenate the timestamp with the filename and add the '.csv' extension
    # file_path = os.path.join(folder, f"{filename}_{timestamp}.csv")
    file_path = os.path.join(folder, f"{filename}.csv")

    test.testPlanner(planner,
                     numTrials,
                     maxTime,
                     maxIters,
                     #  os.path.join(folder,allplanners.filename[plannerType]+'.csv'), 
                     file_path,
                     data_id=data_id)


all_planners = ['ao-est','ao-rrt','r-est','r-est-prune','r-rrt','r-rrt-prune','rrt*','anytime-rrt','stable-sparse-rrt']
rrt_planners = ['ao-rrt','anytime-rrt','r-rrt','r-rrt-prune','stable-sparse-rrt']
est_planners = ['ao-est','r-est','r-est-prune']

all_problems = {'PlanePush', 'PlanePushRrtstar', 'PlanePushReal', 'BalanceGrasp', 'PlanePushMulti',
                'WaterSwing', 'BoxPivot', 'Herding', 'Gripper', 'GripperMulti', 'Shuffling'}

defaultParameters = {'maxTime':30, 'maxIters':200000}
customParameters = {
                    'PlanePush':{'maxTime':20},
                    'PlanePushRrtstar':{'maxTime':20},
                    'PlanePushReal':{'maxTime':20},
                    'PlanePushMulti':{'maxTime':20},
                    'BalanceGrasp':{'maxTime':20},
                    'WaterSwing':{'maxTime':20},
                    'BoxPivot':{'maxTime':20},
                    'Herding':{'maxTime':3000, 'maxIters':3000},
                    'Gripper':{'maxTime':10},
                    'GripperMulti':{'maxTime':10},
                    'Shuffling':{'maxTime':20},
                    }

def parseParameters(problem,planner):
    global defaultParameters,customParameters
    params = copy.deepcopy(defaultParameters)
    if problem in customParameters:
        params.update(customParameters[problem])
    if '(' in planner:
        #parse out key=value,... string
        name,args = planner.split('(',1)
        if args[-1] != ')':
            raise ValueError("Planner string expression must have balanced parenthesis, i.e.: func ( arglist )")
        args = args[:-1]
        args = args.split(',')
        for arg in args:
            kv = arg.split("=")
            if len(kv) != 2:
                raise ValueError("Unable to parse argument "+arg)
            try:
                params[kv[0]] = int(kv[1])
            except ValueError:
                try:
                    params[kv[0]] = float(kv[1])
                except ValueError:
                    params[kv[0]] = kv[1]
        planner = name
    return planner,params

def runTests(problem_name, planner_name, problem):
    planner_name, params = parseParameters(problem_name, planner_name)
    maxTime = params['maxTime']
    maxIters = params['maxIters']
    del params['maxTime']
    del params['maxIters']
    if problem.differentiallyConstrained() and planner_name in allplanners.kinematicPlanners:
        return
        #p does not support differentially constrained problems
    testPlannerDefault(problem, problem_name, maxTime, maxIters, planner_name, **params)
    print("Finished test on problem", problem_name, "with planner", planner_name)
    print("Parameters:")
    for (k,v) in iteritems(params):
        print(" ",k,":",v)
    return

def runViz(problem_name, planner_name, problem):
    #runVisualizer(rrtChallengeTest(),type=planner,nextStateSamplingRange=0.15,edgeCheckTolerance = 0.005)
    planner, params = parseParameters(problem_name, planner_name)
    if 'maxTime' in params:
        del params['maxTime']
    
    print("Planning on problem",problem_name,"with planner",planner)
    print("Parameters:")
    for (k,v) in iteritems(params):
        print(" ",k,":",v)
    runVisualizer(problem, type=planner, **params)


if __name__=="__main__":
    if len(sys.argv) < 3:
        print("Usage: main.py [-v] Problem Planner1 ... Plannerk")
        print()
        print("  Problem can be one of:")
        print("   ",",\n    ".join(sorted(all_problems)))
        print("  or 'all' to test all problems.")
        print()
        print("  Planner can be one of:")
        print("   ",",\n    ".join(sorted(all_planners)))
        print("  or 'all' to test all planners.")
        print()
        print("  If -v is provided, runs an OpenGL visualization of planning")
        exit(0)
    if sys.argv[1] == '-v':
        from pomp.visualizer import runVisualizer
        # visualization mode
        problem_name = sys.argv[2]
        planner_name = sys.argv[3]
        print("Testing visualization with problem", problem_name, "and planner", planner_name)
    else:
        problem_name = sys.argv[1]
        planner_name = sys.argv[2]
        print("Testing problems", problem_name, "with planners", planner_name)

    if problem_name == 'BalanceGrasp':
        dynamics_sim = forwardSimulationBalanceGrasp(gui=0)
        problem = balancegrasp.BalanceGraspTest(dynamics_sim)
    if problem_name == 'PlanePush':
        dynamics_sim = forwardSimulationPlanePush(gui=0)
        problem = planepush.PlanePushTest(dynamics_sim)
    if problem_name == 'PlanePushRrtstar':
        dynamics_sim = forwardSimulationPlanePushRrtstar(gui=1)
        problem = planepushrrtstar.PlanePushRrtstarTest(dynamics_sim)
    if problem_name == 'PlanePushReal':
        dynamics_sim = forwardSimulationPlanePushReal(gui=0)
        problem = planepushreal.PlanePushRealTest(dynamics_sim)
    if problem_name == 'PlanePushMulti':
        dynamics_sim = forwardSimulationPlanePushMulti(gui=0)
        problem = planepushmulti.PlanePushMultiTest(dynamics_sim)
    if problem_name == 'WaterSwing':
        dynamics_sim = forwardSimulationWaterSwing(gui=0)
        problem = waterswing.waterSwingTest(dynamics_sim)
    if problem_name == 'BoxPivot':
        dynamics_sim = forwardSimulationBoxPivot(gui=0)
        problem = boxpivot.BoxPivotTest(dynamics_sim)
    if problem_name == 'Herding':
        num_runs = 6
        # num_robots=10
        for num_robots in range(4,9):  
            dynamics_sim = forwardSimulationHerding(gui=0)
            problem = herding.HerdingTest(dynamics_sim, num_robots=num_robots, save_hyperparams=1)
            for i in range(num_runs):
                runTests(problem_name, planner_name, problem)
    if problem_name == 'Gripper':
        dynamics_sim = forwardSimulationGripper(gui=0)
        problem = gripper.GripperTest(dynamics_sim)
    if problem_name == 'GripperMulti':
        dynamics_sim = forwardSimulationGripperMulti(gui=1)
        problem = grippermulti.GripperMultiTest(dynamics_sim)
    if problem_name == 'Shuffling':
        dynamics_sim = forwardSimulationShuffling(gui=0)
        problem = shuffling.ShufflingTest(dynamics_sim)

    if sys.argv[1] == '-v':
        runViz(problem_name, planner_name, problem)
    else:
        runTests(problem_name, planner_name, problem)
