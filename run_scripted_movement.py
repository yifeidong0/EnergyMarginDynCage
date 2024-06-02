from pomp.example_problems.balancegrasp import *
from pomp.example_problems.planepush import *
from pomp.example_problems.planepushmulti import *
from pomp.example_problems.waterswing import *
from pomp.example_problems.boxpivot import *
from pomp.example_problems.shuffling import *
from pomp.example_problems.gripper import *
from pomp.example_problems.grippermulti import *
from pomp.bullet.scriptedmovement import *
import time
import csv

problem_name = "GripperMulti" # "PlanePush", "PlanePushMulti", "BalanceGrasp", "BoxPivot", "Gripper", "GripperMulti", "Shuffling", "WaterSwing", 
gui = 1
num_via_points = 10
num_trajs = 1
filename = "scripted_movement_viapoints_{}.csv".format(problem_name)
filename_metric = "scripted_movement_heuristics_{}.csv".format(problem_name)
filename_suc_label = "scripted_movement_success_labels_{}.csv".format(problem_name)
filename_capture_label = "scripted_movement_capture_labels_{}.csv".format(problem_name)

if problem_name == 'PlanePush':
    total_time = 2.5
    num_state_planner = 9
    headers = ['num_traj', 'data_id', 'xo', 'yo', 'thetao', 'vxo', 'vyo', 'omegao', 'xg', 'yg', 'thetag', 'vxg', 'vyg', 'omegag']
    headers_metric = ['num_traj', 'data_id', 'shortest_distance', 'S_stick', 'S_engage']
    headers_success = ['num_traj', 'label']
    headers_capture = ['num_traj', 'data_id', 'label', 'lateral_friction_coef', 'lateral_friction_coef_perturb']
    fake_data = [5.0, 6.3, 0.0, 0.0, 0.0, 0.0, 
                 5.0, 6.0, 0.0, 0.0, 2.0, 0.0]
    dynamics_sim = forwardSimulationPlanePush(gui=0)
    cage = PlanePush(fake_data, dynamics_sim)
    x_init = fake_data
    dynamics_sim.finish_sim()
    sim = scriptedMovementSimPlanePush(cage, gui=gui)
if problem_name == 'PlanePushMulti':
    total_time = 2
    num_object = 10
    num_state_planner = 3 + 6*num_object
    headers = ['num_traj', 'data_id', ] + \
               ['xo', 'yo', 'thetao', 'vxo', 'vyo', 'omegao',]*num_object + \
               ['xg', 'yg', 'thetag', 'vxg', 'vyg', 'omegag']
    headers_metric = ['num_traj', 'data_id',] + ['shortest_distance', 'S_stick', 'S_engage']*num_object
    headers_success = ['num_traj', 'success_all_label', 'success_exists_label']
    headers_capture = ['num_traj', 'data_id', 'capture_exists_label', 'capture_all_label', 'lateral_friction_coef']
    fake_data = [1.0, 0.4, 0.0, 0.0, 0.0, 0.0,]*num_object \
                + [1.0, 0.1, 0.0, 0.0, 1.0, 0.2,]
                # 1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 
                # 1.4, 0.7, 0.0, 0.0, 0.0, 0.0, 
    dynamics_sim = forwardSimulationPlanePushMulti(gui=0)
    cage = PlanePushMulti(fake_data, dynamics_sim)
    x_init = fake_data
    dynamics_sim.finish_sim()
    sim = scriptedMovementSimPlanePushMulti(cage, gui=gui)
if problem_name == 'BalanceGrasp':
    total_time = 2.5
    num_state_planner = 9
    headers = ['num_traj', 'data_id', 'xo', 'yo', 'thetao', 'vxo', 'vyo', 'omegao', 'xg', 'yg', 'thetag', 'vxg', 'vyg', 'omegag']
    headers_metric = ['num_traj', 'data_id', 'shortest_distance', 'S_stick', 'S_engage']
    headers_success = ['num_traj', 'label']
    headers_capture = ['num_traj', 'data_id', 'label',]
    fake_data = [5.0, 4.3, 0.0, 0.0, 0.0, 0.0, 
                 5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    dynamics_sim = forwardSimulationBalanceGrasp(gui=0)
    cage = BalanceGrasp(fake_data, dynamics_sim)
    x_init = fake_data
    dynamics_sim.finish_sim()
    sim = scriptedMovementSimBalanceGrasp(cage, gui=gui)
if problem_name == 'BoxPivot':
    for_paper_version = 1
    total_time = 1.5
    num_state_planner = 8
    headers = ['num_traj', 'data_id', 'xo', 'yo', 'thetao', 'vxo', 'vyo', 'omegao', 'xg1', 'xg2', 'vxg1', 'vxg2']
    headers_metric = ['num_traj', 'data_id', 'shortest_distance_spring', 'S_stick_spring', 'S_engage_spring', 'shortest_distance_ground', 'S_stick_ground', 'S_engage_ground']
    headers_success = ['num_traj', 'label',]
    headers_capture = ['num_traj', 'data_id', 'label', 'lateral_friction_coef']
    fake_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, ]
    dynamics_sim = forwardSimulationBoxPivot(gui=0)
    cage = BoxPivot(fake_data, dynamics_sim)
    if for_paper_version:
        x_init = [2.5, 0.5, 0, 0, 0, 0,
                1, 1.7, 0, 0]
    else:
        x_init = [6, 2, 0, 0, 0, 0,
                  2, 3.7, 0, 0]
    dynamics_sim.finish_sim()
    sim = scriptedMovementSimBoxPivot(cage, gui=gui)
if problem_name == 'Gripper':
    total_time = 2
    num_state_planner = 6+6+9+1
    headers = ['num_traj', 'data_id', 
               'xo', 'zo', 'yo', 'thetaxo', 'thetayo', 'thetazo', 
               'vxo', 'vyo', 'vzo', 'omegaxo', 'omegayo', 'omegazo', 
               'j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'zt',
               'jv0', 'jv1', 'jv2', 'jv3', 'jv4', 'jv5', 'jv6', 'jv7', 'jv8', 'vzt',]
    headers_metric = ['num_traj', 'data_id', 'shortest_distance_3', 'S_stick_3', 'S_engage_3', 
                      'shortest_distance_7', 'S_stick_7', 'S_engage_7', 'shortest_distance_11', 'S_stick_11', 'S_engage_11',]
    headers_success = ['num_traj', 'label']
    headers_capture = ['num_traj', 'data_id', 'label', 'lateral_friction_coef', 'mass_object']
    fake_data = [-0.99, 0.6, 0.0, 0.0, 0.0, 0.0] + [0.0,]*6 + [0*math.pi/12]*9 + [0.0,] + [0.0]*9 + [0.0,]
    dynamics_sim = forwardSimulationGripper(gui=0)
    cage = Gripper(fake_data, dynamics_sim)
    x_init = fake_data
    dynamics_sim.finish_sim()
    sim = scriptedMovementSimGripper(cage, gui=gui)
if problem_name == 'GripperMulti':
    total_time = 2
    num_object = 5
    num_state_planner = (6+6)*num_object+4+1
    headers = ['num_traj', 'data_id',] + \
              ['xo', 'zo', 'yo', 'thetaxo', 'thetayo', 'thetazo', 
               'vxo', 'vyo', 'vzo', 'omegaxo', 'omegayo', 'omegazo',]*num_object + \
              ['j0', 'j1', 'j2', 'j3','zt',
               'jv0', 'jv1', 'jv2', 'jv3', 'vzt',]
    headers_metric = ['num_traj', 'data_id', ] + \
                     ['shortest_distance_1', 'S_stick_1', 'S_engage_1', 
                      'shortest_distance_3', 'S_stick_3', 'S_engage_3',]*num_object
    headers_success = ['num_traj', 'success_all_label', 'success_exists_label']
    headers_capture = ['num_traj', 'data_id', 'capture_exists_label', 'capture_all_label', 'lateral_friction_coef',]
    fake_data = ([0.0, 0.0, 0.9, 0.0, 0.0, 0.0] + [0.0,]*6)*num_object +\
                [0,1,0,1,] + [2.2,] + [0.0]*4 + [0.0,]
    dynamics_sim = forwardSimulationGripperMulti(gui=0)
    cage = GripperMulti(fake_data, dynamics_sim)
    x_init = fake_data
    dynamics_sim.finish_sim()
    sim = scriptedMovementSimGripperMulti(cage, gui=gui)
if problem_name == 'WaterSwing':
    headers = ['data_id', 'xo', 'yo', 'thetao', 'vxo', 'vyo', 'omegao', 'xg', 'yg', 'thetag', 'vxg', 'vyg', 'omegag']
    fake_data = [3.0, 5.5, 0.0, 0.0, 0.0, 0.0,
                 3.0, 4.3, 0.0, 0.0, 0.0, 0.0]
    dynamics_sim = forwardSimulationWaterSwing(gui=0)
    cage = WaterSwing(fake_data, dynamics_sim)
    x_init = [cage.x_range/2, cage.y_range/3-0.3, 0, 0, 0, 0,
              cage.x_range/2, cage.y_range/3, 0, 0, 0, 0]
    dynamics_sim.finish_sim()
    sim = scriptedMovementSimWaterSwing(cage, gui=gui)
if problem_name == 'Shuffling':
    total_time = 7.5
    headers = ['data_id', 'y', 'z', 'x', 'thetax', 'thetay', 'thetaz', 
               'vx', 'vy', 'vz', 'omegax', 'omegay', 'omegaz', 
               'j0', 'j1', 'j2', 'j3', 'jv0', 'jv1', 'jv2', 'jv3',
               'zg', 'vzg']
    num_state = len(headers)-1
    fake_data = [0.0, ]*num_state
    dynamics_sim = forwardSimulationShuffling(gui=0)
    cage = Shuffling(fake_data, dynamics_sim)
    x_init = [0, 4.55, 0, 0, np.pi/2, 0,
              0, 0, 0, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 0, 0,
              5.2, 0,
              ]
    dynamics_sim.finish_sim()
    sim = scriptedMovementSimShuffling(cage, gui=gui)

dataset = []
heuriset = []
success_labelset = []
capture_labelset = []
for i in range(num_trajs):
    if problem_name == 'BoxPivot' or problem_name == 'Gripper':
        sim.sample_init_state()
    elif problem_name == 'PlanePush' or problem_name == 'PlanePushMulti' or problem_name == 'BalanceGrasp' or problem_name == 'GripperMulti':
        x_init = sim.sample_init_state()
    
    sim.reset_states(x_init)
    if problem_name == 'BoxPivot':
        # _ = sim.run_forward_sim(num_via_points=1, do_cutdown_test=1, id_traj=i,) # get cutdown time
        # sim.reset_states(x_init)
        # x_news = sim.run_forward_sim(sim.cutoff_t, num_via_points, i, do_cutdown_test=0)
        x_news = sim.run_forward_sim(total_time, num_via_points, i, do_cutdown_test=0)
    else:
        x_news = sim.run_forward_sim(total_time, num_via_points, i)
    heuristics = sim.heuristics_traj

    for k in range(len(x_news)):
        dataset.append([i, k,] + x_news[k])
        heuriset.append([i, k,] + heuristics[k])
        if problem_name == 'PlanePush':
            cage = PlanePush(x_news[k], dynamics_sim)
        if problem_name == 'PlanePushMulti':
            cage = PlanePushMulti(x_news[k], dynamics_sim)
        elif problem_name == 'BalanceGrasp':
            cage = BalanceGrasp(x_news[k], dynamics_sim)
        elif problem_name == 'BoxPivot':
            cage = BoxPivot(x_news[k], dynamics_sim)
        elif problem_name == 'Gripper':
            cage = Gripper(x_news[k], dynamics_sim)
        elif problem_name == 'GripperMulti':
            cage = GripperMulti(x_news[k], dynamics_sim)
        capture_exists_label = 0 if cage.complementCaptureSet().contains(x_news[k][:num_state_planner]) else 1
        if problem_name == 'PlanePush':
            capture_labelset.append([i, k,] + [capture_exists_label, sim.lateral_friction_coef, sim.lateral_friction_coef_perturb])
        elif problem_name == 'PlanePushMulti':
            capture_all_label = 1 if cage.captureSet().contains(x_news[k][:num_state_planner]) else 0
            capture_labelset.append([i, k,] + [capture_exists_label, capture_all_label, sim.lateral_friction_coef])
        elif problem_name == 'BoxPivot':
            capture_labelset.append([i, k,] + [capture_exists_label, sim.lateral_friction_coef,])
        elif problem_name == 'Gripper':
            capture_labelset.append([i, k,] + [capture_exists_label, sim.lateral_friction_coef, sim.mass_object])
        elif problem_name == 'GripperMulti':
            capture_all_label = 1 if cage.captureSet().contains(x_news[k][:num_state_planner]) else 0
            capture_labelset.append([i, k,] + [capture_exists_label, capture_all_label, sim.lateral_friction_coef,])

    if problem_name == 'PlanePushMulti' or problem_name == 'GripperMulti':
        success_labelset.append([i, sim.success_all_label, sim.success_exists_label,])
    else:
        success_labelset.append([i, sim.success_all_label,])
sim.finish_sim()

# Save data to a CSV file with headers
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(dataset)

# Save heuristics to a CSV file with headers
with open(filename_metric, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers_metric)
    writer.writerows(heuriset)

# Save labels to a CSV file with headers
with open(filename_suc_label, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers_success)
    writer.writerows(success_labelset)

# Save labels to a CSV file with headers
with open(filename_capture_label, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers_capture)
    writer.writerows(capture_labelset)