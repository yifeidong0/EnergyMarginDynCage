from pomp.bullet.scriptedmovement import *
from pomp.example_problems.cageplanner import *
from pomp.example_problems.waterswing import *
from pomp.example_problems.planepush import *
from pomp.example_problems.planepushreal import *
from pomp.example_problems.planepushmulti import *
from pomp.example_problems.planepushrrtstar import *
from pomp.example_problems.balancegrasp import *
from pomp.example_problems.boxpivot import *
from pomp.example_problems.shuffling import *
from pomp.example_problems.gripper import *
from pomp.example_problems.grippermulti import *
from pomp.planners import allplanners
from pomp.visualizer import *
import time
import csv
from main import *
import numpy as np
from numpy.linalg import eig, inv

# !!! Things remember to do BEFORE running: pruning, quasistatic_motion, pChooseGoal, densityEstimationRadius, max_dimensions (ESTprojection), goal sets, costs...

plannername = 'ao-est' # 'ao-est', 'rrt*', 'ao-rrt'
prname = 'GripperMulti' # 'PlanePush', 'PlanePushRrtstar', 'PlanePushReal', 'PlanePushMulti', 'BalanceGrasp', 'BoxPivot', 'Gripper', 'GripperMulti', 'WaterSwing', 'Shuffling'
traj_type = 'scripted' # 'mppi', "scripted", "realworld"
vis = 0
maxTime = 10000 # only used when vis=0
maxIters = 300
init_id = 4 if traj_type == 'mppi' else 2 # 0 for scripted, 2 for mppi
record_capture_labels = 0
use_default_friction = 0
default_friction = 0.3

# Randomize the velocity and position of the objects and the friction coefficient - perturbation in estimated state
noise = 0.0
randomize_friction = 0
randomize_velocity = 0
randomize_position = 0
randomize_all = 0
randomize_ellipse_representation = 1
fri_ratio, vel_ratio, pos_ratio = 1.0, 4.0, 2.0 # noise in [0,1]

if prname == 'PlanePush' or prname == 'PlanePushRrtstar':
    if traj_type == 'scripted':
        # filenames = ['data/paper_vis/plane_push/scripted_movement_viapoints_PlanePush.csv',]
        # filename_friction = 'data/paper_vis/plane_push/scripted_movement_maneuver_labels_PlanePush.csv'
        filenames = ['scripted_movement_viapoints_PlanePush.csv',]
        filename_friction = 'scripted_movement_capture_labels_PlanePush.csv'
    if traj_type == 'mppi':
        filenames = ['data/18k_dataset_from_mppi/states_from_mppi.csv',]
if prname == 'PlanePushMulti':
    filenames = ['data/evaluation/workshop/ellipse_representation/10/scripted_movement_viapoints_PlanePushMulti.csv',]
    filename_friction = 'data/evaluation/workshop/ellipse_representation/10/scripted_movement_capture_labels_PlanePushMulti.csv'
if prname == 'PlanePushReal':
    scale_factor = 10
    if traj_type == 'realworld':
        # filenames = ['data/evaluation/real-world/circle-pushes-rectangle/02/apriltag_results.csv',]
        numbers = list(range(12))
        formatted_numbers = [f"{number:02}" for number in numbers]
        # filenames = ['data/evaluation/real-world/circle-pushes-rectangle/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        # filenames = ['data/evaluation/real-world/jaw-pushes-rectangle/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        # filenames = ['data/evaluation/real-world/circle-pushes-triangle/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        # filenames = ['data/evaluation/real-world/jaw-pushes-triangle/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        # filenames = ['data/evaluation/real-world/circle-pushes-convex/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        filenames = ['data/evaluation/real-world/jaw-pushes-convex/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        # filenames = ['data/evaluation/real-world/circle-pushes-concave/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        # filenames = ['data/evaluation/real-world/jaw-pushes-concave/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        # filenames = ['data/evaluation/real-world/circle-pushes-irregular/'+n+'/apriltag_results.csv' for n in formatted_numbers]
        # filenames = ['data/evaluation/real-world/jaw-pushes-irregular/'+n+'/apriltag_results.csv' for n in formatted_numbers]

if prname == 'BalanceGrasp':
    filenames = ['data/evaluation/balance_grasp/rand_traj_1/dataset/scripted_movement_viapoints_BalanceGrasp.csv',]
if prname == 'BoxPivot':
    filenames = ['data/evaluation/box_pivot/rand_fri_coeff/dataset/scripted_movement_viapoints_BoxPivot.csv',]
    filename_friction = 'data/evaluation/box_pivot/rand_fri_coeff/dataset/scripted_movement_maneuver_labels_BoxPivot.csv'
if prname == 'Gripper':
    filenames = ['data/evaluation/gripper/rand_objmass_fri/dataset/scripted_movement_viapoints_Gripper.csv',]
    filename_hyperparams = 'data/evaluation/gripper/rand_objmass_fri/dataset/scripted_movement_maneuver_labels_Gripper.csv'
if prname == 'GripperMulti':
    filenames = ['data/evaluation/workshop/ellipse_representation/gripper-5objects/scripted_movement_viapoints_GripperMulti.csv',]
    filename_hyperparams = 'data/evaluation/workshop/ellipse_representation/gripper-5objects/scripted_movement_capture_labels_GripperMulti.csv'

def move_along_longer_side(pose, d):
    xo, yo, thetao, xg, yg = pose
    
    # Calculate the direction of the longer side of the box
    box_direction = np.array([np.cos(thetao), np.sin(thetao)])
    
    # Add offset d along the longer side of the box to the gripper's position
    new_xg = xg + d * box_direction[0]
    new_yg = yg + d * box_direction[1]
    
    return [xo, yo, thetao, new_xg, new_yg]

def khachiyan_algorithm(P, tolerance=1e-6):
    (n, d) = np.shape(P)
    Q = np.vstack([np.copy(P.T), np.ones(n)])
    QT = Q.T
    err = 1.0 + tolerance
    u = np.ones(n) / n
    while err > tolerance:
        X = np.dot(np.dot(Q, np.diag(u)), QT)
        M = np.diag(np.dot(np.dot(QT, inv(X)), Q))
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
    center = np.dot(u, P)
    A = inv(np.dot(np.dot(P.T, np.diag(u)), P) - np.outer(center, center)) / d
    eigenvalues, eigenvectors = eig(A)
    radii = 1. / np.sqrt(eigenvalues)
    return center, radii, eigenvectors

def sample_points_in_ellipse(center, radii, rotation, num_samples=100):
    # Generate random points in a unit circle
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    r = np.sqrt(np.random.uniform(0, 1, num_samples))  # Square root of uniform distribution for area uniformity
    points = np.vstack((r * np.cos(angles), r * np.sin(angles))).T

    # Scale points to the ellipse using radii
    points *= radii

    # Rotate points according to the ellipse's rotation matrix
    points = np.dot(points, rotation.T)

    # Translate points by the center
    points += center

    return points

def sample_points_in_ellipsoid_3d(center, radii, rotation, num_samples=100):
    # Generate random points in a unit sphere using spherical coordinates
    u = np.random.uniform(0, 1, num_samples)
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    phi = np.arccos(1 - 2 * u)
    r = np.cbrt(np.random.uniform(0, 1, num_samples))  # Cube root for volume uniformity

    # Convert spherical to cartesian coordinates
    points = np.vstack((r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi))).T

    # Scale points by the radii of the ellipsoid
    points *= radii

    # Rotate points to align with the ellipsoid's axes
    points = np.dot(points, rotation.T)

    # Translate points to the center of the ellipsoid
    points += center

    return points
capture_exists_labels = []
for file_id, filename in enumerate(filenames):
    # Read from the CSV file
    rows = []
    ids = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)

        for id, row in enumerate(csv_reader):
            if traj_type == 'mppi':
                rows.append([float(d) for d in row[init_id:init_id+12]])
            elif traj_type == 'scripted':
                rows.append([float(d) for d in row[init_id:]])
            elif traj_type == 'realworld':
                original = [float(d) for d in row[2:8]+row[9:]]
                converted = [original[7]*scale_factor, original[6]*scale_factor, -original[8], original[10]*scale_factor, original[9]*scale_factor, -original[11],
                            original[1]*scale_factor, original[0]*scale_factor, -original[2], original[4]*scale_factor, original[3]*scale_factor, -original[5],
                            ]
                rows.append(converted)
            if traj_type == 'realworld':
                ids.append(id)
            else:
                ids.append(int(row[0]))

    if prname == 'PlanePush':
        fri_coeffs = []
        if use_default_friction:
            fri_coeffs = [default_friction] * len(rows)
        else:
            with open(filename_friction, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)
                for id, row in enumerate(csv_reader):
                    fri_coeffs.append(float(row[3]))
    if prname == 'PlanePushMulti':
        fri_coeffs = []
        with open(filename_friction, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for id, row in enumerate(csv_reader):
                fri_coeffs.append(float(row[4]))
    if prname == 'BoxPivot':
        fri_coeffs = []
        with open(filename_friction, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for id, row in enumerate(csv_reader):
                fri_coeffs.append(float(row[3]))
    if prname == 'Gripper':
        fri_coeffs = []
        obj_mass = []
        with open(filename_hyperparams, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for id, row in enumerate(csv_reader):
                fri_coeffs.append(float(row[3]))
                obj_mass.append(float(row[4]))
    if prname == 'GripperMulti':
        fri_coeffs = []
        obj_mass = []
        with open(filename_hyperparams, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for id, row in enumerate(csv_reader):
                fri_coeffs.append(float(row[4]))
    if prname == 'PlanePushReal':
        rows = rows[-100:] # Keep only the last 100 data points
    params = {'maxTime': maxTime}
    if 'maxTime' in params:
        del params['maxTime']

    t0 = time.time()

    for i, data_i in enumerate(rows):
        # Add velocity noise
        if randomize_velocity:
            data_i[3:6] = [d + random.uniform(-noise, noise) for d in data_i[3:6]]
            data_i[9:12] = [d + random.uniform(-noise, noise) for d in data_i[9:12]]
        # Add push point bias noise
        if randomize_position:
            pose_5d = data_i[:3] + data_i[6:8]
            new_pose_5d = move_along_longer_side(pose_5d, random.uniform(-noise, noise))
            data_i = new_pose_5d[:3] + data_i[3:6] + new_pose_5d[3:] + data_i[8:12]
        if randomize_all:
            fri_coeffs[i] = max(0.0, fri_coeffs[i] + random.uniform(-noise*fri_ratio, noise*fri_ratio))
            data_i[3:6] = [d + random.uniform(-noise*vel_ratio, noise*vel_ratio) for d in data_i[3:6]]
            data_i[9:12] = [d + random.uniform(-noise*vel_ratio, noise*vel_ratio) for d in data_i[9:12]]
            pose_5d = data_i[:3] + data_i[6:8]
            new_pose_5d = move_along_longer_side(pose_5d, random.uniform(-noise*pos_ratio, noise*pos_ratio))
            data_i = new_pose_5d[:3] + data_i[3:6] + new_pose_5d[3:] + data_i[8:12]
        if randomize_ellipse_representation and prname == 'PlanePushMulti':
            num_objects = int((len(data_i)-6)/6)
            positions = [data_i[0+6*i:2+6*i] for i in range((num_objects))]
            positions = np.array(positions)
            center, radii, rotation = khachiyan_algorithm(positions)
            points = sample_points_in_ellipse(center, radii, rotation, num_samples=num_objects)
            # plot_ellipse_and_points(center, radii, rotation, points, positions)
            for j in range(num_objects):
                data_i[0+6*j:2+6*j] = points[j]
                data_i[2+6*j:6+6*j] = [0,]*4
        if randomize_ellipse_representation and prname == 'GripperMulti':
            num_objects = int((len(data_i)-10)/12)
            positions = [data_i[0+12*i:3+12*i] for i in range((num_objects))]
            positions = np.array(positions)
            center, radii, rotation = khachiyan_algorithm(positions)
            points = sample_points_in_ellipsoid_3d(center, radii, rotation, num_samples=num_objects)
            # plot_ellipsoid_and_points(center, radii, rotation, points, positions)
            for j in range(num_objects):
                data_i[0+12*j:3+12*j] = points[j]
                data_i[3+12*j:12+12*j] = [0,]*9

        if prname == 'PlanePush':
            dynamics_sim = forwardSimulationPlanePush(gui=0)
            problem = PlanePushTest(dynamics_sim, data_i, save_hyperparams=1, lateral_friction_coef=fri_coeffs[i])
        if prname == 'PlanePushRrtstar':
            dynamics_sim = forwardSimulationPlanePushRrtstar(gui=0)
            problem = PlanePushRrtstarTest(dynamics_sim, data_i, save_hyperparams=1)
        if prname == 'PlanePushMulti':
            dynamics_sim = forwardSimulationPlanePushMulti(gui=0)
            problem = PlanePushMultiTest(dynamics_sim, data_i, save_hyperparams=1, lateral_friction_coef=fri_coeffs[i])
        if prname == 'PlanePushReal':
            dynamics_sim = forwardSimulationPlanePushReal(gui=0)
            problem = PlanePushRealTest(dynamics_sim, data_i, save_hyperparams=1)
            if record_capture_labels:
                num_state_planner = 9
                cage = PlanePush(data_i, dynamics_sim)
                capture_exists_label = 0 if cage.complementCaptureSet().contains(data_i[:num_state_planner]) else 1
                capture_exists_labels.append([file_id, i, capture_exists_label])
        if prname == 'BalanceGrasp':
            dynamics_sim = forwardSimulationBalanceGrasp(gui=0)
            problem = BalanceGraspTest(dynamics_sim, data_i, save_hyperparams=1)
        if prname == 'BoxPivot':
            dynamics_sim = forwardSimulationBoxPivot(gui=0)
            problem = BoxPivotTest(dynamics_sim, data_i, save_hyperparams=1, lateral_friction_coef=fri_coeffs[i])
        if prname == 'Gripper':
            dynamics_sim = forwardSimulationGripper(gui=0)
            problem = GripperTest(dynamics_sim, data_i, save_hyperparams=1, lateral_friction_coef=fri_coeffs[i], mass_object=obj_mass[i])
        if prname == 'GripperMulti':
            dynamics_sim = forwardSimulationGripperMulti(gui=0)
            problem = GripperMultiTest(dynamics_sim, data_i, save_hyperparams=1, lateral_friction_coef=fri_coeffs[i])

        if vis:
            runVisualizer(problem, type=plannername, **params)
        else:
            print(ids[i])
            testPlannerDefault(problem, prname, maxTime, maxIters, plannername, data_id=i, **params)
        dynamics_sim.finish_sim()
    
    t_elapsed = time.time() - t0
    print('Time elapsed: ', t_elapsed)

# Save labels to a CSV file with headers
if record_capture_labels:
    filename_capture_label = 'capture_labels_'+prname+'.csv'
    with open(filename_capture_label, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['traj_id', 'data_id', 'capture_label',])
        writer.writerows(capture_exists_labels)