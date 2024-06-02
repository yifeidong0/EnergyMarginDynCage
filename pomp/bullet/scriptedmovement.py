from .forwardsimulator import *
from ..structures.toolfunc import *
import random
import pybullet as p
import pybullet_data
import time
import math
import numpy as np
from PIL import Image

def check_bounds(x, bd):
    # Iterate over elements in x and corresponding bounds in bd
    for i, xi in enumerate(x):
        lower_bound, upper_bound = bd[i]
        if xi < lower_bound or xi > upper_bound:
            return False
    return True

def generate_random_point(bd):
    # Generate a random value for each dimension within its bounds
    random_point = []
    for bounds in bd:
        lower_bound, upper_bound = bounds
        random_value = np.random.uniform(lower_bound, upper_bound)
        random_point.append(random_value)
    return random_point


class scriptedMovementSimPlanePush(forwardSimulationPlanePush):
    def __init__(self, cage, gui=False):
        super().__init__(gui=gui)
        self.set_params(cage.params)
        self.create_shapes()
        self.setup_camera()

    def setup_camera(self):
        # Camera settings
        self.width_cam, self.height_cam = 640, 640
        fov = 35
        aspect = self.width_cam / self.height_cam
        near = 0.02
        far = 5

        # Camera position and orientation
        camera_eye = [1, 1, 4.3]  # Example values, adjust as needed
        camera_target = [1, 1, 1]  # Point the camera is looking at
        camera_up = [0, 1, 0]  # Up direction

        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        
    def sample_init_state(self):
        # init_neutral = [5.0, 4.3, 0.0, 0.0, 0.0, 0.0, 
        #                 5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        # data = [1, .4, 0., 0.0, 0.0, 0, # for paper visualization
        #         1, .1, 0.0, 0.0, 1, 0.2],
        # xo = random.uniform(4,6)
        # yo = random.uniform(6.2,8.2)
        xo = random.uniform(1,1) # paper_version
        yo = random.uniform(.4,.4)
        thetao = random.uniform(-math.pi/18, math.pi/18)
        vxo = random.uniform(-0.0, 0.0)
        vyo = random.uniform(-0.0, 0.0)
        omegao = random.uniform(-0.0, 0.0)
        xg = xo + random.uniform(-0.5, 0.5)
        yg = yo + random.uniform(-0.37, -0.35)
        vxg = random.uniform(-0.0, 0.0)
        vyg = random.uniform(0.2, .5)
        init_state = [xo, yo, thetao, vxo, vyo, omegao,
                      xg, yg, 0, vxg, vyg, 0]

        self.lateral_friction_coef = np.random.uniform(0.2,0.4)
        self.lateral_friction_coef_perturb = self.lateral_friction_coef + np.random.uniform(-0.1,0.1)
        p.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.objectUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.gripperUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.obstacleUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)        
        return init_state

    def run_forward_sim(self, total_time=10, num_via_points=20, id_traj=0, do_cutdown_test=False):
        num_steps = int(total_time * 240)  # Number of time steps
        interval = int(num_steps/num_via_points)
        interval = 3 if interval==0 else interval
        save_img_id = 0

        # Step the simulation
        via_points = []
        self.heuristics_traj = []
        self.success_all_label = 0
        for t in range(num_steps):
            # Apply external force
            self.pos_object,_ = p.getBasePositionAndOrientation(self.objectUid)
            self.pos_gripper,_ = p.getBasePositionAndOrientation(self.gripperUid)
            rand_force = [random.uniform(-0.4,0.4), self.lateral_friction_coef/0.3*random.uniform(7,11), 0]
            p.applyExternalForce(self.gripperUid, -1, 
                                rand_force,
                                #  [0,10,0] ,
                                self.pos_gripper, 
                                p.WORLD_FRAME)

            # Print object via-points along the trajectory for visualization
            if t % interval == 0 or t == int(t*240)-1:
                # Get the object and gripper states
                self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
                self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
                self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
                self.pos_gripper, self.quat_gripper = p.getBasePositionAndOrientation(self.gripperUid)
                self.eul_gripper = p.getEulerFromQuaternion(self.quat_gripper)
                self.vel_gripper,self.vel_ang_gripper = p.getBaseVelocity(self.gripperUid)

                # Get contact forces
                res = p.getContactPoints(self.gripperUid, self.objectUid)
                all_contact_normal_forces = [contact[9] for contact in res]
                contact_normal_force = sum(all_contact_normal_forces) if len(all_contact_normal_forces)>0 else 0.0
                s_engage = contact_normal_force
                contact_friction_force_xy = sum([contact[10] for contact in res]) if len(all_contact_normal_forces)>0 else 0 # friction along z is not considered
                # Sticking quality measure in the paper - Criteria for Maintaining Desired Contacts for Quasi-Static Systems
                s_stick = (self.lateral_friction_coef*contact_normal_force - abs(contact_friction_force_xy)) * math.cos(np.arctan(self.lateral_friction_coef))
                
                # Get bodies closest points distance
                dist = p.getClosestPoints(self.gripperUid, self.objectUid, 100)
                dist = np.linalg.norm(np.array(dist[0][5]) - np.array(dist[0][6])) if len(dist)>0 else 0
                
                self.heuristics_traj.append([dist, s_stick, s_engage,])
                new_states = [self.pos_object[0], self.pos_object[1], self.eul_object[2],
                            self.vel_object[0], self.vel_object[1], self.vel_ang_object[2],
                            self.pos_gripper[0], self.pos_gripper[1], self.eul_gripper[2], 
                            self.vel_gripper[0], self.vel_gripper[1], self.vel_ang_gripper[2]
                            ]
                via_points.append(new_states)

                # # Save camera images
                # img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                # image = Image.fromarray(img_arr)
                # image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/plane-push-sim/6-10-K-png/image_{id_traj}_{t:04d}.png')  # Save the image

            p.stepSimulation()

            # Record cutoff time for the manual scripted movement dataset
            object_reached = (abs(self.pos_object[1]-self.y_obstacle) < 0.2 + 0.01)
            gripper_reached = (abs(self.pos_gripper[1]-self.y_obstacle) < (0.1+0.01))
            if do_cutdown_test and (gripper_reached or object_reached):
                self.cutoff_t = t / 240.0 + 0.2
                return via_points
            if not do_cutdown_test and object_reached:
                self.success_all_label = 1
            if self.gui:
                time.sleep(2/240)

            # # Save camera images
            # if t % 5 == 0:
            #     img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
            #     image = Image.fromarray(img_arr)
            #     image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/plane-push-sim/6-trajs-png/image_{id_traj}_{save_img_id:04d}.png')  # Save the image
            #     save_img_id += 1

        return via_points


class scriptedMovementSimPlanePushMulti(forwardSimulationPlanePushMulti):
    def __init__(self, cage, gui=False):
        super().__init__(gui=gui)
        self.set_params(cage.params)
        self.create_shapes()
        self.setup_camera()

    def setup_camera(self):
        # Camera settings
        self.width_cam, self.height_cam = 640, 640
        fov = 35
        aspect = self.width_cam / self.height_cam
        near = 0.02
        far = 5

        # Camera position and orientation
        camera_eye = [1, 1, 4.3]  # Example values, adjust as needed
        camera_target = [1, 1, 1]  # Point the camera is looking at
        camera_up = [0, 1, 0]  # Up direction

        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Function to calculate the Euclidean distance between two points
    def distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def sample_init_state(self, min_distance = 0.1):
        # Generate points
        points = []
        while len(points) < self.num_objects:
            # Generate a new point
            new_point = [random.uniform(0.7, 1.3), random.uniform(0.6, 1.0), 0.0,
                         random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), 0.0]
            
            # Check if the new point is far enough from all existing points
            if all(self.distance(new_point, existing_point) > min_distance for existing_point in points):
                points.append(new_point)

        sum_xo = sum([point[0] for point in points])
        xg = sum_xo/self.num_objects + random.uniform(-0.0, 0.0)
        min_yo = min([point[1] for point in points])
        yg = min_yo + random.uniform(-0.3, -0.2)
        vxg = random.uniform(-0.1, 0.1)
        vyg = random.uniform(0.02, 0.06)
        alphag = random.uniform(-0.8, 0.8)
        omegag = random.uniform(-0.0, 0.0)
        points = [point for sublist in points for point in sublist] # remove inner bracket of points
        init_state = points + [xg, yg, alphag, vxg, vyg, omegag]

        self.lateral_friction_coef = np.random.uniform(0.4,0.4)
        # self.lateral_friction_coef_perturb = self.lateral_friction_coef + np.random.uniform(-0.1,0.1)
        p.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        for i in range(self.num_objects):
            p.changeDynamics(self.objectUid[i], -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                                rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.gripperUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.obstacleUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)        
        return init_state

    def run_forward_sim(self, total_time=10, num_via_points=20, id_traj=0, do_cutdown_test=False):
        num_steps = int(total_time * 240)  # Number of time steps
        interval = int(num_steps/num_via_points)
        interval = 3 if interval==0 else interval
        # save_img_id = 0

        # Step the simulation
        via_points = []
        self.heuristics_traj = []
        self.success_exists_label = 0
        self.success_all_label = 0
        for t in range(num_steps):
            # Apply external force
            for i in range(self.num_objects):
                self.pos_object[i],_ = p.getBasePositionAndOrientation(self.objectUid[i])
            self.pos_gripper,_ = p.getBasePositionAndOrientation(self.gripperUid)
            rand_force = [random.uniform(-0.4,0.4), self.lateral_friction_coef/0.3*random.uniform(7,11), 0]
            p.applyExternalForce(self.gripperUid, -1, 
                                rand_force,
                                self.pos_gripper, 
                                p.WORLD_FRAME)

            # Print object via-points along the trajectory for visualization
            if t % interval == 0 or t == int(t*240)-1:
                # Get the object and gripper states
                for i in range(self.num_objects):
                    self.pos_object[i], self.quat_object[i] = p.getBasePositionAndOrientation(self.objectUid[i])
                    self.eul_object[i] = p.getEulerFromQuaternion(self.quat_object[i]) # rad
                    self.vel_object[i], self.vel_ang_object[i] = p.getBaseVelocity(self.objectUid[i])
                self.pos_gripper, self.quat_gripper = p.getBasePositionAndOrientation(self.gripperUid)
                self.eul_gripper = p.getEulerFromQuaternion(self.quat_gripper)
                self.vel_gripper,self.vel_ang_gripper = p.getBaseVelocity(self.gripperUid)

                # Get contact forces
                force_scores = []
                for i in range(self.num_objects):
                    res = p.getContactPoints(self.gripperUid, self.objectUid[i])
                    all_contact_normal_forces = [contact[9] for contact in res]
                    contact_normal_force = sum(all_contact_normal_forces) if len(all_contact_normal_forces)>0 else 0.0
                    s_engage = contact_normal_force
                    contact_friction_force_xy = sum([contact[10] for contact in res]) if len(all_contact_normal_forces)>0 else 0 # friction along z is not considered
                    # Sticking quality measure in the paper - Criteria for Maintaining Desired Contacts for Quasi-Static Systems
                    s_stick = (self.lateral_friction_coef*contact_normal_force - abs(contact_friction_force_xy)) * math.cos(np.arctan(self.lateral_friction_coef))
                    
                    # Get bodies closest points distance
                    dist = p.getClosestPoints(self.gripperUid, self.objectUid[i], 100)
                    dist = np.linalg.norm(np.array(dist[0][5]) - np.array(dist[0][6])) if len(dist)>0 else 0

                    force_scores = force_scores + [dist, s_stick, s_engage,]
                self.heuristics_traj.append(force_scores)

                new_states = []
                for i in range(self.num_objects):
                    new_states = new_states + [self.pos_object[i][0], self.pos_object[i][1], self.eul_object[i][2],
                                            self.vel_object[i][0], self.vel_object[i][1], self.vel_ang_object[i][2],]
                new_states = new_states + [self.pos_gripper[0], self.pos_gripper[1], self.eul_gripper[2], 
                                        self.vel_gripper[0], self.vel_gripper[1], self.vel_ang_gripper[2],]
                via_points.append(new_states)

                # Save camera images
                img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                image = Image.fromarray(img_arr)
                image.save(f'/home/yif/Downloads/image_{id_traj}_{t:04d}.png')  # Save the image

            p.stepSimulation()

            # Record cutoff time for the manual scripted movement dataset
            all_objects_reached = 1 if all(abs(self.pos_object[i][1]-self.y_obstacle) < 0.1 + 0.03 for i in range(self.num_objects)) else 0
            exists_objects_reached = 0 if all(abs(self.pos_object[i][1]-self.y_obstacle) > 0.1 + 0.03 for i in range(self.num_objects)) else 1
            if all_objects_reached:
                self.success_all_label = 1
            if exists_objects_reached:
                self.success_exists_label = 1
            if self.gui:
                time.sleep(2/240)

            # Save camera images
            # if t % 5 == 0:
            #     img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
            #     image = Image.fromarray(img_arr)
            #     image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/plane-push-sim/6-trajs-png/image_{id_traj}_{save_img_id:04d}.png')  # Save the image
            #     save_img_id += 1

        return via_points


class scriptedMovementSimBalanceGrasp(forwardSimulationBalanceGrasp):
    def __init__(self, cage, gui=False):
        super().__init__(gui=gui)
        self.set_params(cage.params)
        self.create_shapes()
        self.setup_camera()
    
    def sample_init_state(self):
        # init_neutral = [5.0, 4.3, 0.0, 0.0, 0.0, 0, # point gripper with cylinder/box object
                        # 5.0, 4, 0.0, 0.0, 0.0, 0]
        xo = 5.5
        yo = 4.8
        vxo = random.uniform(-0.01, 0.01)
        vyo = random.uniform(-0.01, 0.01)
        xg = xo + random.uniform(-0.3, 0.3)
        yg = yo - 0.3
        # thetag = random.uniform(-math.pi/15, math.pi/15)
        vxg = random.uniform(-0.01, 0.01)
        vyg = random.uniform(-0.01, 0.01)
        # omegag = random.uniform(-0.01, 0.01)
        init_state = [xo, yo, 0, vxo, vyo, 0,
                      xg, yg, 0, vxg, vyg, 0]
        return init_state

    def setup_camera(self):
        # Camera settings
        self.width_cam, self.height_cam = 640, 640
        fov = 35
        aspect = self.width_cam / self.height_cam
        near = 0.02
        far = 5

        # Camera position and orientation
        camera_eye = [5.5, 5.5, 4.3]  # Example values, adjust as needed
        camera_target = [5.5, 5.5, 1]  # Point the camera is looking at
        camera_up = [0, 1, 0]  # Up direction

        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    def run_forward_sim(self, total_time=10, num_via_points=20, id_traj=0, taulim=1):
        num_steps = int(total_time * 240)  # Number of time steps
        interval = int(num_steps/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        self.heuristics_traj = []
        self.success_all_label = 0
        save_img_id = 0
        save_img_id_k = 0
        for t in range(num_steps):
            self.pos_gripper,_ = p.getBasePositionAndOrientation(self.gripperUid)
            p.applyExternalForce(self.gripperUid, -1, # gravity compensation
                                [0, -self.g*self.mass_gripper*math.sin(self.angle_slope), 0], 
                                self.pos_gripper, 
                                p.WORLD_FRAME)
            p.applyExternalForce(self.gripperUid, -1, # gravity compensation
                                [0, np.random.uniform(0,1)*self.mass_gripper*math.sin(self.angle_slope), 0], 
                                self.pos_gripper, 
                                p.WORLD_FRAME)
            p.applyExternalTorque(self.gripperUid, -1, 
                                [0,0,np.random.uniform(-taulim*self.mass_gripper,taulim*self.mass_gripper)],
                                # [0,0,800],
                                p.WORLD_FRAME)
            
            # Print object via-points along the trajectory for visualization
            if (t+1) % interval == 0:
                # Get the object and gripper states
                self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
                self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
                self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
                self.pos_gripper, self.quat_gripper = p.getBasePositionAndOrientation(self.gripperUid)
                self.eul_gripper = p.getEulerFromQuaternion(self.quat_gripper)
                self.vel_gripper,self.vel_ang_gripper = p.getBaseVelocity(self.gripperUid)

                # Get contact forces
                res = p.getContactPoints(self.gripperUid, self.objectUid)
                all_contact_normal_forces = [contact[9] for contact in res]
                contact_normal_force = sum(all_contact_normal_forces) if len(all_contact_normal_forces)>0 else 0.0
                s_engage = contact_normal_force
                contact_friction_force_xy = sum([contact[10] for contact in res]) if len(all_contact_normal_forces)>0 else 0 # friction along z is not considered
                # Sticking quality measure in the paper - Criteria for Maintaining Desired Contacts for Quasi-Static Systems
                s_stick = (self.lateral_friction_coef*contact_normal_force - abs(contact_friction_force_xy)) * math.cos(np.arctan(self.lateral_friction_coef))

                # Get bodies closest points distance
                # com_dist = np.linalg.norm(np.array(self.pos_gripper) - np.array(self.pos_object)) 
                dist = p.getClosestPoints(self.gripperUid, self.objectUid, 100)
                dist = np.linalg.norm(np.array(dist[0][5]) - np.array(dist[0][6])) if len(dist)>0 else 0
                
                self.heuristics_traj.append([dist, s_stick, s_engage,])
                new_states = [self.pos_object[0], self.pos_object[1], self.eul_object[2],
                            self.vel_object[0], self.vel_object[1], self.vel_ang_object[2],
                            self.pos_gripper[0], self.pos_gripper[1], self.eul_gripper[2], 
                            self.vel_gripper[0], self.vel_gripper[1], self.vel_ang_gripper[2]
                            ]
                via_points.append(new_states)

                # Save camera images
                img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                image = Image.fromarray(img_arr)
                image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/balance-grasp-sim/6-10-K-png/image_K_{id_traj}_{save_img_id_k:04d}.png')  # Save the image
                save_img_id_k += 1

            p.stepSimulation()
            if self.gui:
                time.sleep(2/240)

            # Save camera images
            if t % 5 == 0:
                img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                image = Image.fromarray(img_arr)
                # image.save(f'image_{t}.png')  # Save the image
                image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/balance-grasp-sim/6-trajs-png/image_{id_traj}_{save_img_id:04d}.png')  # Save the image
                save_img_id += 1

        # Record cutoff time for the manual scripted movement dataset
        dist = p.getClosestPoints(self.gripperUid, self.objectUid, 100)
        dist = np.linalg.norm(np.array(dist[0][5]) - np.array(dist[0][6])) if len(dist)>0 else 0
        object_balanced = (dist < 1e-1)
        # gripper_reached = (abs(self.pos_gripper[1]-self.y_obstacle) < (0.1+0.01))
        # if do_cutdown_test and (gripper_reached or object_reached):
        #     self.cutoff_t = t / 240.0 + 0.2
        #     return via_points
        # if not do_cutdown_test and object_reached:
        if object_balanced:
            self.success_all_label = 1

        return via_points


class scriptedMovementSimBoxPivot(forwardSimulationBoxPivot):
    def __init__(self, cage, gui=False):
        super().__init__(gui=gui)
        self.cage = cage
        self.set_params(cage.params)
        self.create_shapes()
        self.setup_camera()

    def setup_camera(self):
        # Camera settings
        self.width_cam, self.height_cam = 640, 640
        fov = 40
        aspect = self.width_cam / self.height_cam
        near = 0.02
        far = 5

        # Camera position and orientation
        camera_eye = [3., -4, .7]  # Example values, adjust as needed
        camera_target = [3., 1, .7]  # Point the camera is looking at
        camera_up = [0, 0, 1]  # Up direction

        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    def sample_init_state(self):
        if self.for_paper_vis:
            self.lateral_friction_coef = np.random.uniform(0.6,1.5)
        else:
            self.lateral_friction_coef = np.random.uniform(0.5,1.5)
        p.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.objectUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)

    def run_forward_sim(self, total_time=5, num_via_points=20, id_traj=0, do_cutdown_test=False):
        num_steps = int(total_time * 240)  # Number of time steps
        interval = int(num_steps/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        self.heuristics_traj = []
        via_points = []
        self.success_all_label = 0
        if do_cutdown_test: self.rand_forces = []
        self.cutoff_t = total_time
        save_img_id = 0
        save_img_id_k = 0
        for t in range(num_steps):
            if do_cutdown_test:
                if self.for_paper_vis:
                    rand_force = [random.uniform(1,2), 0, 0] 
                else:
                    rand_force = [random.uniform(1,7), 0, 0]
                self.rand_forces.append(rand_force)
                p.applyExternalForce(self.gripperUid1, -1, # push spring
                                    # [2,0,0], # force
                                    rand_force,
                                    [0,0,0], 
                                    p.LINK_FRAME)
            else:
                p.applyExternalForce(self.gripperUid1, -1, # push spring
                                    # self.rand_forces[t], # force
                                    [random.uniform(1,1), 0, 0],
                                    [0,0,0], 
                                    p.LINK_FRAME)
            # Get positions of the boxes
            self.pos_gripper1, _ = p.getBasePositionAndOrientation(self.gripperUid1)
            self.pos_gripper2, _ = p.getBasePositionAndOrientation(self.gripperUid2)
            
            # Update the maxForce for the spring constraint
            maxForce = self.k * np.abs(np.linalg.norm(np.array(self.pos_gripper2) - np.array(self.pos_gripper1)) - self.rest_length)
            p.changeConstraint(self.c_spring, maxForce=maxForce)

            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if (t+1) % interval == 0:
                # Get the object and gripper states
                self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
                self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
                self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
                self.pos_gripper1, _ = p.getBasePositionAndOrientation(self.gripperUid1)
                self.pos_gripper2, _ = p.getBasePositionAndOrientation(self.gripperUid2)
                self.vel_gripper1,_ = p.getBaseVelocity(self.gripperUid1)
                self.vel_gripper2,_ = p.getBaseVelocity(self.gripperUid2)

                # Get contact forces
                heuristics = []
                for contact in [self.gripperUid2, self.planeUid]:
                    res = p.getContactPoints(contact, self.objectUid)
                    all_contact_normal_forces = [contact[9] for contact in res]
                    contact_normal_force = sum(all_contact_normal_forces) if len(all_contact_normal_forces)>0 else 0.0
                    s_engage = contact_normal_force
                    contact_friction_force_xy = sum([contact[12] for contact in res]) if len(all_contact_normal_forces)>0 else 0 # friction along z is not considered
                    # Sticking quality measure in the paper - Criteria for Maintaining Desired Contacts for Quasi-Static Systems
                    s_stick = (self.lateral_friction_coef*contact_normal_force - abs(contact_friction_force_xy)) * math.cos(np.arctan(self.lateral_friction_coef))

                    # Get bodies closest points distance
                    dist = p.getClosestPoints(contact, self.objectUid, 100)
                    dist = np.linalg.norm(np.array(dist[0][5]) - np.array(dist[0][6])) if len(dist)>0 else 0
                    heuristics.append(dist)
                    heuristics.append(s_stick)
                    heuristics.append(s_engage)
                
                self.heuristics_traj.append(heuristics)

                point_relative_position = [2, 0, -2]
                angular_velocity = self.vel_ang_object
                linear_velocity = self.vel_object
                point_velocity = linear_velocity + np.cross(angular_velocity, point_relative_position)
                point_velocity = np.linalg.norm(point_velocity)
                new_states = [self.pos_object[0], self.pos_object[2], self.eul_object[1],
                            self.vel_object[0], self.vel_object[2], self.vel_ang_object[1],
                            self.pos_gripper1[0], self.pos_gripper2[0],  
                            self.vel_gripper1[0], self.vel_gripper2[0], 
                            ]
                via_points.append(new_states)

                # Save camera images
                img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                image = Image.fromarray(img_arr)
                # image.save(f'image_{t}.png')  # Save the image
                image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/box-pivot-sim/6-10-K-png/image_K_{id_traj}_{save_img_id_k:04d}.png')  # Save the image
                save_img_id_k += 1

            # Save camera images
            if t % 3 == 0:
                img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                image = Image.fromarray(img_arr)
                # image.save(f'image_{t}.png')  # Save the image
                image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/box-pivot-sim/6-trajs-png/image_{id_traj}_{save_img_id:04d}.png')  # Save the image
                save_img_id += 1

            self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
            self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
            object_reached = self.eul_object[1] > math.pi/2-0.5*self.cage.task_goal_margin
            if do_cutdown_test and object_reached:
                self.cutoff_t = t / 240.0
                return via_points
            if self.gui:
                time.sleep(1/240)
                
        self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
        self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
        if self.eul_object[1] > math.pi/2-self.cage.task_goal_margin:
            self.success_all_label = 1
        return via_points
    

class scriptedMovementSimGripper(forwardSimulationGripper):
    def __init__(self, cage, gui=False):
        super().__init__(gui=gui)
        self.cage = cage
        self.set_params(cage.params)
        self.create_shapes()
        self.setup_camera()
    
    def setup_camera(self):
        # Camera settings
        self.width_cam, self.height_cam = 640, 640
        fov = 60
        aspect = self.width_cam / self.height_cam
        near = 0.02
        far = 5

        # Camera position and orientation
        camera_eye = [-1, 3.3, 1.3]  # Example values, adjust as needed
        camera_target = [-.5, 0, 1]  # Point the camera is looking at
        camera_up = [0, 0, 1]  # Up direction

        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    def sample_init_state(self):
        self.mass_object = np.random.uniform(1,5)
        self.lateral_friction_coef = np.random.uniform(0.1,1)
        print("lateral_friction_coef: ", self.lateral_friction_coef)
        print("mass_object: ", self.mass_object)
        print("")
        p.changeDynamics(self.objectUid, -1, mass=self.mass_object) # fix the base link
        for i in range(-1, self.num_links-1):
            p.changeDynamics(self.gripperUid, i, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)

    def run_forward_sim(self, total_time=5, num_via_points=10, id_traj=0, clench_time=.05, lift_time=.3, lift_acc=-.7):
        num_steps = int(total_time * 240)  # Number of time steps
        interval = int(num_steps/num_via_points)

        # Step the simulation
        self.heuristics_traj = []
        via_points = []
        self.success_all_label = 0
        self.constraint_added = 0
        save_img_id_k = 0
        save_img_id = 0
        for t in range(num_steps):
            # Apply the calculated torques to all joints at once  
            if t > clench_time*240:
                p.setJointMotorControlArray(bodyUniqueId=self.gripperUid,
                                            jointIndices=self.movable_joints,
                                            controlMode=p.POSITION_CONTROL,
                                            # targetPositions=self.start_gripper_pos,
                                            targetPositions=[math.pi/8,]*len(self.start_gripper_pos),
                                            targetVelocities=[0,]*self.num_movable_joints,
                                            positionGains=self.stiffness,
                                            velocityGains=self.damping,
                                            forces=[10,]*self.num_movable_joints,)
                
            if t > lift_time*240:
                # Apply downward force on the table to equavalently lift the gripper
                p.changeDynamics(self.tableUid, -1, mass=self.mass_table) # unfix the table
                self.pos_table, _ = p.getBasePositionAndOrientation(self.tableUid)
                p.applyExternalForce(self.tableUid, -1, 
                                    [0, 0, self.mass_table*(lift_acc-self.g)], # gravity compensation and lifting force
                                    self.pos_table, 
                                    p.WORLD_FRAME)
                pivotInBox = [0, 0, 0] 
                pivotInWorld = [0, 0, 0]
                if not self.constraint_added: # multiple runs slow down the scripted-movement simulation
                    constraint1 = p.createConstraint(self.tableUid, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], pivotInBox, pivotInWorld)
                    self.constraint_added = 1
            else:
                p.changeDynamics(self.tableUid, -1, mass=0) # fix the table

            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if (t+1) % interval == 0:
                # Get the object and gripper states
                pos, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
                pos_object_GL = [pos[0], pos[2], pos[1]] # x,z,y
                self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
                self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
                joint_states = p.getJointStates(self.gripperUid, self.movable_joints)
                self.pos_gripper = [state[0] for state in joint_states]
                self.vel_gripper = [state[1] for state in joint_states]
                pos_table, _ = p.getBasePositionAndOrientation(self.tableUid)
                vel_table, _ = p.getBaseVelocity(self.tableUid)

                # Get contact forces
                heuristics = []
                for id in self.fingertip_link_ids:
                    res = p.getContactPoints(self.gripperUid, self.objectUid, id, -1)
                    all_contact_normal_forces = [contact[9] for contact in res]
                    contact_normal_force = sum(all_contact_normal_forces) if len(all_contact_normal_forces)>0 else 0.0
                    s_engage = contact_normal_force # engage metric
                    contact_friction_force_1 = sum([contact[10] for contact in res]) if len(all_contact_normal_forces)>0 else 0
                    contact_friction_force_2 = sum([contact[12] for contact in res]) if len(all_contact_normal_forces)>0 else 0
                    contact_friction_force = np.sqrt(contact_friction_force_1**2 + contact_friction_force_2**2)

                    # Sticking quality measure in the paper - Criteria for Maintaining Desired Contacts for Quasi-Static Systems
                    s_stick = (self.lateral_friction_coef*contact_normal_force - contact_friction_force) * math.cos(np.arctan(self.lateral_friction_coef))

                    # Get bodies closest points distance
                    res = p.getClosestPoints(self.gripperUid, self.objectUid, 100, id, -1) # 3,7,11
                    dist = res[0][8] if (len(res)>0) else 0 # and res[0][8]>=0
                    heuristics.append(dist)
                    heuristics.append(s_stick)
                    heuristics.append(s_engage)

                self.heuristics_traj.append(heuristics)
                new_states = (pos_object_GL + list(self.eul_object) + list(self.vel_object) + list(self.vel_ang_object) + self.pos_gripper
                              + [pos_table[2],] + self.vel_gripper + [vel_table[2],])
                via_points.append(new_states)

                # Save camera images
                img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                image = Image.fromarray(img_arr)
                image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/gripper-sim/6-10-K-png/image_K_{id_traj}_{save_img_id_k:04d}.png')  # Save the image
                save_img_id_k += 1

            if self.gui:
                time.sleep(1/240)

            # Save camera images
            if t % 4 == 0:
                img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                image = Image.fromarray(img_arr)
                # image.save(f'image_{t}.png')  # Save the image
                image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/gripper-sim/6-trajs-png/image_{id_traj}_{save_img_id:04d}.png')  # Save the image
                save_img_id += 1
        
        self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
        self.vel_object, _ = p.getBaseVelocity(self.objectUid)
        if self.vel_object[2] > -0.5 and self.pos_object[2] > 0.25:
            self.success_all_label = 1
        return via_points
    

class scriptedMovementSimGripperMulti(forwardSimulationGripperMulti):
    def __init__(self, cage, gui=False):
        super().__init__(gui=gui)
        self.cage = cage
        self.set_params(cage.params)
        self.create_shapes()
        self.setup_camera()
    
    def setup_camera(self):
        # Camera settings
        self.width_cam, self.height_cam = 640, 640
        fov = 60
        aspect = self.width_cam / self.height_cam
        near = 0.02
        far = 5

        # Camera position and orientation
        camera_eye = [2, 2.3, 1.7]  # Example values, adjust as needed
        camera_target = [0, 0, 1]  # Point the camera is looking at
        camera_up = [0, 0, 1.3]  # Up direction

        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    def distance(self, p1, p2):
        """Function to calculate the Euclidean distance between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5

    def sample_init_state(self):
        # TODO: randomize object positions
        min_distance = self.length_object * 1.5
        points = []
        while len(points) < self.num_objects:
            # Generate a new point
            new_point = [random.uniform(-0.4, 0.4), random.uniform(-0.1, 0.1), random.uniform(0.4, 0.4), 0.0, 0.0, 0.0,
                         random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(0.05, 0.05), 0.0, 0.0, 0.0,]
            
            # Check if the new point is far enough from all existing points
            if all(self.distance(new_point, existing_point) > min_distance for existing_point in points):
                points.append(new_point)

        init_state = []
        for i in range(self.num_objects):
            init_state = init_state + points[i]
        init_state = init_state + [0.0,0.0,0.0,0.0,] + [2.2,] + [0.0]*4 + [0.0,]

        self.lateral_friction_coef = np.random.uniform(0.1,1)
        for i in range(-1, self.num_links-1):
            p.changeDynamics(self.gripperUid, i, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        
        return init_state

    def run_forward_sim(self, total_time=5, num_via_points=10, id_traj=0, pre_time=0.5,):
        num_steps = int(total_time * 240)  # Number of time steps
        num_via_points = 30
        interval = int((num_steps-pre_time*240) / num_via_points)

        # Step the simulation
        self.heuristics_traj = []
        via_points = []
        self.constraint_added = 0
        save_img_id_k = 0
        save_img_id = 0
        for t in range(num_steps):
            # Apply the calculated torques to all joints at once  
            p.setJointMotorControlArray(bodyUniqueId=self.gripperUid,
                                        jointIndices=self.movable_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self.target_gripper_joint_pos,
                                        targetVelocities=[0.0,]*len(self.movable_joints),
                                        positionGains=self.stiffness,
                                        velocityGains=self.damping,
                                        # controlMode=p.TORQUE_CONTROL,
                                        forces=[1.0,]*len(self.movable_joints),)
            
            # if t == pre_time*240:
            #     joint_states = p.getJointStates(self.gripperUid, self.movable_joints)
            #     joint_positions = [state[0] for state in joint_states]
            #     self.target_gripper_joint_pos = joint_positions

            # apply upward force on the gripper
            self.pos_gripper_base, _ = p.getBasePositionAndOrientation(self.gripperUid)
            if t < pre_time*240:
                p.applyExternalForce(objectUniqueId=self.gripperUid, linkIndex=-1, forceObj=[0,0,-self.force_z_on_gripper], posObj=self.pos_gripper_base, flags=p.WORLD_FRAME)
            else:
                p.applyExternalForce(objectUniqueId=self.gripperUid, linkIndex=-1, forceObj=[0,0,self.force_z_on_gripper], posObj=self.pos_gripper_base, flags=p.WORLD_FRAME)

            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if (t+1) % interval == 0:
                # Get the object and gripper states
                for i in range(self.num_objects):
                    self.pos_object[i], self.quat_object[i] = p.getBasePositionAndOrientation(self.objectUid[i])
                    # pos_object_GL = [pos[0], pos[2], pos[1]] # x,z,y
                    self.eul_object[i] = p.getEulerFromQuaternion(self.quat_object[i]) # rad
                    self.vel_object[i], self.vel_ang_object[i] = p.getBaseVelocity(self.objectUid[i])

                joint_states = p.getJointStates(self.gripperUid, self.movable_joints)
                self.pos_gripper = [state[0] for state in joint_states]
                self.vel_gripper = [state[1] for state in joint_states]
                self.pos_gripper_base, _ = p.getBasePositionAndOrientation(self.gripperUid)
                self.vel_gripper_base, _ = p.getBaseVelocity(self.gripperUid)
                
                # Get contact forces
                heuristics = []
                for i in range(self.num_objects):
                    for id in self.fingertip_link_ids:
                        res = p.getContactPoints(self.gripperUid, self.objectUid[i], id, -1)
                        all_contact_normal_forces = [contact[9] for contact in res]
                        contact_normal_force = sum(all_contact_normal_forces) if len(all_contact_normal_forces)>0 else 0.0
                        s_engage = contact_normal_force # engage metric
                        contact_friction_force_1 = sum([contact[10] for contact in res]) if len(all_contact_normal_forces)>0 else 0
                        contact_friction_force_2 = sum([contact[12] for contact in res]) if len(all_contact_normal_forces)>0 else 0
                        contact_friction_force = np.sqrt(contact_friction_force_1**2 + contact_friction_force_2**2)

                        # Sticking quality measure in the paper - Criteria for Maintaining Desired Contacts for Quasi-Static Systems
                        s_stick = (self.lateral_friction_coef*contact_normal_force - contact_friction_force) * math.cos(np.arctan(self.lateral_friction_coef))

                        # Get bodies closest points distance
                        res = p.getClosestPoints(self.gripperUid, self.objectUid[i], 100, id, -1) # 1,3,
                        dist = res[0][8] if (len(res)>0) else 0 # and res[0][8]>=0
                        heuristics.append(dist)
                        heuristics.append(s_stick)
                        heuristics.append(s_engage)
                self.heuristics_traj.append(heuristics)

                new_states = []
                for i in range(self.num_objects):
                    new_states = new_states + list(self.pos_object[i]) + list(self.eul_object[i]) + list(self.vel_object[i]) + list(self.vel_ang_object[i])
                new_states = new_states + self.pos_gripper + [self.pos_gripper_base[2],] + self.vel_gripper + [self.vel_gripper_base[2],]
                via_points.append(new_states)

                # # Save camera images
                img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
                image = Image.fromarray(img_arr)
                image.save(f'/home/yif/Downloads/image_K_{id_traj}_{save_img_id_k:04d}.png')  # Save the image
                save_img_id_k += 1

            if self.gui:
                time.sleep(3/240)

            # # Save camera images
            # if t % 4 == 0:
            #     img_arr = p.getCameraImage(self.width_cam, self.height_cam, self.view_matrix, self.projection_matrix)[2]  # Capture the image
            #     image = Image.fromarray(img_arr)
            #     # image.save(f'image_{t}.png')  # Save the image
            #     image.save(f'/home/yif/Documents/KTH/research/dynamicCage/submission/sup-video/gripper-sim/6-trajs-png/image_{id_traj}_{save_img_id:04d}.png')  # Save the image
            #     save_img_id += 1
        
        for i in range(self.num_objects):
            self.pos_object[i], self.quat_object[i] = p.getBasePositionAndOrientation(self.objectUid[i])
            # self.vel_object[i], _ = p.getBaseVelocity(self.objectUid[i])
        self.success_all_label = 1 if all(self.pos_object[i][2] > self.success_z_thres for i in range(self.num_objects)) else 0
        self.success_exists_label = 0 if all(self.pos_object[i][2] <= self.success_z_thres for i in range(self.num_objects)) else 1
        return via_points


class scriptedMovementSimWaterSwing(forwardSimulationWaterSwing):
    def __init__(self, cage, gui=False):
        super().__init__(gui=gui)
        p.setGravity(0, 0, self.g)

        self.set_params(cage.params)
        self.create_shapes()

    def run_forward_sim(self, total_time=10, num_via_points=20):
        num_steps = int(total_time * 240)  # Number of time steps
        radius = 3.0  # Radius of the circular path
        velocity = 2 * np.pi * radius/ total_time  # Radians per second for a full circle
        angular_velocity = 2 * np.pi / total_time  # Radians per second for a full circle
        initial_angular_velocity = -2 * np.pi / total_time  # Initial velocity for a full circle
        vel_gripper = [velocity, 0, 0]
        vel_angular_gripper = [0, initial_angular_velocity, 0]
        p.resetBaseVelocity(self.gripperUid, vel_gripper, vel_angular_gripper) # linear and angular vels both in world coordinates

        dt = total_time / num_steps # 1/240

        interval = int(num_steps/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        
        for t in range(num_steps):
            # Calculate the current angular displacement
            theta = angular_velocity * t * dt
            
            # Calculate centripetal acceleration
            centripetal_acceleration = (velocity ** 2) / radius

            # Calculate force components along x and y axes (centripetal force)
            force_x = -centripetal_acceleration * self.mass_gripper * np.sin(theta)
            force_z = centripetal_acceleration*self.mass_gripper*np.cos(theta) + self.mass_gripper*(-self.g)

            # Apply external force
            self.pos_gripper,_ = p.getBasePositionAndOrientation(self.gripperUid)
            self.pos_object,_ = p.getBasePositionAndOrientation(self.objectUid)
            p.applyExternalForce(self.gripperUid, -1, 
                                [force_x, 0, force_z], 
                                self.pos_gripper, 
                                p.WORLD_FRAME)

            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if t % interval == 0 or t == int(t*240)-1:
                # Get the object and gripper states
                self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
                self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
                self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
                self.pos_gripper, self.quat_gripper = p.getBasePositionAndOrientation(self.gripperUid)
                self.eul_gripper = p.getEulerFromQuaternion(self.quat_gripper)
                self.vel_gripper,self.vel_ang_gripper = p.getBaseVelocity(self.gripperUid)

                new_states = [self.pos_object[0], self.pos_object[2], correct_euler(self.eul_object)[1],
                            self.vel_object[0], self.vel_object[2], self.vel_ang_object[1],
                            self.pos_gripper[0], self.pos_gripper[2], correct_euler(self.eul_gripper)[1], 
                            self.vel_gripper[0], self.vel_gripper[2], self.vel_ang_gripper[1]
                            ]
                via_points.append(new_states)

            if self.gui:
                time.sleep(2/240)

        return via_points


class scriptedMovementSimShuffling(forwardSimulationShuffling):
    def __init__(self, cage, gui=False):
        super().__init__(gui=gui)
        # p.setGravity(0, 0, self.g)

        self.set_params(cage.params)
        self.create_shapes()
        self.t_start_side_push = 200

    def run_forward_sim(self, total_time=6, num_via_points=10):
        num_steps = int(total_time * 240)  # Number of time steps
        interval = int(num_steps/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        for t in range(num_steps):
            # Push the card stack from the side
            # linkBasePosition, _ = p.getBasePositionAndOrientation(self.objectUid)
            if t>self.t_start_side_push and t < self.t_start_side_push+1*240:
                p.applyExternalForce(self.objectUid, 1,
                                    [0,-1,0], 
                                    [0,0,0], 
                                    p.LINK_FRAME)

            # Squeezing the card stack
            joint_states = p.getJointStates(self.objectUid, list(range(self.num_joints)))
            joint_angles = [state[0] for state in joint_states]
            upperPosition, _ = p.getBasePositionAndOrientation(self.gripperUid)
            if t < 10*240:
                force = [0,0,-4/(sum(joint_angles)+0.2)] # simple feedback control
                p.applyExternalForce(self.gripperUid, -1,
                                force, 
                                upperPosition, 
                                p.WORLD_FRAME)
                
            # Calculate the torques induced by joint elasticity and external force

            joint_vels = [state[1] for state in joint_states]
            error_angles = [desired - current for desired, current in zip(self.target_positions, joint_angles)]
            error_vels = [desired - current for desired, current in zip(self.target_velocities, joint_vels)]
            target_torques = [kp*e - kd*av for kp, e, kd, av in zip(self.stiffness, error_angles, self.damping, error_vels)]

            # Mimic elastic cards stack behavior
            p.setJointMotorControlArray(bodyUniqueId=self.objectUid,
                                        jointIndices=list(range(self.num_joints)),
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[100,]*self.num_joints,
                                        forces=target_torques,)

            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if t % interval == 0 or t == int(t*240)-1:
                pos, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
                pos_object_GL = [pos[1], pos[2], pos[0]] # y,z,x
                self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
                self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
                joint_states = p.getJointStates(self.objectUid, list(range(self.num_joints)))
                self.pos_object_joints = [state[0] for state in joint_states]
                self.vel_object_joints = [state[1] for state in joint_states]
                self.pos_gripper, _ = p.getBasePositionAndOrientation(self.gripperUid)
                self.vel_gripper, _ = p.getBaseVelocity(self.gripperUid)

                new_states = (pos_object_GL + list(self.eul_object) + list(self.vel_object) + list(self.vel_ang_object) + 
                              self.pos_object_joints + self.vel_object_joints + [self.pos_gripper[2], self.vel_gripper[2]]
                              )
                via_points.append(new_states)

            if self.gui:
                time.sleep(2/240)

        return via_points
