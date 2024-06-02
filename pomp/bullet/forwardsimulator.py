import pybullet as p
import pybullet_data
import time
import math
from ..structures.toolfunc import *
import numpy as np
import time

class forwardSimulationPlanePush():
    def __init__(self, gui=False):
        self.visualShapeId = -1
        self.gui = gui
        if self.gui:
            p.connect(p.GUI) # p.GUI
        else:
            p.connect(p.DIRECT) # p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
        self.g = -9.81
        # self.set_params(params)
        # self.create_shapes()
    
    def set_params(self, params):
        self.angle_slope = params[5] # equivalent to on a slope
        p.setGravity(0, 
                     self.g*math.sin(self.angle_slope),
                     self.g*math.cos(self.angle_slope),)

        # Kinodynamics
        self.mass_object = params[0]
        self.moment_object = params[1] # moment of inertia
        self.pos_object = [1000,0,0]
        self.quat_object = p.getQuaternionFromEuler([math.pi/2,0,0])
        self.vel_object = [.1,0,0]
        
        self.mass_gripper = params[2]
        self.moment_gripper = params[3]
        self.y_obstacle = params[4]
        self.object_name = params[6] # 'box', 'cylinder'
        self.gripper_name = params[7] # 'box', 'cylinder', 'bowl'
        self.lateral_friction_coef = params[8]

        self.pos_gripper = [1000,0,2]
        self.quat_gripper = p.getQuaternionFromEuler([0,0,0])
        self.vel_gripper = [0,0,0]
        self.vel_ang_gripper = [0,0,0]

        self.z_bodies = .05
        self.half_extent_obstacle = [7, .5, self.z_bodies]
        self.pos_obstacle = [5,self.y_obstacle+self.half_extent_obstacle[1],0]
        self.quat_obstacle = p.getQuaternionFromEuler([0,0,0])

    def create_shapes(self):
        # Create a plane
        #  The default frictional coefficients used by PyBullet are 0.5 for the lateral friction and 0 for both the rolling and spinning.
        # self.planeUid = p.loadURDF("plane.urdf", basePosition=[0,0,-self.z_bodies])
        planeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, self.z_bodies])
        self.planeUid = p.createMultiBody(0, 
                                            planeId, 
                                            self.visualShapeId, 
                                            [0,0,-2*self.z_bodies],
                                            p.getQuaternionFromEuler([0,0,0]))
        p.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        # p.changeVisualShape(self.planeUid, -1, rgbaColor=[193/255, 193/255, 193/255, 1])

        # Create an object
        if self.object_name == 'box':
            objectId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.6, .2, self.z_bodies])
            self.objectUid = p.createMultiBody(self.mass_object, 
                                               objectId, 
                                               self.visualShapeId, 
                                               self.pos_object,
                                               self.quat_object)
        elif self.object_name == 'cylinder':
            objectId = p.createCollisionShape(p.GEOM_CYLINDER, radius=.2, height=2*self.z_bodies)
            self.objectUid = p.createMultiBody(self.mass_object, 
                                            objectId, 
                                            self.visualShapeId, 
                                            self.pos_object,
                                            self.quat_object)
        p.changeDynamics(self.objectUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.objectUid, -1, rgbaColor=[247/255, 143/255, 0/255, 1])
        
        # Create a robot
        if self.gripper_name == 'box':
            gripperId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.6, .1, self.z_bodies])
            self.gripperUid = p.createMultiBody(self.mass_gripper, 
                                           gripperId, 
                                           self.visualShapeId, 
                                           self.pos_gripper,
                                           self.quat_gripper)
        elif self.gripper_name == 'cylinder':
            gripperId = p.createCollisionShape(p.GEOM_CYLINDER, radius=.1, height=2*self.z_bodies)
            self.gripperUid = p.createMultiBody(self.mass_gripper,
                                                gripperId,
                                                self.visualShapeId,
                                                self.pos_gripper,
                                                self.quat_gripper)
        elif self.gripper_name == 'bowl':
            self.gripperUid = p.loadURDF("asset/bowl/2d-bowl.urdf", self.pos_gripper, self.quat_gripper, globalScaling=1)
        p.changeDynamics(self.gripperUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.gripperUid, -1, rgbaColor=[112/255, 190/255, 83/255, 1])
        
        # Create a static obstacle
        obstacleId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extent_obstacle)
        self.obstacleUid = p.createMultiBody(0, 
                                       obstacleId, 
                                       self.visualShapeId, 
                                       self.pos_obstacle,
                                       self.quat_obstacle)
        p.changeDynamics(self.obstacleUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.obstacleUid, -1, rgbaColor=[.3,.3,.3,1])
        
    def reset_states(self, states):
        xo, yo, thetao, vxo, vyo, omegao, xg, yg, thetag, vxg, vyg, omegag = states # 12 DoF

        # Object kinematics
        self.pos_object = [xo, yo, 0.0]
        self.quat_object = p.getQuaternionFromEuler([0.0, 0.0, thetao])
        self.vel_object = [vxo, vyo, 0.0]
        self.vel_ang_object = [0.0, 0.0, omegao]

        # Gripper kinematics
        self.pos_gripper = [xg, yg, 0.0]
        if self.gripper_name == 'bowl':
            self.quat_gripper = p.getQuaternionFromEuler([-math.pi/2, 0, thetag])
        else:
            self.quat_gripper = p.getQuaternionFromEuler([0.0, 0.0, thetag])
        self.vel_gripper = [vxg, vyg, 0.0]
        self.vel_ang_gripper = [0.0, 0.0, omegag]

        # Reset the kinematics
        p.resetBasePositionAndOrientation(self.objectUid, self.pos_object, self.quat_object)
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper, self.quat_gripper)
        p.resetBaseVelocity(self.objectUid, self.vel_object, self.vel_ang_object)
        p.resetBaseVelocity(self.gripperUid, self.vel_gripper, self.vel_ang_gripper) # linear and angular vels both in world coordinates

    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t, ax, ay, omega = inputs
        interval = int(int(t*240)/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        force_on_object = [self.mass_object*ax, self.mass_object*ay, 0.0]
        torque_on_object = [0.0, 0.0, self.moment_object*omega]
        for i in range(int(t*240)):
            # Apply external force on object
            self.pos_object,_ = p.getBasePositionAndOrientation(self.objectUid)
            p.applyExternalForce(self.objectUid, -1, 
                                force_on_object, # gravity compensated 
                                self.pos_object, 
                                p.WORLD_FRAME)
            p.applyExternalTorque(self.objectUid, -1, 
                                torque_on_object,
                                p.WORLD_FRAME)
            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if print_via_points and (i % interval == 0 or i == int(t*240)-1):
                via_points.append([self.pos_object[0], self.pos_object[1]])

            if self.gui:
                time.sleep(10/240)

        # Get the object and gripper states
        self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
        self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
        self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
        self.pos_gripper, self.quat_gripper = p.getBasePositionAndOrientation(self.gripperUid)
        self.eul_gripper = p.getEulerFromQuaternion(self.quat_gripper)
        self.vel_gripper,self.vel_ang_gripper = p.getBaseVelocity(self.gripperUid)

        new_states = [self.pos_object[0], self.pos_object[1], self.eul_object[2],
                      self.vel_object[0], self.vel_object[1], self.vel_ang_object[2],
                      self.pos_gripper[0], self.pos_gripper[1], self.eul_gripper[2], 
                      self.vel_gripper[0], self.vel_gripper[1], self.vel_ang_gripper[2]
                      ]
        return new_states, via_points

    def finish_sim(self):
        # Clean up and close the simulation
        p.disconnect()


class forwardSimulationPlanePushReal(forwardSimulationPlanePush):
    def set_params(self, params):
        self.angle_slope = params[5] # equivalent to on a slope
        p.setGravity(0, 
                     self.g*math.sin(self.angle_slope),
                     self.g*math.cos(self.angle_slope),)

        # Kinodynamics
        self.mass_object = params[0]
        self.moment_object = params[1] # moment of inertia
        self.pos_object = [1000,0,0]
        self.quat_object = p.getQuaternionFromEuler([math.pi/2,0,0])
        self.vel_object = [.1,0,0]
        
        self.mass_gripper = params[2]
        self.moment_gripper = params[3]
        self.y_obstacle = params[4]
        self.object_name = params[6] # 'box', 'cylinder'
        self.gripper_name = params[7] # 'box', 'cylinder', 'bowl'
        self.lateral_friction_coef = params[8]
        self.scale_factor = params[9]

        self.pos_gripper = [1000,0,2]
        self.quat_gripper = p.getQuaternionFromEuler([0,0,0])
        self.vel_gripper = [0,0,0]
        self.vel_ang_gripper = [0,0,0]

        self.z_bodies = 0.03/2 * self.scale_factor
        self.half_extent_obstacle = [7, .5, self.z_bodies]
        self.pos_obstacle = [5,self.y_obstacle-self.half_extent_obstacle[1],0]
        self.quat_obstacle = p.getQuaternionFromEuler([0,0,0])

    def create_shapes(self):
        # Create a plane
        #  The default frictional coefficients used by PyBullet are 0.5 for the lateral friction and 0 for both the rolling and spinning.
        self.planeUid = p.loadURDF("plane.urdf", basePosition=[0,0,-self.z_bodies])
        p.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        
        # Create an object
        if self.object_name == 'rectangle':
            self.objectUid = p.loadURDF("asset/real-world/rectangle-object/rectangle-object.urdf", self.pos_object, self.quat_object, globalScaling=1)
        elif self.object_name == 'triangle':
            self.objectUid = p.loadURDF("asset/real-world/triangle-object/triangle-object.urdf", self.pos_object, self.quat_object, globalScaling=1)
        elif self.object_name == 'convex':
            self.objectUid = p.loadURDF("asset/real-world/convex-object/convex-object.urdf", self.pos_object, self.quat_object, globalScaling=1)
        elif self.object_name == 'concave':
            self.objectUid = p.loadURDF("asset/real-world/concave-object/concave-object.urdf", self.pos_object, self.quat_object, globalScaling=1)
        elif self.object_name == 'irregular':
            self.objectUid = p.loadURDF("asset/real-world/irregular-object/irregular-object.urdf", self.pos_object, self.quat_object, globalScaling=1)
        p.changeDynamics(self.objectUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        
        # Create a robot
        if self.gripper_name == 'circle':
            self.gripperUid = p.loadURDF("asset/real-world/circle-gripper/circle-gripper.urdf", self.pos_gripper, self.quat_gripper, globalScaling=1)
        elif self.gripper_name == 'jaw':
            self.gripperUid = p.loadURDF("asset/real-world/jaw-gripper/jaw-gripper.urdf", self.pos_gripper, self.quat_gripper, globalScaling=1)
        p.changeDynamics(self.gripperUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        
        # Create a static obstacle
        obstacleId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extent_obstacle)
        self.obstacleUid = p.createMultiBody(0, 
                                       obstacleId, 
                                       self.visualShapeId, 
                                       self.pos_obstacle,
                                       self.quat_obstacle)
        p.changeDynamics(self.obstacleUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        
    def reset_states(self, states):
        xo, yo, thetao, vxo, vyo, omegao, xg, yg, thetag, vxg, vyg, omegag = states # 12 DoF

        # Object kinematics
        self.pos_object = [xo, yo, 0.0]
        self.quat_object = p.getQuaternionFromEuler([0.0, 0.0, thetao])
        self.vel_object = [vxo, vyo, 0.0]
        self.vel_ang_object = [0.0, 0.0, omegao]

        # Gripper kinematics
        self.pos_gripper = [xg, yg, 0.0]
        self.quat_gripper = p.getQuaternionFromEuler([0.0, 0.0, thetag])
        self.vel_gripper = [vxg, vyg, 0.0]
        self.vel_ang_gripper = [0.0, 0.0, omegag]

        # Reset the kinematics
        p.resetBasePositionAndOrientation(self.objectUid, self.pos_object, self.quat_object)
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper, self.quat_gripper)
        p.resetBaseVelocity(self.objectUid, self.vel_object, self.vel_ang_object)
        p.resetBaseVelocity(self.gripperUid, self.vel_gripper, self.vel_ang_gripper) # linear and angular vels both in world coordinates


class forwardSimulationPlanePushMulti(forwardSimulationPlanePush):
    def set_params(self, params):
        self.angle_slope = params[5] # equivalent to on a slope
        p.setGravity(0, 
                     self.g*math.sin(self.angle_slope),
                     self.g*math.cos(self.angle_slope),)

        # Kinodynamics
        self.mass_object = params[0]
        self.moment_object = params[1] # moment of inertia
        self.pos_object = [1000,0,0]
        self.quat_object = p.getQuaternionFromEuler([math.pi/2,0,0])
        self.vel_object = [.1,0,0]
        
        self.mass_gripper = params[2]
        self.moment_gripper = params[3]
        self.y_obstacle = params[4]
        self.object_name = params[6] # 'box', 'cylinder'
        self.gripper_name = params[7] # 'box', 'cylinder', 'bowl'
        self.lateral_friction_coef = params[8]
        self.num_objects = params[9]

        self.pos_gripper = [1000,0,2]
        self.quat_gripper = p.getQuaternionFromEuler([0,0,0])
        self.vel_gripper = [0,0,0]
        self.vel_ang_gripper = [0,0,0]

        # self.z_bodies = 0.03/2 * self.scale_factor
        self.z_bodies = 0.03/2
        self.half_extent_obstacle = [7, .5, self.z_bodies]
        self.pos_obstacle = [5,self.y_obstacle+self.half_extent_obstacle[1],0]
        self.quat_obstacle = p.getQuaternionFromEuler([0,0,0])

    def create_shapes(self):
        # Create a plane
        planeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, self.z_bodies])
        self.planeUid = p.createMultiBody(0, 
                                            planeId, 
                                            self.visualShapeId, 
                                            [0,0,-2*self.z_bodies],
                                            p.getQuaternionFromEuler([0,0,0]))
        p.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        # p.changeVisualShape(self.planeUid, -1, rgbaColor=[193/255, 193/255, 193/255, 1])

        # Create objects
        objectId = []
        self.objectUid = []
        for i in range(self.num_objects):
            if self.object_name == 'box':
                # objectId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.6, .2, self.z_bodies])
                objectId.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[.6, .2, self.z_bodies]))
                self.objectUid.append(p.createMultiBody(self.mass_object, 
                                                        objectId[i], 
                                                        self.visualShapeId, 
                                                        self.pos_object,
                                                        self.quat_object))
            elif self.object_name == 'cylinder':
                # objectId = p.createCollisionShape(p.GEOM_CYLINDER, radius=.2, height=2*self.z_bodies)
                objectId.append(p.createCollisionShape(p.GEOM_CYLINDER, radius=.05, height=2*self.z_bodies))
                self.objectUid.append(p.createMultiBody(self.mass_object, 
                                                        objectId[i], 
                                                        self.visualShapeId, 
                                                        self.pos_object,
                                                        self.quat_object))
            p.changeDynamics(self.objectUid[i], -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
            p.changeVisualShape(self.objectUid[i], -1, rgbaColor=[247/255, 143/255, 0/255, 1])
        
        # Create a robot
        if self.gripper_name == 'box':
            gripperId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.4, .1, self.z_bodies])
            self.gripperUid = p.createMultiBody(self.mass_gripper, 
                                           gripperId, 
                                           self.visualShapeId, 
                                           self.pos_gripper,
                                           self.quat_gripper)
        elif self.gripper_name == 'cylinder':
            gripperId = p.createCollisionShape(p.GEOM_CYLINDER, radius=.1, height=2*self.z_bodies)
            self.gripperUid = p.createMultiBody(self.mass_gripper,
                                                gripperId,
                                                self.visualShapeId,
                                                self.pos_gripper,
                                                self.quat_gripper)
        elif self.gripper_name == 'bowl':
            self.gripperUid = p.loadURDF("asset/bowl/2d-bowl.urdf", self.pos_gripper, self.quat_gripper, globalScaling=1)
        p.changeDynamics(self.gripperUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.gripperUid, -1, rgbaColor=[112/255, 190/255, 83/255, 1])
        
        # Create a static obstacle
        obstacleId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extent_obstacle)
        self.obstacleUid = p.createMultiBody(0, 
                                            obstacleId, 
                                            self.visualShapeId, 
                                            self.pos_obstacle,
                                            self.quat_obstacle)
        p.changeDynamics(self.obstacleUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.obstacleUid, -1, rgbaColor=[.3,.3,.3,1])
        
    def reset_states(self, states):
        self.pos_object = [[states[6*i], states[6*i+1], 0.0] for i in range(self.num_objects)]
        self.eul_object = [[0.0, 0.0, states[6*i+2]] for i in range(self.num_objects)]
        self.quat_object = [p.getQuaternionFromEuler([0.0, 0.0, states[6*i+2]]) for i in range(self.num_objects)]
        self.vel_object = [[states[6*i+3], states[6*i+4], 0.0] for i in range(self.num_objects)]
        self.vel_ang_object = [[0.0, 0.0, states[6*i+5]] for i in range(self.num_objects)]

        # Gripper kinematics
        self.pos_gripper = [states[-6], states[-5], 0.0]
        self.quat_gripper = p.getQuaternionFromEuler([0.0, 0.0, states[-4]])
        self.vel_gripper = [states[-3], states[-2], 0.0]
        self.vel_ang_gripper = [0.0, 0.0, states[-1]]

        # Reset the kinematics
        for i in range(self.num_objects):
            p.resetBasePositionAndOrientation(self.objectUid[i], self.pos_object[i], self.quat_object[i])
            p.resetBaseVelocity(self.objectUid[i], self.vel_object[i], self.vel_ang_object[i])
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper, self.quat_gripper)
        p.resetBaseVelocity(self.gripperUid, self.vel_gripper, self.vel_ang_gripper) # linear and angular vels both in world coordinates

    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t = inputs[0]
        interval = int(int(t*240)/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        force_on_object = [[self.mass_object*inputs[3*i+1], self.mass_object*inputs[3*i+2], 0.0] for i in range(self.num_objects)]
        torque_on_object = [[0.0, 0.0, self.moment_object*inputs[3*i+3],] for i in range(self.num_objects)]
        for i in range(int(t*240)):
            # Apply external force on object
            for i in range(self.num_objects):
                self.pos_object[i],_ = p.getBasePositionAndOrientation(self.objectUid[i])
                p.applyExternalForce(self.objectUid[i], -1, 
                                    force_on_object[i],
                                    self.pos_object[i], 
                                    p.WORLD_FRAME)
                p.applyExternalTorque(self.objectUid[i], -1,
                                    torque_on_object[i],
                                    p.WORLD_FRAME)
            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if print_via_points and (i % interval == 0 or i == int(t*240)-1):
                via_points.append([self.pos_object[0][0], self.pos_object[0][1]])

            if self.gui:
                time.sleep(10/240)

        # Get the object and gripper states
        self.pos_gripper, self.quat_gripper = p.getBasePositionAndOrientation(self.gripperUid)
        self.eul_gripper = p.getEulerFromQuaternion(self.quat_gripper)
        self.vel_gripper,self.vel_ang_gripper = p.getBaseVelocity(self.gripperUid)
        for i in range(self.num_objects):
            self.pos_object[i], self.quat_object[i] = p.getBasePositionAndOrientation(self.objectUid[i])
            self.eul_object[i] = p.getEulerFromQuaternion(self.quat_object[i]) # rad
            self.vel_object[i], self.vel_ang_object[i] = p.getBaseVelocity(self.objectUid[i])

        new_states = []
        for i in range(self.num_objects):
            new_states = new_states + [self.pos_object[i][0], self.pos_object[i][1], self.eul_object[i][2],
                                    self.vel_object[i][0], self.vel_object[i][1], self.vel_ang_object[i][2],]
        new_states = new_states + [self.pos_gripper[0], self.pos_gripper[1], self.eul_gripper[2], 
                                   self.vel_gripper[0], self.vel_gripper[1], self.vel_ang_gripper[2],]
        
        return new_states, via_points


class forwardSimulationPlanePushPlanner(forwardSimulationPlanePush):
    def __init__(self, gui=False):
        super().__init__(gui=gui)

    def check_collision(self, thres=-5e-3):
        # Check for collisions  
        p.stepSimulation()
        contacts0 = p.getClosestPoints(self.objectUid, self.gripperUid, distance=thres)
        contacts1 = p.getClosestPoints(self.objectUid, self.obstacleUid, distance=thres)
        if len(contacts0)+len(contacts1) > 0:
            # print("Bodies are in collision")
            return True
        else:
            # print("Bodies are not in collision")
            return False
        
    def run_forward_sim(self, inputs, print_via_points=False):
        t, ax, ay, omega = inputs

        # Step the simulation
        force_on_gripper = [self.mass_gripper*ax, self.mass_gripper*ay, 0.0]
        torque_on_gripper = [0.0, 0.0, self.moment_gripper*omega]
        for _ in range(int(t*240)):
            # Apply external force on object
            self.pos_gripper,_ = p.getBasePositionAndOrientation(self.gripperUid)
            p.applyExternalForce(self.gripperUid, -1, 
                                force_on_gripper, # gravity compensated 
                                self.pos_gripper, 
                                p.WORLD_FRAME)
            p.applyExternalTorque(self.gripperUid, -1, 
                                torque_on_gripper,
                                p.WORLD_FRAME)
            p.stepSimulation()

            if self.gui:
                time.sleep(10/240)

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
        
        self.heuristics = [dist, s_stick, s_engage,]

        new_states = [self.pos_object[0], self.pos_object[1], self.eul_object[2],
                      self.vel_object[0], self.vel_object[1], self.vel_ang_object[2],
                      self.pos_gripper[0], self.pos_gripper[1], self.eul_gripper[2], 
                      self.vel_gripper[0], self.vel_gripper[1], self.vel_ang_gripper[2]
                      ]

        # return new_states, via_points
        return new_states, None
    

class forwardSimulationPlanePushRrtstar(forwardSimulationPlanePush):
    def __init__(self, gui=False):
        super().__init__(gui=gui)
    
    def set_gripper_pos(self, gripper_pose):
        xg, yg, thetag = gripper_pose
        self.pos_gripper = [xg, yg, 0.0]
        if self.gripper_name == 'bowl':
            self.quat_gripper = p.getQuaternionFromEuler([-math.pi/2, 0, thetag])
        else:
            self.quat_gripper = p.getQuaternionFromEuler([0.0, 0.0, thetag])

        # Reset the kinematics
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper, self.quat_gripper)
        
    def reset_states(self, states):
        xo, yo, thetao = states # 3 DoF
        self.pos_object = [xo, yo, 0.0]
        self.quat_object = p.getQuaternionFromEuler([0.0, 0.0, thetao])
        p.resetBasePositionAndOrientation(self.objectUid, self.pos_object, self.quat_object)
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper, self.quat_gripper)

    def check_collision(self, thres=-5e-3):
        # Check for collisions  
        p.stepSimulation()
        contacts0 = p.getClosestPoints(self.objectUid, self.gripperUid, distance=thres)
        contacts1 = p.getClosestPoints(self.objectUid, self.obstacleUid, distance=thres)
        if len(contacts0)+len(contacts1) > 0:
            # print("Bodies are in collision")
            return False
        else:
            # print("Bodies are not in collision")
            return True


class forwardSimulationBalanceGrasp(forwardSimulationPlanePush):
    def __init__(self, gui=False):
        super().__init__(gui=gui)

    def create_shapes(self):
        # Create a plane
        # self.planeId = p.loadURDF("plane.urdf", basePosition=[0,0,-self.z_bodies])
        planeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, self.z_bodies])
        self.planeUid = p.createMultiBody(0, 
                                            planeId, 
                                            self.visualShapeId, 
                                            [0,0,-2*self.z_bodies],
                                            p.getQuaternionFromEuler([0,0,0]))
        p.changeDynamics(self.planeUid, -1, lateralFriction=0.5, spinningFriction=0, # no friction
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.planeUid, -1, rgbaColor=[1,1,1,1])
        
        # Create an object
        if self.object_name == 'box':
            objectId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.6, .2, self.z_bodies])
            self.objectUid = p.createMultiBody(self.mass_object, 
                                               objectId, 
                                               self.visualShapeId, 
                                               self.pos_object,
                                               self.quat_object)
        if self.object_name == 'square':
            objectId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.2, .2, self.z_bodies])
            self.objectUid = p.createMultiBody(self.mass_object, 
                                               objectId, 
                                               self.visualShapeId, 
                                               self.pos_object,
                                               self.quat_object)
        elif self.object_name == 'cylinder':
            objectId = p.createCollisionShape(p.GEOM_CYLINDER, radius=.2, height=2*self.z_bodies)
            self.objectUid = p.createMultiBody(self.mass_object, 
                                            objectId, 
                                            self.visualShapeId, 
                                            self.pos_object,
                                            self.quat_object)
        p.changeDynamics(self.objectUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.objectUid, -1, rgbaColor=[247/255, 143/255, 0/255, 1])

        # Create a robot
        if self.gripper_name == 'box':
            gripperId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.6, .1, self.z_bodies])
            self.gripperUid = p.createMultiBody(self.mass_gripper, 
                                           gripperId, 
                                           self.visualShapeId, 
                                           self.pos_gripper,
                                           self.quat_gripper)
        elif self.gripper_name == 'cylinder':
            gripperId = p.createCollisionShape(p.GEOM_CYLINDER, radius=.1, height=2*self.z_bodies)
            self.gripperUid = p.createMultiBody(self.mass_gripper,
                                                gripperId,
                                                self.visualShapeId,
                                                self.pos_gripper,
                                                self.quat_gripper)
        elif self.gripper_name == 'bowl':
            self.gripperUid = p.loadURDF("asset/bowl/2d-bowl.urdf", self.pos_gripper, self.quat_gripper, globalScaling=1)
        p.changeDynamics(self.gripperUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.gripperUid, -1, rgbaColor=[112/255, 190/255, 83/255, 1])

        # Create a static obstacle to catch the falling object
        self.half_extent_obstacle = [20, .05, self.z_bodies]
        obstacleId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extent_obstacle)
        self.obstacleUid = p.createMultiBody(0, 
                                       obstacleId, 
                                       self.visualShapeId, 
                                       self.pos_obstacle,
                                       self.quat_obstacle)
        p.changeDynamics(self.obstacleUid, -1, lateralFriction=1, spinningFriction=1, 
                         rollingFriction=1, linearDamping=1, angularDamping=1)
        
    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t, ax, ay, omega = inputs
        interval = int(int(t*240)/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        force_on_object = [self.mass_object*ax, self.mass_object*ay, 0.0]
        torque_on_object = [0.0, 0.0, self.moment_object*omega]
        for i in range(int(t*240)):
            # Apply external force on object
            self.pos_object, _ = p.getBasePositionAndOrientation(self.objectUid)
            self.pos_gripper, _ = p.getBasePositionAndOrientation(self.gripperUid)
            p.applyExternalForce(self.objectUid, -1, 
                                force_on_object,
                                self.pos_object, 
                                p.WORLD_FRAME)
            p.applyExternalForce(self.gripperUid, -1, 
                                [0, -self.g*(self.mass_gripper+self.mass_object)*math.sin(self.angle_slope), 0], # gravity compensation
                                self.pos_gripper, 
                                p.WORLD_FRAME)
            p.applyExternalTorque(self.objectUid, -1, 
                                torque_on_object,
                                p.WORLD_FRAME)
            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if print_via_points and (i % interval == 0 or i == int(t*240)-1):
                via_points.append([self.pos_object[0], self.pos_object[1]])

            if self.gui:
                time.sleep(1/240)

        # Get the object and gripper states
        self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
        self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
        self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
        self.pos_gripper, self.quat_gripper = p.getBasePositionAndOrientation(self.gripperUid)
        self.eul_gripper = p.getEulerFromQuaternion(self.quat_gripper)
        self.vel_gripper,self.vel_ang_gripper = p.getBaseVelocity(self.gripperUid)

        new_states = [self.pos_object[0], self.pos_object[1], self.eul_object[2],
                      self.vel_object[0], self.vel_object[1], self.vel_ang_object[2],
                      self.pos_gripper[0], self.pos_gripper[1], self.eul_gripper[2], 
                      self.vel_gripper[0], self.vel_gripper[1], self.vel_ang_gripper[2]
                      ]

        return new_states, via_points
    

class forwardSimulationBoxPivot():
    def __init__(self, gui=False):
        self.visualShapeId = -1
        self.gui = gui
        if self.gui:
            p.connect(p.GUI) # p.GUI
        else:
            p.connect(p.DIRECT) # p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
        self.g = -9.81
        self.for_paper_vis = 1
        # p.setGravity(0, 0, self.g)

        # self.set_params(params)
        # self.create_shapes()
    
    def set_params(self, params):
        # Kinodynamics
        self.mass_object = params[0]
        self.moment_object = params[1] # moment of inertia
        self.pos_object = [1000,0,0]
        self.quat_object = p.getQuaternionFromEuler([0,0,0])
        self.vel_object = [0,0,0]
        
        self.mass_gripper = params[2]
        self.moment_gripper = params[3]
        self.lateral_friction_coef = params[4]
        self.rest_length = params[5]
        self.k = params[6]  # Spring constant
        self.height_spring = params[7]

        self.pos_gripper1 = [1000,0,self.height_spring]
        self.pos_gripper2 = [1000,0,self.height_spring]
        self.quat_gripper1 = p.getQuaternionFromEuler([0,0,0])
        self.quat_gripper2 = p.getQuaternionFromEuler([0,0,0])
        self.vel_gripper1 = [0,0,0]
        self.vel_gripper2 = [0,0,0]

    def create_shapes(self):
        # self.planeUid = p.loadURDF("plane.urdf")
        planeUid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[100,100,0.1])
        self.planeUid = p.createMultiBody(0, 
                                           planeUid, 
                                           -1, 
                                           [0,0,-0.1])
        p.changeVisualShape(self.planeUid, -1, rgbaColor=[.3,.3,.3,1])
        
        # Add spring base object
        if self.for_paper_vis:
            gripperUid1 = p.createCollisionShape(p.GEOM_SPHERE, radius=.1)
        else:
            gripperUid1 = p.createCollisionShape(p.GEOM_SPHERE, radius=.3)
        self.gripperUid1 = p.createMultiBody(self.mass_gripper, 
                                    gripperUid1, 
                                    -1, 
                                    [-3, 0, self.height_spring])
        
        # Add spring tip object
        if self.for_paper_vis:
            gripperUid2 = p.createCollisionShape(p.GEOM_SPHERE, radius=.1)
        else:
            gripperUid2 = p.createCollisionShape(p.GEOM_SPHERE, radius=.3)
        self.gripperUid2 = p.createMultiBody(self.mass_gripper, 
                                    gripperUid2, 
                                    -1, 
                                    [0, 0, self.height_spring])
        p.changeVisualShape(self.gripperUid1, -1, rgbaColor=[112/255, 190/255, 83/255, 1])
        p.changeVisualShape(self.gripperUid2, -1, rgbaColor=[112/255, 190/255, 83/255, 1])

        # Add box
        if self.for_paper_vis:
            objectUid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5,]*3)
            self.objectUid = p.createMultiBody(self.mass_object, 
                                        objectUid, 
                                        -1, 
                                        [0,0,0.5])
        else:
            objectUid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2,2,2])
            self.objectUid = p.createMultiBody(self.mass_object, 
                                        objectUid, 
                                        -1, 
                                        [6,0,2])
            
        p.changeDynamics(self.gripperUid1, -1, lateralFriction=0, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.gripperUid2, -1, lateralFriction=0.2, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.objectUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        p.changeVisualShape(self.objectUid, -1, rgbaColor=[247/255, 143/255, 0/255, 1])

        # Create a fixed joint between the cubes, acting like a spring
        self.c_spring = p.createConstraint(self.gripperUid1, -1, self.gripperUid2, -1, p.JOINT_FIXED, [1, 0, 0], [self.rest_length/2, 0, 0], [-self.rest_length/2, 0, 0])

        # Create point-to-point constraints to keep boxes along a line
        pivotInBox = [0, 0, 0] 
        pivotInWorld = [0, 0, self.height_spring]
        constraint1 = p.createConstraint(self.gripperUid1, -1, -1, -1, p.JOINT_PRISMATIC, [1, 0, 0], pivotInBox, pivotInWorld)
        constraint2 = p.createConstraint(self.gripperUid2, -1, -1, -1, p.JOINT_PRISMATIC, [1, 0, 0], pivotInBox, pivotInWorld)

    def reset_states(self, states):
        xo, zo, thetao, vxo, vzo, omegao, xg1, xg2, vg1, vg2 = states # 12 DoF

        # Object kinematics
        self.pos_object = [xo, 0.0, zo]
        self.quat_object = p.getQuaternionFromEuler([0.0, thetao, 0.0])
        self.vel_object = [vxo, 0.0, vzo]
        self.vel_ang_object = [0.0, omegao, 0.0]

        # Gripper kinematics
        self.pos_gripper1 = [xg1, 0.0, self.height_spring]
        self.pos_gripper2 = [xg2, 0.0, self.height_spring]
        self.quat_gripper1 = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.quat_gripper2 = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.vel_gripper1 = [vg1, 0.0, 0.0]
        self.vel_gripper2 = [vg2, 0.0, 0.0]
        self.vel_ang_gripper1 = [0.0, 0.0, 0.0]
        self.vel_ang_gripper2 = [0.0, 0.0, 0.0]

        # Reset the kinematics
        p.resetBasePositionAndOrientation(self.objectUid, self.pos_object, self.quat_object)
        p.resetBaseVelocity(self.objectUid, self.vel_object, self.vel_ang_object)
        p.resetBasePositionAndOrientation(self.gripperUid1, self.pos_gripper1, self.quat_gripper1)
        p.resetBasePositionAndOrientation(self.gripperUid2, self.pos_gripper2, self.quat_gripper2)
        p.resetBaseVelocity(self.gripperUid1, self.vel_gripper1, self.vel_ang_gripper1) # linear and angular vels both in world coordinates
        p.resetBaseVelocity(self.gripperUid2, self.vel_gripper2, self.vel_ang_gripper2)

    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t, ax, omega = inputs
        interval = int(int(t*240)/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        force_on_object = [self.mass_object*ax, 0.0, 0.0]
        torque_on_object = [0.0, self.moment_object*omega, 0.0]
        for i in range(int(t*240)):
            # Apply external force on object
            self.pos_object, _ = p.getBasePositionAndOrientation(self.objectUid)
            p.applyExternalForce(self.objectUid, -1, 
                                force_on_object, # gravity compensated 
                                self.pos_object, 
                                p.WORLD_FRAME)
            p.applyExternalTorque(self.objectUid, -1, 
                                torque_on_object,
                                p.WORLD_FRAME)
            
            # Get positions of the boxes
            self.pos_gripper1, _ = p.getBasePositionAndOrientation(self.gripperUid1)
            self.pos_gripper2, _ = p.getBasePositionAndOrientation(self.gripperUid2)
            
            # Update the maxForce for the spring constraint
            maxForce = self.k * np.abs(np.linalg.norm(np.array(self.pos_gripper2) - np.array(self.pos_gripper1)) - self.rest_length)
            p.changeConstraint(self.c_spring, maxForce=maxForce)

            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if print_via_points and (i % interval == 0 or i == int(t*240)-1):
                via_points.append([self.pos_object[0], self.pos_object[2]])

            if self.gui:
                time.sleep(1/240)

        # Get the object and gripper states
        self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
        self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
        self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
        self.pos_gripper1, _ = p.getBasePositionAndOrientation(self.gripperUid1)
        self.pos_gripper2, _ = p.getBasePositionAndOrientation(self.gripperUid2)
        self.vel_gripper1,_ = p.getBaseVelocity(self.gripperUid1)
        self.vel_gripper2,_ = p.getBaseVelocity(self.gripperUid2)

        new_states = [self.pos_object[0], self.pos_object[2], self.eul_object[1],
                      self.vel_object[0], self.vel_object[2], self.vel_ang_object[1],
                      self.pos_gripper1[0], self.pos_gripper2[0],  
                      self.vel_gripper1[0], self.vel_gripper2[0], 
                      ]
        return new_states, via_points

    def finish_sim(self):
        # Clean up and close the simulation
        p.disconnect()


class forwardSimulationWaterSwing():
    def __init__(self, gui=False):
        self.visualShapeId = -1
        self.gui = gui
        if self.gui:
            p.connect(p.GUI) # p.GUI
        else:
            p.connect(p.DIRECT) # p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
        self.g = -9.81
        # p.setGravity(0, 0, self.g)

        # self.set_params(params)
        # self.create_shapes()
    
    def set_params(self, params):
        # Kinodynamics
        self.mass_object = params[0]
        self.moment_object = params[1] # moment of inertia
        self.pos_object = [1000,0,0]
        self.quat_object = p.getQuaternionFromEuler([0,0,0])
        self.vel_object = [.1,0,0]
        
        self.mass_gripper = params[2]
        self.moment_gripper = params[3]
        self.pos_gripper = [1000,0,2]
        self.quat_gripper = p.getQuaternionFromEuler([0,0,0])
        self.vel_gripper = [0,0,0]
        self.vel_ang_gripper = [0,0,0]

        self.lateral_friction_coef = 0.8

        # Geometrics
        self.y_bodies = .3

    def create_shapes(self):
        # Create a plane
        # self.planeId = p.loadURDF("plane.urdf", basePosition=[0,0,-self.z_bodies])
        # p.changeDynamics(self.planeId, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
        #                  rollingFriction=0, linearDamping=0, angularDamping=0)
        
        # Create an object # TODO import complex shapes from files
        objectId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.3, self.y_bodies, .3])
        self.objectUid = p.createMultiBody(self.mass_object, 
                                           objectId, 
                                           self.visualShapeId, 
                                           self.pos_object,
                                           self.quat_object)
        p.changeDynamics(self.objectUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)
        
        # Create a robot
        # gripperId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.5, self.y_bodies, .5])
        # self.gripperUid = p.createMultiBody(self.mass_gripper, 
        #                                gripperId, 
        #                                self.visualShapeId, 
        #                                self.pos_gripper,
        #                                self.quat_gripper)
        self.gripperUid = p.loadURDF("asset/4face-bottle.urdf", self.pos_gripper, self.quat_gripper)
        p.changeDynamics(self.gripperUid, -1, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                         rollingFriction=0, linearDamping=0, angularDamping=0)

    def reset_states(self, states):
        xo, zo, thetao, vxo, vzo, omegao, xg, zg, thetag, vxg, vzg, omegag = states # 12 DoF

        # Object kinematics
        self.pos_object = [xo, 0.0, zo]
        self.quat_object = p.getQuaternionFromEuler([0.0, thetao, 0.0])
        self.vel_object = [vxo, 0.0, vzo]
        self.vel_ang_object = [0.0, omegao, 0.0]

        # Gripper kinematics
        self.pos_gripper = [xg, 0.0, zg]
        self.quat_gripper = p.getQuaternionFromEuler([0.0, thetag, 0.0])
        self.vel_gripper = [vxg, 0.0, vzg]
        self.vel_ang_gripper = [0.0, omegag, 0.0]

        # Reset the kinematics
        p.resetBasePositionAndOrientation(self.objectUid, self.pos_object, self.quat_object)
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper, self.quat_gripper)
        p.resetBaseVelocity(self.objectUid, self.vel_object, self.vel_ang_object)
        p.resetBaseVelocity(self.gripperUid, self.vel_gripper, self.vel_ang_gripper) # linear and angular vels both in world coordinates

    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t, ax, az, omega = inputs
        interval = int(int(t*240)/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        force_on_object = [self.mass_object*ax, 0.0, self.mass_object*az]
        torque_on_object = [0.0, self.moment_object*omega, 0.0]
        for i in range(int(t*240)):
            # Apply external force on object
            p.applyExternalForce(self.objectUid, -1, 
                                force_on_object, # gravity compensated 
                                self.pos_object, 
                                p.WORLD_FRAME)
            p.applyExternalTorque(self.objectUid, -1, 
                                torque_on_object,
                                p.WORLD_FRAME)
            p.stepSimulation()
            self.pos_object, _ = p.getBasePositionAndOrientation(self.objectUid)

            # Print object via-points along the trajectory for visualization
            if print_via_points and (i % interval == 0 or i == int(t*240)-1):
                via_points.append([self.pos_object[0], self.pos_object[2]])

            if self.gui:
                time.sleep(10/240)

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
        return new_states, via_points

    def finish_sim(self):
        # Clean up and close the simulation
        p.disconnect()


class forwardSimulationHerding():
    def __init__(self, gui=False):
        pass

    def set_params(self, params):
        # Kinodynamics
        # self.num_robots = 5
        self.dim = 2
        self.mass_object = params[0]
        self.mass_gripper = params[1]
        self.num_robots = params[2]

    def create_shapes(self):
        pass

    def get_repulsive_force(self, pos_object, pos_gripper, k=5.0):
        # Get the repulsive force
        pos_object_g = np.array(pos_object) - np.array(pos_gripper)
        dist = np.linalg.norm(pos_object_g)
        force_norm_direction = pos_object_g / dist
        if dist <= 0.2:
            force = 5.0 * k * self.mass_gripper * force_norm_direction
        else:
            force = 1.0 * k / dist * self.mass_gripper * force_norm_direction
        return force
    
    def reset_states(self, states):
        xo, yo, vxo, vyo = states[:4] # 4+10+10 DoF

        # Object kinematics
        self.pos_object = np.asarray([xo, yo])
        self.vel_object = np.asarray([vxo, vyo])

        # # Gripper kinematics
        self.pos_gripper = np.asarray(states[4:4+self.dim*self.num_robots])
        self.vel_gripper = np.asarray(states[4+self.dim*self.num_robots:4+2*self.dim*self.num_robots])

    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t, ax, ay = inputs
        dt = t / float(num_via_points)

        # Step the simulation
        via_points = []
        for i in range(num_via_points):
            # Calculate the repulsive force
            force_on_object = np.zeros(self.dim)
            for j in range(self.num_robots):
                force_on_object += self.get_repulsive_force(self.pos_object, self.pos_gripper[2*j:2*j+2])

            # Apply the force on the object
            acc_object = force_on_object / self.mass_object
            acc_object += np.asarray([ax, ay])
            # acc_object = np.asarray([ax, ay])
            self.pos_object += self.vel_object * dt + 0.5 * acc_object * dt * dt
            self.vel_object += acc_object * dt
                        
            # Update robots position
            self.pos_gripper += self.vel_gripper * dt

            # Print object via-points along the trajectory for visualization
            new_states = np.concatenate([self.pos_object, self.vel_object, self.pos_gripper]).tolist()
            if print_via_points:
                via_points.append(new_states[:2])

        return new_states, via_points

    def finish_sim(self):
        pass


class forwardSimulationGripper():
    def __init__(self, gui=False):
        self.visualShapeId = -1
        self.gui = gui
        if self.gui:
            p.connect(p.GUI) # p.GUI
        else:
            p.connect(p.DIRECT) # p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
        self.g = -9.81
        p.setGravity(0, 0, self.g)
    
    def set_params(self, params):
        # Kinodynamics
        self.mass_object, self.moment_object, self.length_object, self.movable_joints, self.start_env_pos, self.lateral_friction_coef = params
        self.mass_table = 1e9
        self.mass_gripper_base = 10
        self.num_movable_joints = len(self.movable_joints)
        self.num_joints = 12
        self.num_links = 13
        self.non_movable_joints = [i for i in range(self.num_joints) if i not in self.movable_joints]
        self.dim_workspace = 3
        self.fingertip_link_ids = [3, 7, 11]
        self.stiffness = [1e-2,] * self.num_movable_joints  # P gain for each joint
        self.damping = [1e-1,] * self.num_movable_joints  # D gain for each joint

        self.pos_object = [-0.9,0,.6]
        self.quat_object = p.getQuaternionFromEuler([0,0,0])
        self.vel_object = [0,0,0]

        self.pos_table = [0,0,self.start_env_pos[-1]]
        self.vel_table = [0,0,0]
        self.pivot_in_table = [0, 0, 0] # for the prismatic constraint
        self.pivot_in_world = [0, 0, 0]

        self.start_gripper_pos = self.start_env_pos[:-1]
        self.pos_gripper_base = [-1.1,0,1.9]
        self.quat_gripper_base = p.getQuaternionFromEuler([-math.pi/2,-0.3,0])

    def create_shapes(self):
        # Add a table
        boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1,1,0.3])
        mass = self.mass_table if self.pos_table[2] < -1e-4 else 0 # lift phase / clench phase
        self.tableUid = p.createMultiBody(mass, boxId, -1, self.pos_table)
        constraint = p.createConstraint(self.tableUid, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], self.pivot_in_table, self.pivot_in_world)
        p.changeVisualShape(self.tableUid, -1, rgbaColor=[.3,.3,.3,1])

        # Add a box
        boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.length_object,]*3)
        self.objectUid = p.createMultiBody(self.mass_object, boxId, -1, self.pos_object)
        p.changeVisualShape(self.objectUid, -1, rgbaColor=[247/255, 143/255, 0/255, 1])

        # Add a 3-finger gripper
        self.gripperUid = p.loadURDF(fileName='asset/robotiq_3f_gripper_visualization/cfg/robotiq-3f-gripper_articulated.urdf', 
                                     basePosition=self.pos_gripper_base, 
                                     baseOrientation=self.quat_gripper_base,
                                     globalScaling=10,
                                     )
        p.changeDynamics(self.gripperUid, -1, mass=0) # fix the base link
        for i in range(-1, self.num_links-1):
            p.changeDynamics(self.gripperUid, i, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
            p.changeVisualShape(self.gripperUid, i, rgbaColor=[112/255, 190/255, 83/255, 1])
            
    def reset_states(self, states):
        """states: 6+6+9+1+9+1 DoF"""
        # Object kinematics
        pos = states[:self.dim_workspace]
        self.pos_object = [pos[0], pos[2], pos[1]] # pos: x,z,y
        self.quat_object = p.getQuaternionFromEuler(states[self.dim_workspace:2*self.dim_workspace])
        self.vel_object = states[2*self.dim_workspace:3*self.dim_workspace]
        self.vel_ang_object = states[3*self.dim_workspace:4*self.dim_workspace]

        # Gripper kinematics
        self.pos_gripper = states[4*self.dim_workspace:4*self.dim_workspace+self.num_movable_joints]
        self.vel_gripper = states[4*self.dim_workspace+self.num_movable_joints+1:-1]

        # Table kinematics
        self.pos_table = [0,0,states[4*self.dim_workspace+self.num_movable_joints]]
        self.quat_table = p.getQuaternionFromEuler([0,0,0])
        self.vel_table = [0,0,states[-1]]
        self.vel_ang_table = [0,0,0]

        # Reset the kinematics of the object, the gripper and the table
        p.resetBasePositionAndOrientation(self.objectUid, self.pos_object, self.quat_object)
        p.resetBaseVelocity(self.objectUid, self.vel_object, self.vel_ang_object)
        for i, jid in enumerate(self.movable_joints):
            p.resetJointState(self.gripperUid, jid, targetValue=self.pos_gripper[i], targetVelocity=self.vel_gripper[i])
        p.resetBasePositionAndOrientation(self.tableUid, self.pos_table, self.quat_table)
        p.resetBaseVelocity(self.tableUid, self.vel_table, self.vel_ang_table)

        # Reset joints of the gripper that is not movable
        for i, jid in enumerate(self.non_movable_joints):
            p.resetJointState(self.gripperUid, jid, targetValue=0.0, targetVelocity=0.0)

    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t = inputs[0]
        interval = int(int(t*240)/num_via_points)
        interval = 3 if interval==0 else interval
        # t1 = time.time()

        # Step the simulation
        via_points = []
        force_on_object = [self.mass_object*acc for acc in inputs[1:4]]
        torque_on_object = np.dot(np.diag(self.moment_object), np.asarray(inputs[4:7])).tolist()
        for i in range(int(t*240)):
            # Apply the calculated torques to all joints at once  
            p.setJointMotorControlArray(bodyUniqueId=self.gripperUid,
                                        jointIndices=self.movable_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self.start_gripper_pos,
                                        targetVelocities=[0,]*self.num_movable_joints,
                                        positionGains=self.stiffness,
                                        velocityGains=self.damping,
                                        forces=[10,]*self.num_movable_joints,)
            
            # Apply external force on object
            self.pos_object, _ = p.getBasePositionAndOrientation(self.objectUid)
            p.applyExternalForce(self.objectUid, -1, 
                                force_on_object,
                                self.pos_object, 
                                p.WORLD_FRAME)
            p.applyExternalTorque(self.objectUid, -1, 
                                torque_on_object,
                                p.WORLD_FRAME)
            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if print_via_points and (i % interval == 0 or i == int(t*240)-1):
                via_points.append([self.pos_object[0], self.pos_object[2]])
            if self.gui:
                time.sleep(3/240)
            
        # t2 = time.time()
        # print("!!!!!!!!Time elapsed: ", t2-t1)

        # Get the object and gripper states
        pos, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
        pos_object_GL = [pos[0], pos[2], pos[1]] # x,z,y
        self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
        self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
        joint_states = p.getJointStates(self.gripperUid, self.movable_joints)
        self.pos_gripper = [state[0] for state in joint_states]
        pos_table, _ = p.getBasePositionAndOrientation(self.tableUid)
        
        new_states = pos_object_GL + list(self.eul_object) + list(self.vel_object) + list(self.vel_ang_object) + self.pos_gripper + [pos_table[2],] # 3+3+3+3+9+1=22
        return new_states, via_points

    def finish_sim(self):
        # Clean up and close the simulation
        p.disconnect()


class forwardSimulationGripperMulti():
    def __init__(self, gui=False):
        self.visualShapeId = -1
        self.gui = gui
        if self.gui:
            p.connect(p.GUI) # p.GUI
        else:
            p.connect(p.DIRECT) # p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
        self.g = -9.81
        p.setGravity(0, 0, self.g)
    
    def set_params(self, params):
        # Kinodynamics
        self.mass_object, self.moment_object, self.length_object, self.movable_joints, self.target_gripper_joint_pos, self.lateral_friction_coef, self.num_objects, self.success_z_thres = params
        self.mass_table = 0
        # self.mass_gripper_base = 10
        self.num_movable_joints = len(self.movable_joints)
        self.num_joints = 4
        self.num_links = self.num_joints + 1
        self.non_movable_joints = []
        self.dim_workspace = 3
        self.force_z_on_gripper = 15
        self.stiffness = [1e1,] * self.num_movable_joints  # P gain for each joint
        self.damping = [1e9,] * self.num_movable_joints  # D gain for each joint
        self.fingertip_link_ids = [1,3,]

        self.pos_object = [-0.9,0,.6]
        self.quat_object = p.getQuaternionFromEuler([0,0,0])
        self.vel_object = [0,0,0]

        self.pos_table = [0,0,0]
        # self.pos_table = [0,0,self.start_env_pos[-1]]
        # self.vel_table = [0,0,0]
        # self.pivot_in_table = [0, 0, 0] # for the prismatic constraint
        # self.pivot_in_world = [0, 0, 0]

        self.start_gripper_pos = self.target_gripper_joint_pos
        self.pos_gripper_base = [0,0,2.7]
        self.quat_gripper_base = p.getQuaternionFromEuler([math.pi,0,0])

    def create_shapes(self):
        # Add a table
        boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1,1,0.3])
        # mass = self.mass_table if self.pos_table[2] < -1e-4 else 0 # lift phase / clench phase
        self.tableUid = p.createMultiBody(self.mass_table, boxId, -1, self.pos_table)
        # constraint = p.createConstraint(self.tableUid, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], self.pivot_in_table, self.pivot_in_world)
        p.changeVisualShape(self.tableUid, -1, rgbaColor=[.3,.3,.3,1])

        # Create objects
        objectId = []
        self.objectUid = []
        for i in range(self.num_objects):
            objectId.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.length_object,]*3))
            self.objectUid.append(p.createMultiBody(self.mass_object, objectId[i], -1, self.pos_object))
            p.changeVisualShape(self.objectUid[i], -1, rgbaColor=[247/255, 143/255, 0/255, 1])

        # Create a scooping gripper
        self.gripperUid = p.loadURDF(fileName='asset/lc_soft_enable_wide_grip/lc_soft_enable_wide_grip.urdf', 
                                     basePosition=self.pos_gripper_base, 
                                     baseOrientation=self.quat_gripper_base,
                                     globalScaling=10,
                                     )
        # p.changeDynamics(self.gripperUid, -1, mass=0) # fix the base link
        pivotInBox = [0, 0, 0] 
        pivotInWorld = [0, 0, 0]
        constraint = p.createConstraint(self.gripperUid, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], pivotInBox, pivotInWorld, p.getQuaternionFromEuler([1*math.pi,0,0]))
        for i in range(-1, self.num_links-1):
            p.changeDynamics(self.gripperUid, i, lateralFriction=self.lateral_friction_coef, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
            p.changeVisualShape(self.gripperUid, i, rgbaColor=[112/255, 190/255, 83/255, 1])
            
    def reset_states(self, states):
        """states: 12n+(4+1)*2 DoF"""
        # Object kinematics
        self.pos_object = [states[12*i:12*i+self.dim_workspace] for i in range(self.num_objects)]
        self.eul_object = [states[12*i+self.dim_workspace:12*i+2*self.dim_workspace] for i in range(self.num_objects)]
        self.quat_object = [p.getQuaternionFromEuler(self.eul_object[i]) for i in range(self.num_objects)]
        self.vel_object = [states[12*i+2*self.dim_workspace:12*i+3*self.dim_workspace] for i in range(self.num_objects)]
        self.vel_ang_object = [states[12*i+3*self.dim_workspace:12*i+4*self.dim_workspace] for i in range(self.num_objects)]

        # Gripper kinematics
        self.pos_gripper = states[(-1-self.num_joints)*2:-1-self.num_joints-1]
        self.vel_gripper = states[-1-self.num_joints:-1]
        self.pos_gripper_base = [0, 0, states[-1-self.num_joints-1]]
        self.vel_gripper_base = [0, 0, states[-1]]

        # Reset the kinematics of the object, the gripper
        for i in range(self.num_objects):
            p.resetBasePositionAndOrientation(self.objectUid[i], self.pos_object[i], self.quat_object[i])
            p.resetBaseVelocity(self.objectUid[i], self.vel_object[i], self.vel_ang_object[i])
        for i, jid in enumerate(self.movable_joints):
            p.resetJointState(self.gripperUid, jid, targetValue=self.pos_gripper[i], targetVelocity=self.vel_gripper[i])
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper_base, self.quat_gripper_base)
        p.resetBaseVelocity(self.gripperUid, self.vel_gripper_base, [0,0,0])

    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t = inputs[0]
        interval = int(int(t*240)/num_via_points)
        interval = 3 if interval==0 else interval
        # t1 = time.time()

        # Step the simulation
        via_points = []
        force_on_object = [[self.mass_object*inputs[6*i+1], self.mass_object*inputs[6*i+2], self.mass_object*inputs[6*i+3]] for i in range(self.num_objects)]
        torque_on_object = [np.dot(np.diag(self.moment_object), np.asarray(inputs[6*i+4:6*i+7])).tolist() for i in range(self.num_objects)]
        for i in range(int(t*240)):
            # TODO: apply upward forces on the gripper base??
            self.pos_gripper_base,_ = p.getBasePositionAndOrientation(self.gripperUid)
            p.applyExternalForce(self.gripperUid, -1, 
                                [0,0,self.force_z_on_gripper],
                                self.pos_gripper_base, 
                                p.WORLD_FRAME)
            
            # Apply the calculated torques to all joints at once  
            p.setJointMotorControlArray(bodyUniqueId=self.gripperUid,
                                        jointIndices=self.movable_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self.target_gripper_joint_pos, # target gripper joint positions in the scripted movement
                                        targetVelocities=[0,]*self.num_movable_joints,
                                        positionGains=self.stiffness,
                                        velocityGains=self.damping,
                                        forces=[10,]*self.num_movable_joints,)
            
            # Apply external force on object
            for i in range(self.num_objects):
                self.pos_object[i],_ = p.getBasePositionAndOrientation(self.objectUid[i])
                p.applyExternalForce(self.objectUid[i], -1, 
                                    force_on_object[i],
                                    self.pos_object[i], 
                                    p.WORLD_FRAME)
                p.applyExternalTorque(self.objectUid[i], -1,
                                    torque_on_object[i],
                                    p.WORLD_FRAME)
            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if print_via_points and (i % interval == 0 or i == int(t*240)-1):
                via_points.append([self.pos_object[0], self.pos_object[1]])
            if self.gui:
                time.sleep(3/240)
        # t2 = time.time()

        # Get the object and gripper states
        for i in range(self.num_objects):
            self.pos_object[i], self.quat_object[i] = p.getBasePositionAndOrientation(self.objectUid[i])
            self.eul_object[i] = p.getEulerFromQuaternion(self.quat_object[i]) # rad
            self.vel_object[i], self.vel_ang_object[i] = p.getBaseVelocity(self.objectUid[i])
        self.pos_gripper_base, _ = p.getBasePositionAndOrientation(self.gripperUid)
        joint_states = p.getJointStates(self.gripperUid, self.movable_joints)
        self.pos_gripper = [state[0] for state in joint_states]

        new_states = []
        for i in range(self.num_objects): # 12n
            new_states = new_states + list(self.pos_object[i]) + list(self.eul_object[i]) + \
                list(self.vel_object[i]) + list(self.vel_ang_object[i])
        new_states = new_states + self.pos_gripper + [self.pos_gripper_base[2],] # 12n+(4+1)

        return new_states, via_points

    def finish_sim(self):
        # Clean up and close the simulation
        p.disconnect()


class forwardSimulationShuffling():
    def __init__(self, gui=False):
        self.visualShapeId = -1
        self.gui = gui
        if self.gui:
            p.connect(p.GUI) # p.GUI
        else:
            p.connect(p.DIRECT) # p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
        self.g = -9.81
        # p.setGravity(0, 0, self.g)
    
    def set_params(self, params):
        # General
        self.dim_workspace = 3
        self.dim_se3 = 6
        self.dim_gripper = 1

        # Object
        self.pos_object = [0, 0, 4.55]
        self.quat_object = p.getQuaternionFromEuler([0,np.pi/2,0])
        self.vel_object = [0,0,0]
        self.target_positions = [0,0,0,0] # [0]*numJointsLink,
        self.target_velocities = [0,0,0,0] # [0.0,]*numJointsLink,
        self.mass_object = 1
        self.objectUid = p.loadURDF("asset/linkage.urdf", basePosition=self.pos_object, baseOrientation=self.quat_object)
        self.num_joints = p.getNumJoints(self.objectUid)
        self.dim_state = 2*(self.dim_se3 + self.num_joints) + self.dim_gripper # 21
        self.stiffness = [1e0,]*self.num_joints # [8e-3,5e-3,5e-3,1e-2]  # P gain for each joint
        self.damping = [1e-1,] * self.num_joints  # D gain for each joint

        # Gripper
        self.pos_gripper = [0, 0, 0]
        self.quat_gripper = p.getQuaternionFromEuler([0,0,0])
        self.vel_gripper = [0,0,0]
        self.vel_ang_gripper = [0,0,0]
        self.mass_up_gripper = 100

    def create_shapes(self):
        # Set up object
        for i in range(self.num_joints+1):
            p.changeDynamics(self.objectUid, i, lateralFriction=1, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)

        # Set up environment
        upperID = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3,2,.2])
        self.gripperUid = p.createMultiBody(self.mass_up_gripper, upperID, -1, [0,0,5.4])
        p.changeDynamics(self.gripperUid, -1, lateralFriction=1, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)
        pivotInBox = [0, 0, 0] 
        pivotInWorld = [0, 0, 3]
        _ = p.createConstraint(self.gripperUid, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], pivotInBox, pivotInWorld)

        lowerID = p.loadURDF("plane.urdf")
        p.changeDynamics(lowerID, -1, lateralFriction=1, spinningFriction=0, 
                            rollingFriction=0, linearDamping=0, angularDamping=0)

    def reset_states(self, states):
        """states: 2*6+2*4+1+1 DoF"""
        # Object kinematics
        pos = states[:self.dim_workspace]
        self.pos_object = [pos[2], pos[0], pos[1]] # pos: y,z,x
        self.quat_object = p.getQuaternionFromEuler(states[self.dim_workspace:2*self.dim_workspace])
        self.vel_object = states[2*self.dim_workspace:3*self.dim_workspace]
        self.vel_ang_object = states[3*self.dim_workspace:4*self.dim_workspace]
        self.pos_object_joints = states[4*self.dim_workspace:4*self.dim_workspace+self.num_joints]
        self.vel_object_joints = states[4*self.dim_workspace+self.num_joints:4*self.dim_workspace+2*self.num_joints]

        # Gripper kinematics
        self.pos_gripper = [0, 0, states[-2]]
        self.vel_gripper = [0, 0, states[-1]]

        # Reset the kinematics of the object and the gripper
        p.resetBasePositionAndOrientation(self.objectUid, self.pos_object, self.quat_object)
        p.resetBaseVelocity(self.objectUid, self.vel_object, self.vel_ang_object)
        for i, jid in enumerate(list(range(self.num_joints))):
            p.resetJointState(self.objectUid, jid, targetValue=self.pos_object_joints[i], targetVelocity=self.vel_object_joints[i])

        # Reset gripper
        p.resetBasePositionAndOrientation(self.gripperUid, self.pos_gripper, self.quat_gripper)
        p.resetBaseVelocity(self.gripperUid, self.vel_gripper, self.vel_ang_gripper)

    def run_forward_sim(self, inputs, print_via_points=True, num_via_points=10):
        t = inputs[0]
        interval = int(int(t*240)/num_via_points)
        interval = 3 if interval==0 else interval

        # Step the simulation
        via_points = []
        external_torque = inputs[1:] # list[4,]
        for i in range(int(t*240)):
            # Calculate the torques induced by joint elasticity and external force
            joint_states = p.getJointStates(self.objectUid, list(range(self.num_joints)))
            joint_angles = [state[0] for state in joint_states]
            joint_vels = [state[1] for state in joint_states]
            error_angles = [desired - current for desired, current in zip(self.target_positions, joint_angles)]
            error_vels = [desired - current for desired, current in zip(self.target_velocities, joint_vels)]
            target_torques = [kp*e - kd*av for kp, e, kd, av in zip(self.stiffness, error_angles, self.damping, error_vels)]
            total_torques = [ext+tar for ext, tar in zip(external_torque, target_torques)]

            # Mimic elastic cards stack behavior
            p.setJointMotorControlArray(bodyUniqueId=self.objectUid,
                                        jointIndices=list(range(self.num_joints)),
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[100,]*self.num_joints,
                                        forces=total_torques,)
            p.stepSimulation()

            # Print object via-points along the trajectory for visualization
            if print_via_points and (i % interval == 0 or i == int(t*240)-1):
                self.pos_object, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
                via_points.append([self.pos_object[1], self.pos_object[2]])
            if self.gui:
                time.sleep(3/240)

        # Get the object and gripper states
        pos, self.quat_object = p.getBasePositionAndOrientation(self.objectUid)
        pos_object_GL = [pos[1], pos[2], pos[0]] # y,z,x
        self.eul_object = p.getEulerFromQuaternion(self.quat_object) # rad
        self.vel_object, self.vel_ang_object = p.getBaseVelocity(self.objectUid)
        joint_states = p.getJointStates(self.objectUid, list(range(self.num_joints)))
        self.pos_object_joints = [state[0] for state in joint_states]
        self.vel_object_joints = [state[1] for state in joint_states]
        self.pos_gripper, _ = p.getBasePositionAndOrientation(self.gripperUid)

        new_states = (pos_object_GL + list(self.eul_object) + list(self.vel_object) + list(self.vel_ang_object) + 
                      self.pos_object_joints + self.vel_object_joints + [self.pos_gripper[2],])
        return new_states, via_points

    def finish_sim(self):
        # Clean up and close the simulation
        p.disconnect()
