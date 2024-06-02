import sys
sys.path.append('/home/yif/Documents/KTH/git/dynamicCageForMPPI/pomp')
import torch
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal
from ..example_problems.planepush import *
import copy
import math 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import torch.nn as nn
import time 
# from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

# Function to check if x is within boundaries
def is_within_boundaries(x, x_boundary):
    for i in range(len(x)):
        if x[i] < x_boundary[i][0] or x[i] > x_boundary[i][1]:
            return False
    return True

def predict(model, input_data, scale_, min_):
    """
        input_data: torch.tensor(ndata,nx).cuda()
        scale_: torch.tensor(nx,).cuda()
        min_: torch.tensor(nx,).cuda()

        predictions: torch.tensor(ndata,).cuda()
    """
    # Preprocess the input data
    input_data_scaled = input_data * scale_ + min_
    
    # Perform prediction
    with torch.no_grad():
        predictions = model(input_data_scaled)

    return predictions

def get_object_corners(pose, half_extents_object):
    # Calculate the corner points of the rotated box
    half_length, half_height = half_extents_object
    center_x, center_y, theta = pose
    cos_theta = math.cos(-theta)
    sin_theta = math.sin(-theta)

    p1 = (center_x - cos_theta * half_length + sin_theta * half_height,
            center_y + sin_theta * half_length + cos_theta * half_height)
    p2 = (center_x + cos_theta * half_length + sin_theta * half_height,
            center_y - sin_theta * half_length + cos_theta * half_height)
    p3 = (center_x + cos_theta * half_length - sin_theta * half_height,
            center_y - sin_theta * half_length - cos_theta * half_height)
    p4 = (center_x - cos_theta * half_length - sin_theta * half_height,
            center_y + sin_theta * half_length - cos_theta * half_height)
    
    return torch.tensor([p1, p2, p3, p4]) # 4x2

def draw_object(corners, ax, alpha=0.5):
    points = corners.numpy() # 4x2
    x, y = points[:, 0], points[:, 1]
    for i in range(len(points)):
        next_i = (i + 1) % len(points)  # To loop back to the first point
        ax.plot([x[i], x[next_i]], [y[i], y[next_i]], 'orange', alpha=alpha, linewidth=4)  # 'b-' for blue lines

# Define the same model class used for training the cage stability network
class NeuralNetwork2DOutput(nn.Module):
    def __init__(self):
        super(NeuralNetwork2DOutput, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output layer for 2D output
        )
    def forward(self, x):
        return self.linear_relu_stack(x)

class NeuralNetwork3DOutput(nn.Module):
    def __init__(self):
        super(NeuralNetwork3DOutput, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Adjusted for 3D output
        )
    def forward(self, x):
        return self.linear_relu_stack(x)
    
class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """
    def __init__(self, 
                 nx, 
                 nu, 
                 cage,
                 dynamics_sim,
                 K, 
                 T, 
                 running_cost, 
                 device="cpu", 
                 terminal_state_cost=None,
                 lambda_=1.,
                 goal_thres=0.1,
                 noise_mu=torch.tensor(0., dtype=torch.double),
                 noise_sigma=torch.tensor(1., dtype=torch.double),
                 u_init=torch.tensor(1., dtype=torch.double),
                 num_vis_samples=5,
                 dt=0.5,
                 cost_type='simple',
                 ):
        self.goal_thres = goal_thres
        self.cage = cage
        self.dynamics_sim = dynamics_sim
        self.d = device
        self.K = K  # N_SAMPLES
        self.T = T  # HORIZON
        self.dt = dt
        self.num_vis_samples = num_vis_samples # nu. of rollouts
        self.cost_type = cost_type
        self.theta_max = math.pi/2
        self.theta_min = -math.pi/2
        self.max_acceleration = 3 # 2 for dataset generation

        # dimensions of state and control
        self.nx = nx
        self.nu = nu
        self.lambda_ = lambda_

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        # T x nu control sequence
        self.u_init = u_init
        self.U = u_init.repeat(self.T,1).to(self.d) + self.noise_dist.sample((self.T,)) # T x nu, initial control sequence

        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.state = None
        self.rollout_state = None
        self.rollout_cutdown_id = None
        self.reached_goal = 0

        if self.cost_type != 'simple':
            self._prepare_nn()

        self.g = self.cage.gravity
        self.control_space = self.cage.controlSpace()
        self.u_boundary = [
                           [-self.max_acceleration, self.max_acceleration],
                           [-self.max_acceleration, self.max_acceleration],
                           [0,0],
                           ]
        self.c_space_boundary = self.cage.c_space_boundary
        self.half_extents_object = self.cage.half_extents_object

    def _prepare_nn(self):
        if self.cost_type == 'ours':
            # scaler_filename = 'data/evaluation/mppi/18k_dataset_from_mppi/approaches/ours/scaler_minmax_ours.pkl'
            # model_filename = 'data/evaluation/mppi/18k_dataset_from_mppi/approaches/ours/model_ours.pth'
            scaler_filename = 'data/evaluation/mppi/28k_dataset_from_mppi/approaches/ours/scaler_minmax_ours.pkl'
            model_filename = 'data/evaluation/mppi/28k_dataset_from_mppi/approaches/ours/model_ours.pth'
            scaler = joblib.load(scaler_filename)
            self.model = NeuralNetwork2DOutput()
            self.model.load_state_dict(torch.load(model_filename))
        elif self.cost_type == 'hou':
            # scaler_filename = 'data/evaluation/mppi/18k_dataset_from_mppi/approaches/hou/scaler_minmax_hou.pkl'
            # model_filename = 'data/evaluation/mppi/18k_dataset_from_mppi/approaches/hou/model_hou.pth'
            scaler_filename = 'data/evaluation/mppi/28k_dataset_from_mppi/approaches/hou/scaler_minmax_hou.pkl'
            model_filename = 'data/evaluation/mppi/28k_dataset_from_mppi/approaches/hou/model_hou.pth'
            scaler = joblib.load(scaler_filename)
            self.model = NeuralNetwork3DOutput()
            self.model.load_state_dict(torch.load(model_filename))

        # Convert the scaler parameters to PyTorch tensors
        self.scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=self.d)
        self.scaler_min = torch.tensor(scaler.min_, dtype=torch.float32, device=self.d)
        self.model.to(self.d)
        self.model.eval()

    def _reset_start_goal(self, x_init):
        self.state_start = torch.tensor(x_init, device=self.d) # dim=12

    def _check_collision(self):
        # Check if bodies are in collision
        self.cage.dynamics_sim.reset_states(self.state_start.cpu().tolist())
        is_in_collision = self.cage.dynamics_sim.check_collision()
        if is_in_collision:
            print('!!!Collision at start state!!!')
            return True
        print('!!!Not in Collision!!!')
        return False

    def _initialize_rollout_container(self):
        self.rollout_state = torch.zeros((self.K, self.T+1, self.nx), device=self.d)
        self.rollout_quality = torch.zeros((self.K, self.T+1, 3), device=self.d)
        self.rollout_cutdown_id = torch.zeros((self.K), device=self.d).fill_(self.T+1)

    def _start_action_consideration(self):
        # reseample noise each time we take an action; these can be done at the start
        self.cost_total = torch.zeros(self.K, device=self.d)
        self.noise = self.noise_dist.sample((self.K, self.T))
        # cache action cost
        self.action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv

    def _compute_total_cost(self, k):
        vis_rollouts = False
        if k < self.num_vis_samples: vis_rollouts = True
        state = self.state.clone()
        self.rollout_state[k,0,:] = state.clone().detach() # save rollouts for dataset generation

        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]

            # Clamping each element in u to its respective boundary
            for i in range(len(perturbed_action_t)): 
                perturbed_action_t[i] = torch.clamp(perturbed_action_t[i], min=self.u_boundary[i][0], max=self.u_boundary[i][1])

            # Rollout
            state = self.control_space.nextState(state.tolist(), 
                                                 [self.dt,]+perturbed_action_t.tolist(), 
                                                 is_planner=True) # TODO: dynamics take 95% of the runtime
            quality = self.control_space.dynamics_sim.heuristics
            
            # State space boundary
            is_valid_state = is_within_boundaries(state, self.c_space_boundary)
            if not is_valid_state:
                self.cost_total[k] += 1e4
                if vis_rollouts and t+1 <= self.T: 
                    stacked_state = torch.tensor(state).unsqueeze(0).expand(self.T-(t+1)+1, -1)
                    self.rollout_state[k,t+1:,:] = stacked_state # save rollouts for visualization
                self.rollout_cutdown_id[k] = t + 1
                state = torch.tensor(state, device=self.d)
                break

            # Cage cost TODO: the inference can be done in parallel once every K*T times
            # increased_weight = (t+1) / ((1+self.T)*self.T/2)
            state = torch.tensor(state, device=self.d)
            
            if self.cost_type == 'simple':
                cost_t = self.running_cost(state, perturbed_action_t).item()
                self.cost_total[k] += cost_t

            # Add action perturbation cost
            # c_action = perturbed_action_t @ self.action_cost[k, t]
            # self.cost_total[k] += c_action

            self.rollout_state[k,t+1,:] = state.clone().detach().to(self.d) # save rollouts for dataset generation
            self.rollout_quality[k,t+1,:] = torch.tensor(quality, device=self.d) # save quality metrics for dataset generation

            # Break the for loop if current state is in goal
            if abs(state[1]-self.cage.y_obstacle) < self.goal_thres: # block already against the wall (PlanePush)
                if vis_rollouts and t+2 <= self.T: 
                    stacked_state = torch.tensor(state).unsqueeze(0).expand(self.T-(t+2)+1, -1)
                    self.rollout_state[k,t+2:,:] = stacked_state # save rollouts for visualization
                if t+2 <= self.T: 
                    self.rollout_cutdown_id[k] = t+2
                break

        # this is the additional terminal cost (running state cost at T already accounted for)
        if self.terminal_state_cost:
            self.cost_total[k] += self.terminal_state_cost(state)

    def _add_cage_cost(self, weights_ours=[-5,-1,], weights_hou=[1,-1,-1,],):
        # Inference in batches
        # Create a mask to select the columns based on rollout_cutdown_id
        mask = torch.arange(self.rollout_state.shape[1], device=self.d).expand(self.rollout_state.shape[0], -1) < self.rollout_cutdown_id.view(-1, 1)

        # Apply the mask to state and reshape the result
        stacked_states = self.rollout_state[mask].reshape(-1, self.rollout_state.shape[-1]) # torch.Size([-1,self.nx]), gpu

        # Make a prediction
        stability_cage = predict(self.model, stacked_states, self.scaler_scale, self.scaler_min)
        if self.cost_type == 'ours':
            # cost_cage_success = torch.max(torch.tensor(-3.0, device='cuda:0'), 1 - weights_ours[0]*stability_cage[:,0]) # torch.Size([-1,1]), gpu
            cost_cage_success = weights_ours[0] * stability_cage[:,0] # torch.Size([-1,1]), gpu
            cost_cage_maneuver = weights_ours[1] * stability_cage[:,1] # torch.Size([-1,1]), gpu
            cost_cage = cost_cage_success + cost_cage_maneuver
        elif self.cost_type == 'hou':
            # distance,S_stick,S_engage
            cost_s_distance = weights_hou[0] * stability_cage[:,0] # torch.Size([-1,1]), gpu
            cost_s_stick = weights_hou[1] * stability_cage[:,1] # torch.Size([-1,1]), gpu
            cost_s_engage = weights_hou[2] * stability_cage[:,2] # torch.Size([-1,1]), gpu
            cost_cage = cost_s_distance + cost_s_stick + cost_s_engage

        # Calculate cumulative sum of rollout_cutdown_id
        cumulative_rollout = torch.cumsum(self.rollout_cutdown_id, dim=0)

        # Generate start indices of each group
        start_indices = cumulative_rollout - self.rollout_cutdown_id

        # Create a range tensor for comparison
        range_tensor = torch.arange(cumulative_rollout[-1], device=self.d)

        # Broadcast and create a mask for grouping
        group_mask = (range_tensor >= start_indices[:, None]) & (range_tensor < cumulative_rollout[:, None]) # torch.Size([nsample,ndata])

        # Sum the elements of cost_cage for each group
        cost_cage_total = torch.sum(cost_cage.reshape(1,-1) * group_mask, dim=1) # (self.K,)
        self.cost_total = self.cost_total + cost_cage_total # (self.K,)

    def _ensure_non_zero(self, cost, beta, factor):
        return torch.exp(-factor * (cost - beta))

    def command(self, state):
        """MPPI algorithm block II in the paper."""
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.U.dtype, device=self.d) # get state estimate

        self._start_action_consideration()
        # TODO dynamics not easily parallelizable - refer to this https://openreview.net/pdf?id=fvfZKL1hCx for robot simulation
        for k in range(self.K):
            self._compute_total_cost(k) # rollout dynamics and cost TODO:95% of total runtime

        # Add penalization against configurations that are unmaneuverable or of less probability of tasks success
        if self.cost_type != 'simple':
            self._add_cage_cost()
        
        beta = torch.min(self.cost_total) # min cost
        cost_total_non_zero = self._ensure_non_zero(self.cost_total, beta, 1/self.lambda_)

        eta = torch.sum(cost_total_non_zero) # normalizer
        omega = (1. / eta) * cost_total_non_zero # weights of each sample
        for t in range(self.T): # retrieve the control sequence by importance sampling
            self.U[t] += torch.sum(omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.U[0] # send to actuator

        # Shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

        return action


def run_mppi(mppi, iter=10, episode=0, do_bullet_vis=0):
    xhist = torch.zeros((iter+1, mppi.nx), device=mppi.d)
    uhist = torch.zeros((iter, mppi.nu), device=mppi.d)
    object_hist = torch.zeros((iter+1, 4, 2), device=mppi.d) # rectangular object, 4 corners, 2d points
    rollouts_hist = torch.zeros((iter, mppi.num_vis_samples, mppi.T+1, mppi.nx), device=mppi.d)
    rollouts_quality_hist = torch.zeros((iter, mppi.num_vis_samples, mppi.T+1, 3), device=mppi.d)
    cutdown_hist = torch.zeros((iter, mppi.num_vis_samples), device=mppi.d).fill_(mppi.T+1) # cutdown of #step in the horizon because reaching goals
    final_traj = torch.zeros((iter+1, mppi.nx), device=mppi.d)

    mppi.reached_goal = 0
    cutdown_iter = iter
    state = mppi.state_start
    final_traj[0,:] = state.clone().detach()
    mppi._initialize_rollout_container()
    xhist[0] = torch.tensor(state)
    object_hist[0] = get_object_corners(state[:3], mppi.half_extents_object).clone().detach()
    capture_labelset = []
    success_labelset = []
    for t in range(iter):
        print('')
        print('----------iter----------', t)
        action = mppi.command(state)
        state = mppi.control_space.nextState(state.tolist(), [mppi.dt,]+action.tolist(), is_planner=True)
        state = torch.tensor(state, device=mppi.d)
        # stability_cage = predict(mppi.model, state.reshape(-1, mppi.nx), mppi.scaler_scale, mppi.scaler_min)[0,0]
        # cost = 0. # TODO: print cost

        # Log and visualize
        final_traj[t+1,:] = state.clone().detach()
        xhist[t+1] = torch.tensor(state, device=mppi.d)
        uhist[t] = torch.tensor(action, device=mppi.d)
        object_hist[t+1] = get_object_corners(state[:3], mppi.half_extents_object).clone().detach()

        if t % 1 == 0:
            visualize_mppi(mppi, xhist, uhist, object_hist, t, epi=episode)
        
        # Save maneuverability labels
        cage = PlanePush(state.tolist(), mppi.dynamics_sim)
        capture_exists_label = 0 if cage.complementCaptureSet().contains(state[:9].tolist()) else 1
        capture_labelset.append([episode, t,] + [capture_exists_label,])

        # Check if goal is reached
        curr = state[1] # object position
        print('dist_to_goal', abs(mppi.cage.y_obstacle - curr))
        dist_to_goal = abs(mppi.cage.y_obstacle - curr)
        if dist_to_goal < mppi.goal_thres:
            print('REACHED!')
            cutdown_iter = t
            mppi.reached_goal = 1
            if do_bullet_vis:
                mppi.cage.dynamics_sim.finish_sim()
            visualize_mppi(mppi, xhist, uhist, object_hist, t, mppi.reached_goal, epi=episode, do_bullet_vis=do_bullet_vis)
            break

        # Save rollouts and clean up the container
        rollouts_hist[t,:,:,:] = mppi.rollout_state[:mppi.num_vis_samples,:,:]
        rollouts_quality_hist[t,:,:,:] = mppi.rollout_quality[:mppi.num_vis_samples,:,:]
        cutdown_hist[t,:] = mppi.rollout_cutdown_id[:mppi.num_vis_samples]
        mppi._initialize_rollout_container()

    success_labelset.append([episode,] + [mppi.reached_goal,])
    dist_to_goal = abs(mppi.cage.y_obstacle - curr)
    if dist_to_goal > mppi.goal_thres:
        print('NOT REACHED!')

    return rollouts_hist, cutdown_hist, cutdown_iter, rollouts_quality_hist, success_labelset, capture_labelset, final_traj


def visualize_mppi(mppi, xhist, uhist, object_hist, t, reached_goal=False, epi=0, do_bullet_vis=0):
    if reached_goal and do_bullet_vis:
        xhist = xhist.cpu().tolist()
        uhist = uhist.cpu().tolist()
        dynamics_sim = forwardSimulationPlanePushPlanner(gui=do_bullet_vis)
        dynamics_sim.set_params(mppi.cage.params)
        dynamics_sim.create_shapes()
        for i in range(t+1):
            print('----------vis----------', i)
            dynamics_sim.reset_states(xhist[i])
            dynamics_sim.run_forward_sim([mppi.dt,]+uhist[i])
        dynamics_sim.finish_sim()
    else:
        if reached_goal: t += 1
        # Visualize the basic setup
        fig, ax = plt.subplots()

        if not reached_goal:
            ax.plot(mppi.rollout_state[:mppi.num_vis_samples,:,0].T.cpu(), 
                    mppi.rollout_state[:mppi.num_vis_samples,:,1].T.cpu(), 
                    'k', alpha=0.8, zorder=3, linewidth=1) # rollout obj states
            ax.plot(mppi.rollout_state[0,:,0].cpu(), mppi.rollout_state[0,:,1].cpu(), 'k', alpha=0.1, label="rollout")

        ax.plot(xhist[:t+1,0].cpu().tolist(), xhist[:t+1,1].cpu().tolist(), 'orange', linewidth=1) # obj past trajectory
        ax.scatter(xhist[:t+1,0].cpu().tolist(), xhist[:t+1,1].cpu().tolist(), c='orange', s=11, label="object") # obj past positions
        ax.scatter(xhist[0,6].cpu().tolist(), xhist[0,7].cpu().tolist(), c='g', s=8, label="gripper", alpha=1) # gripper past positions
        ax.plot(xhist[:t+1,6].cpu().tolist(), xhist[:t+1,7].cpu().tolist(), c='g', linewidth=1)
        for i in range(t+1): # gripper past positions (circle)
            c1 = plt.Circle(xhist[i,6:8].cpu().tolist(), .1, color='g', linewidth=3, fill=0, zorder=7, alpha=float((i+1)/(t+1))**2.8)
            ax.add_patch(c1)

        for i in range(t+1):
            draw_object(object_hist[i].cpu(), ax, alpha=float((i+1)/(t+1))**2.7) # object past poses

        rectangle = patches.Rectangle((0, 9), 10, 0.5, edgecolor='black', facecolor='black', fill=True, linewidth=1, alpha=0.6) # wall
        ax.add_patch(rectangle)

        ax.set_xlim([2,6])
        ax.set_ylim([6.5,9.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.legend(loc="upper left")
        ax.set_aspect("equal")
        plt.tight_layout()
        # plt.show()
        fig.savefig('mppi-plot-{}-{}.png'.format(epi, t), dpi=300)
        plt.close()