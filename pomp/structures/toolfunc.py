import math

def limit_angle_to_pi(angle, theta_min=-math.pi, theta_max=math.pi):
    # Make input angle fall in [-pi, pi]
    if angle > theta_max:
        theta = (angle - theta_max) % (2*math.pi) + theta_min
    elif angle < theta_min:
        theta = theta_max - (theta_min - angle) % (2*math.pi)
    else:
        theta = angle
    return theta

def correct_euler(euler):
    if euler[0] > -1 and euler[0] < 1:
        y = euler[1]
    elif (euler[0] > 1 or euler[0] < -1) and euler[1] > 0:
        y = math.pi - euler[1]
    elif (euler[0] > 1 or euler[0] < -1) and euler[1] < 0:
        y = -math.pi - euler[1]

    return [0, y, 0]