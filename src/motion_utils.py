import numpy as np

def get_odom(lidar_data):
    '''
        Get absolute pose of the robot at time t
        
        Input:
            lidar_data - lidar scan data
        Outputs:
            robot_pose - absolute pose of the robot in the world frame for each time stamp
    '''
    robot_pose = np.zeros((len(lidar_data),3))
    curr_x = 0
    curr_y = 0
    curr_angle = 0

    for idx, data in enumerate(lidar_data):
        curr_x += data['delta_pose'][0][0]
        curr_y += data['delta_pose'][0][1]
        curr_angle += data['delta_pose'][0][2]
        robot_pose[idx][0] = curr_x
        robot_pose[idx][1] = curr_y
        robot_pose[idx][2] = curr_angle

    return robot_pose

def get_motion(robot_pose, lidar_data, idx, step_size):
    '''
        Get relative motion between each iteration
        
        Input:
            robot_pose - absolute pose of the robot in the world frame for each time stamp
            lidar_data - lidar scan data
            idx        - current iteration
            step_size  - step size
        Outputs:
            delta_pose - relative motion in (x, y, theta)
    '''
    if(idx >= step_size): 
        delta_pose = robot_pose[idx] - robot_pose[idx-step_size]
    else: 
        delta_pose = lidar_data['delta_pose']
        
    return delta_pose

def motion_model_predict(particles, motion, var_scale):
    '''
        motion model prediction
        
        Input:
            particles    - list of particle states
            motion       - relative motion in (x, y, theta)
            var_scale    - scaling factor of Gaussian noise
        Outputs:
            particles    - list of predicted particle states
    '''
    motion_noise = np.random.randn(particles.shape[0],3) * var_scale
    particles = particles + motion + motion_noise
    particles[:,2] = particles[:,2] % (2*np.pi)
    return particles
