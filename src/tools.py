import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    '''
        Softmax function
    '''
    x = np.exp(x-np.max(x))
    return x / x.sum()

def particleInit(num=128):
    '''
        Initialize particles

        Input:
            num - number of particles
        Outputs:
            Particles - dictionary contains particles info
    '''
    Particles = {}
    Particles['number'] = num
    Particles['weight'] = np.ones(Particles['number']) / Particles['number']
    Particles['states'] = np.zeros((Particles['number'], 3)) + np.random.randn(Particles['number'],3) * np.array([0.1, 0.1, 0.1*np.pi/180])
    return Particles

def t_align(lidar_time, joint_time):
    '''
        Time stamp alignment
        
        Input:
            lidar_time - current lidar scan time stamp
            joint_time - list of joint time stamp
        Outputs:
            idx - corresponding joint time stamp index
    '''
    idx = np.argmin(abs(lidar_time - joint_time))
    return idx

def joint_align(lidar_data, joint_data):
    '''
        Get the corresponding joint data for lidar scan
        
        Input:
            lidar_data - lidar scan data
            joint_data - head and neck angle of the robot
        Outputs:
            lidar_data - lidar scan data with corresponding joint angle
    '''
    joint_time = joint_data['ts']
    for i in range(len(lidar_data)):
        lidar_time = lidar_data[i]['t'][0][0]
        idx = t_align(lidar_time, joint_time)
        lidar_data[i]['joint'] = [joint_data['head_angles'][0][idx], joint_data['head_angles'][1][idx]]
    
    return lidar_data

def get_roration(head_angle, pose):
    '''
        Get transformation matrix from lidar to world frame
        
        Input:
            head_angle - neck and head angle
            pose       - absolute odometry in world frame
        Outputs:
            T - transformation matrix from lidar to world frame
    '''
    phi      = head_angle[0]  # neck angle
    theta    = head_angle[1]  # head angle
    x_diff   = pose[0]
    y_diff   = pose[1]
    phi_diff = pose[2]
    h_T_l = np.array([[1, 0, 0, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, 1, 0.15], 
                      [0, 0, 0, 1]])
    b_T_h = np.array([[np.cos(phi)*np.cos(theta), -np.sin(phi), np.cos(phi)*np.sin(theta), 0], 
                      [np.sin(phi)*np.cos(theta), np.cos(phi), np.sin(phi)*np.sin(theta), 0], 
                      [-np.sin(theta), 0, np.cos(theta), 0.33], 
                      [0, 0, 0, 1]])
    w_T_b = np.array([[np.cos(phi_diff), -np.sin(phi_diff), 0, x_diff], 
                      [np.sin(phi_diff), np.cos(phi_diff), 0, y_diff], 
                      [0, 0, 1, 0.93], 
                      [0, 0, 0, 1]])
    T = np.dot(w_T_b, np.dot(b_T_h, h_T_l))
    return T

def polar2cart(points, angles):
    '''
        Transform points from polar to catesian coordinate
        
        Input:
            points - point distance measured from lidar
            angles - lidar scan range, from -135° to 135°
        Outputs:
            x - x coordinate of points
            y - y coordinate of points
    '''
    x, y = points * np.cos(angles), points * np.sin(angles)
    return x, y

def transform(x, y, joint, pose):
    '''
        Transform data from lidar frame to world frame
        
        Input:
            X     - x coordinate of the data in lidar frame
            Y     - y coordinate of the data in lidar frame
            joint - current robot joint angle (head_angle, neck_angle)
            pose  - current robot pose (x, y, theta)
        Outputs:
            post_scan - lidar data in world frame (len(data_points) x 4)
    '''
    # adjust data structure for easier computation later
    scan = np.empty([1,4])
    for (x_i,y_i) in zip(x.ravel().tolist(), y.ravel().tolist()):
        scan = np.vstack((scan, np.array([[x_i, y_i, 0, 1]])))
    scan = scan[1:].T
    
    # transform the lidar scan to the world frame
    T = get_roration(joint, pose)
    scan = np.dot(T, scan)
    scan = scan.T
    
    return scan

def filtering(scan, pose, zmin=0.1, zmax=2.7, min_dist=0.1, max_dist=15):
    '''
        Remove scan points that are too close, too far, or hit the ground
        
        Input:
            scan        - lidar data in world frame
            zmin        - minimum value of the z coordinate of the point
            zmax        - maximum value of the z coordinate of the point
            dist_thresh - distance threshold
        Outputs:
            post_scan - lidar data after filtering, (4 x len(data_points))
    '''
    post_scan = np.empty([1,4])

    for i in range(len(scan)):
        x = scan[i][0] - pose[0]
        y = scan[i][1] - pose[1]
        distance = np.sqrt(x**2 + y**2)
        if(scan[i][2] < zmax and scan[i][2] > zmin and distance > min_dist and distance < max_dist):
            post_scan = np.vstack((post_scan, scan[i]))
        
    post_scan = post_scan[1:].T
    return post_scan

def resampling_wheel(P):
    '''
        Resampling step of particle filter
        For more info, see "https://www.youtube.com/watch?v=wNQVo6uOgYA"
        
        Input:
            P - dictionary contain particle info
        Outputs:
            P - resample particles
    '''
    N = len(P['number'])
    beta = 0
    chose_idx = []
    index = int(np.random.choice(np.arange(N), 1, p=[1/N]*N))  # choose an index uniformly

    for _ in range(N):
        beta = beta + np.random.uniform(low=0, high=2*np.max(P['weight']), size=1)
        while(P['weight'][index] < beta):
            beta  = beta - P['weight'][index]
            index = (index+1) % N
        chose_idx.append(index)
    
    P['states'] = P['states'][chose_idx]
    P['weight'].fill(1/P['number'])

    return P

def plot(grid, res, robot_pose, trajectory):
    '''
        Plot the result
        
        Input:
            grid       - grid map
            res        - map resolution
            robot_pose - LiDAR odometry data
            trajectory - odometry after particle filter
    '''
    print("Plot...")
    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(121)
    plt.plot(robot_pose.T[0], robot_pose.T[1], label="Lidar Odom")
    plt.scatter((trajectory[1:].T[0] - grid.shape[0]//2)*res, (trajectory[1:].T[1] - grid.shape[1]//2)*res, label="Particle Filter Odom", s=2, c='r')
    plt.legend(loc='upper left')
    
    ax2 = fig.add_subplot(122)
    plt.imshow(grid, cmap='gray', vmin=-100, vmax=100, origin='lower')
    plt.scatter(trajectory[1:].T[1], trajectory[1:].T[0], s=1, c='r')
    #plt.colorbar()
    plt.title("Occupancy grid (log-odds)")
    
    plt.show()
