import os
import numpy as np

from tools        import particleInit, joint_align, resampling_wheel, plot
from load_data    import get_joint, get_lidar
from map_utils    import mapInit, mapping, measurement_model_update
from motion_utils import get_odom, get_motion, motion_model_predict


def main():
    # read data
    lidar_file_name = "lidar/train_lidar2"
    joint_file_name = "joint/train_joint2"
    lidar_data = get_lidar(lidar_file_name)
    joint_data = get_joint(joint_file_name)

    robot_pose = get_odom(lidar_data)
    lidar_data = joint_align(lidar_data, joint_data)
    
    # particles init
    P = particleInit(num=3)

    # grid map init
    MAP = mapInit()

    # init parameters
    step_size    = 20
    trajectory   = np.empty(shape=(1,2))
    var_scale    = np.array([0.001, 0.001, 0.01*np.pi/180])
    lidar_angles = np.arange(-135,135.25,0.25)*np.pi/180

    for i in range(0, len(lidar_data), step_size):
        if(i%100==0): 
            print(i)
        
        ''' Predict '''
        delta_pose = get_motion(robot_pose, lidar_data[i], i, step_size)
        P['states'] = motion_model_predict(P['states'], delta_pose, var_scale)

        ''' Update '''
        best_particle = measurement_model_update(MAP, P, lidar_data[i], lidar_angles)
        trajectory = np.vstack((trajectory, [int(best_particle[0]/MAP['res']) + MAP['sizex']//2, int(best_particle[1]/MAP['res']) + MAP['sizey']//2]))

        ''' Mapping '''
        MAP['map'] = mapping(MAP['map'], lidar_data[i], best_particle, MAP['res'], lidar_angles)
        
        ''' Resample '''
        N_eff = 1 / np.sum(P['weight']**2)
        if N_eff < 0.3 * P['number']:
            print("Resampling...")
            P = resampling_wheel(P)
            
    # plot 
    plot(MAP['map'], MAP['res'], robot_pose, trajectory)


if __name__ == "__main__":
    main()
