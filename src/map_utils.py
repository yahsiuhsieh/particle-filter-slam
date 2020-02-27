import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from tools import  softmax, polar2cart, transform, filtering
from mpl_toolkits.mplot3d import Axes3D

def mapInit(xmin=-30, xmax=30, ymin=-30, ymax=30, res=0.05):
    '''
        Initialize grid map

        Input:
            xmin - minimum x value in meter
            ymin - minimum y value in meter
            xmax - maximum x value in meter
            ymax - maximum y value in meter
            res  - resolution in meter
        Outputs:
            MAP - dictionary contains map info
    '''
    MAP = {}
    MAP['res']   = res  #meters
    MAP['xmin']  = xmin #meters
    MAP['ymin']  = ymin
    MAP['xmax']  = xmax
    MAP['ymax']  = ymax
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res']))
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res']))
    MAP['map']   = np.zeros((MAP['sizex'],MAP['sizey']))
    return MAP

def mapping(grid, data, pose, res, angles):
    '''
        Mapping with the given data points and robot pose
        
        Input:
            grid  - N x N Map
            data  - data included lidar scan data and joint data
            pose  - best estimated robot pose (x, y, theta)
            res   - resolution
            angles - lidar scan angles (from -135 ~ 135 degree)
        Outputs:
            grid  - constructed map
    '''
    free_odd = np.log(9)/4
    occu_odd = np.log(9)

    X, Y     = polar2cart(data['scan'], angles)  # polar coord -> cartesian coord
    scan     = transform(X, Y, data['joint'], pose)
    scan     = filtering(scan, pose)
    xi, yi   = (scan[0]/res).astype(int), (scan[1]/res).astype(int)

    for (a, b) in zip(xi, yi):
        line = bresenham2D(int(pose[0]/res), int(pose[1]/res), a, b).astype(np.int16)
        x    = a + grid.shape[0]//2  # offset to center
        y    = b + grid.shape[1]//2  # offset to center
        grid[x, y] += occu_odd
        grid[line[0] + grid.shape[0]//2, line[1] + grid.shape[1]//2] -= free_odd
        
    grid[grid >= 100]  = 100
    grid[grid <= -100] = -100
        
    return grid

def measurement_model_update(MAP, P, data, angles):
    '''
        measurement model update
        
        Input:
            MAP    - list of particle states
            P      - relative motion in (x, y, theta)
            data   - current scan data
            angles - lidar scan angles (from -135 ~ 135 degree)
        Outputs:
            best_particle - chosen best particle (x, y, theta)
    '''
    # calculate map correlation for each particle
    l = 2
    corrs = []
    res = MAP['res']
    particles = P['states']

    grid_tmp = np.zeros_like(MAP['map'])  # for calculate correlation
    grid_tmp[MAP['map'] > 0] = 1          # occupied
    grid_tmp[MAP['map'] < 0] = 0          # free

    X, Y  = polar2cart(data['scan'], angles)  # polar coord -> cartesian coord
    x_im, y_im = np.arange(MAP['xmin'], MAP['xmax']+res, res), np.arange(MAP['ymin'], MAP['ymax']+res, res)
    x_range, y_range = np.arange(-res*l, res*l+res, res), np.arange(-res*l, res*l+res, res)

    for i in range(len(particles)):
        scan = transform(X, Y, data['joint'], particles[i])
        scan = filtering(scan, particles[i])

        x, y = scan[0], scan[1]
        corr = mapCorrelation(grid_tmp, x_im, y_im, np.vstack((x,y)), particles[i][0]+x_range, particles[i][1]+y_range)
        corrs.append(np.max(corr))
    
    # get the particle with largest weight
    corrs = np.array(corrs)
    P['weight'] = softmax(P['weight'] * corrs)
    best_idx = np.where(P['weight']==np.max(P['weight']))[0][0]
    best_particle = particles[best_idx]

    return best_particle


# INPUT 
# im              the map 
# x_im,y_im       physical x,y positions of the grid map cells
# vp(0:2,:)       occupied x,y positions from range sensor (in physical unit)  
# xs,ys           physical x,y,positions you want to evaluate "correlation" 
#
# OUTPUT 
# c               sum of the cell values of all the positions hit by range sensor
def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))

    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                    np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr


# Bresenham's ray tracing algorithm in 2D.
# Inputs:
#	(sx, sy)	start point of ray
#	(ex, ey)	end point of ray
def bresenham2D(sx, sy, ex, ey):
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y))



### TEST FUNCTIONS ###
def test_mapCorrelation():
    dataIn = io.loadmat("lidar/train_lidar0.mat")
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges = np.double(dataIn['lidar'][0][110]['scan'][0][0]).T

    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -20  #meters
    MAP['ymin']  = -20
    MAP['xmax']  =  20
    MAP['ymax']  =  20 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])

    # convert position in the map frame here 
    Y = np.concatenate([np.concatenate([xs0,ys0],axis=0),np.zeros(xs0.shape)],axis=0)

    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[0][indGood[0]],yis[0][indGood[0]]]=1

    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)

    print("Testing map_correlation...")
    c = mapCorrelation(MAP['map'],x_im,y_im,Y[0:3,:],x_range,y_range)

    c_ex = np.array([[3,4,8,162,270,132,18,1,0],
            [25  ,1   ,8   ,201  ,307 ,109 ,5  ,1   ,3],
            [314 ,198 ,91  ,263  ,366 ,73  ,5  ,6   ,6],
            [130 ,267 ,360 ,660  ,606 ,87  ,17 ,15  ,9],
            [17  ,28  ,95  ,618  ,668 ,370 ,271,136 ,30],
            [9   ,10  ,64  ,404  ,229 ,90  ,205,308 ,323],
            [5   ,16  ,101 ,360  ,152 ,5   ,1  ,24  ,102],
            [7   ,30  ,131 ,309  ,105 ,8   ,4  ,4   ,2],
            [16  ,55  ,138 ,274  ,75  ,11  ,6  ,6   ,3]])
            
    if np.sum(c==c_ex) == np.size(c_ex):
        print("...Test passed.")
    else:
        print("...Test failed. Close figures to continue tests.")	


    fig = plt.figure(figsize=(18,6))

    #plot original lidar points
    ax1 = fig.add_subplot(131)
    plt.plot(xs0,ys0,'.k')
    plt.scatter(0,0,s=30,c='r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Laser reading (red being robot location)")
    plt.axis('equal')

    #plot map
    ax2 = fig.add_subplot(132)
    plt.imshow(MAP['map'],cmap="hot")
    plt.title("Occupancy map")

    #plot correlation
    ax3 = fig.add_subplot(133,projection='3d')
    X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
    ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
    plt.title("Correlation coefficient map")

    plt.show()
  
def test_bresenham2D():
    sx = 0
    sy = 1
    print("Testing bresenham2D...")
    r1 = bresenham2D(sx, sy, 10, 5)
    r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    plt.scatter([sx,10],[sy,5],s=750,c='r',marker="s")
    plt.title("Given start and end point")
    plt.axis('equal')
    ax2 = fig.add_subplot(122)
    plt.scatter(r1_ex[0],r1_ex[1],s=750,c='b',marker="s")
    plt.title("bresenham2D return all the integer coordinates in-between")
    plt.axis('equal')
    plt.show()

    r2 = bresenham2D(sx, sy, 9, 6)
    r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    plt.scatter([sx,9],[sy,6],s=750,c='r',marker="s")
    plt.title("Given start and end point")
    plt.axis('equal')
    ax2 = fig.add_subplot(122)
    plt.scatter(r2_ex[0],r2_ex[1],s=750,c='b',marker="s")
    plt.title("bresenham2D return all the integer coordinates in-between")
    plt.axis('equal')
    plt.show()

    if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
        print("...Test passed.")
    else:
        print("...Test failed.")

    # Timing for 1000 random rays
    num_rep = 1000
    start_time = time.time()
    for i in range(0,num_rep):
        x, y = bresenham2D(sx, sy, 500, 200)
    print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))
  


if __name__ == "__main__":
    test_mapCorrelation()
    test_bresenham2D()
