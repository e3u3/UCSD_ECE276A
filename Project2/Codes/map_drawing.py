"""""""""""""""""""""""""""""""""""""""""""""
DESCRIPTION
: In this file, you can load .mat file data in python dictionary format.
  The output of the "get_joint" function is a dictionary with eight different data (read data description for details).
  Each dictionary is an array with the same length.
  The "get_joint_index" function returns joint ID number.
  The output of the "get_lidar" function is an array with dictionary elements. The length of the array is the length of data.   
  The output of the "get_rgb" function is an array with dictionary elements. The length of the array is the length of data.
  The output of the "get_depth" function is an array with dictionary elements. The length of the array is the lenght of data.
	The replay_* functions help you to visualize and understand the lidar, depth, and rgb data. 
"""""""""""""""""""""""""""""""""""""""""""""

#import pickle
from scipy import io
from p2_utils import bresenham2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math

def get_joint(file_name):
	key_names_joint = ['ts', 'head_angles']
	data = io.loadmat(file_name+".mat")
	joint = {kn: data[kn] for kn in key_names_joint}
	return joint

def get_lidar(file_name):
	data = io.loadmat(file_name+".mat")
	lidar = []
	for m in data['lidar'][0]:
		tmp = {}
		tmp['t']= m[0][0][0]
		nn = len(m[0][0])
		if (nn != 3):
			raise ValueError("different length!")
		tmp['delta_pose'] = m[0][0][nn-1]
		tmp['scan'] = m[0][0][nn-2]
		lidar.append(tmp)
	return lidar

def replay_lidar(lidar_data):
	# lidar_data type: array where each array is a dictionary with a form of 't','pose','res','rpy','scan'
	theta = np.arange(0,270.25,0.25)*np.pi/float(180)

	for i in range(0,len(lidar_data),1):
		for (k,v) in enumerate(lidar_data[i]['scan'][0]):
			if v > 30: #the maximum distance is 30
				lidar_data[i]['scan'][0][k] = 0.0

	ax = plt.subplot(111, projection='polar')
	ax.plot(theta, lidar_data[i]['scan'][0])
	ax.set_rmax(10)
	ax.set_rticks([2,4])  # less radial ticks
	ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
	ax.grid(True)
	ax.set_title("Lidar scan data", va='bottom')
	plt.show()

def get_rgb(folder_name):
    n_rgb = len(os.listdir(folder_name))-1
    rgb = []
    time_file = open(os.path.join(folder_name,"timestamp.txt"))
    for i in range(n_rgb):
        rgb_img = cv2.imread(os.path.join(folder_name,"%d.jpg"%(i+1)))
        rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)
        time = time_file.readline().split()
        rgb_dict = {'image':rgb_img,'t':float(time[1])}
        rgb.append(rgb_dict)
    return rgb

def replay_rgb(rgb_data):
	for k in range(len(rgb_data)):
		R = rgb_data[k]['image']
		R = np.flip(R,1)
		plt.imshow(R)
		plt.draw()
		plt.pause(0.01)

def get_depth(folder_name):
    n_depth = len(os.listdir(folder_name))-1
    depth = []
    time_file = open(os.path.join(folder_name,"timestamp.txt"))
    for i in range(n_depth):
        depth_img = cv2.imread(os.path.join(folder_name,"%d.png"%(i+1)),-1)
        time = time_file.readline().split()
        depth_dict = {'depth':depth_img,'t':float(time[1])}
        depth.append(depth_dict)
    return depth

def replay_depth(depth_data):
	DEPTH_MAX = 4500
	DEPTH_MIN = 400
	for k in range(len(depth_data)):
		D = depth_data[k]['depth']
		D = np.flip(D,1)
		for r in range(len(D)):
			for (c,v) in enumerate(D[r]):
				if (v<=DEPTH_MIN) or (v>=DEPTH_MAX):
					D[r][c] = 0.0
		plt.imshow(D)
		plt.draw()
		plt.pause(0.01)

def getExtrinsics_IR_RGB():
  # The following define a transformation from the IR to the RGB frame:
    rgb_R_ir = np.array( [
      [0.99996855100876,0.00589981445095168,0.00529992291318184],
      [-0.00589406393353581,0.999982024861347,-0.00109998388535087],
      [-0.00530631734715523,0.00106871120747419,0.999985350318977]])
    rgb_T_ir = np.array([0.0522682,0.0015192,-0.0006059]) # meters
    return {'rgb_R_ir':rgb_R_ir, 'rgb_T_ir':rgb_T_ir}

def getIRCalib():
    '''For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/'''
    #-- Focal length:
    fc = np.array([364.457362485643273,364.542810626989194])
    #-- Principal point:
    cc = np.array([258.422487561914693,202.487139940005989])
    #-- Skew coefficient:
    alpha_c = 0.000000000000000
    #-- Distortion coefficients:
    kc = np.array([0.098069182739161,-0.249308515140031,0.000500420465085,0.000529487524259,0.000000000000000])
    #-- Focal length uncertainty:
    fc_error = np.array([1.569282671152671 , 1.461154863082004 ])
    #-- Principal point uncertainty:
    cc_error = np.array([2.286222691982841 , 1.902443125481905 ])
    #-- Skew coefficient uncertainty:
    alpha_c_error = 0.000000000000000
    #-- Distortion coefficients uncertainty:
    kc_error = np.array([0.012730833002324 , 0.038827084194026 , 0.001933599829770 , 0.002380503971426 , 0.000000000000000 ])
    #-- Image size: nx x ny
    nxy = np.array([512,424])
    return {'fc':fc, 'cc':cc, 'ac':alpha_c, 'kc':kc, 'nxy':nxy,
            'fce':fc_error, 'cce':cc_error, 'ace':alpha_c_error, 'kce':kc_error}

def getRGBCalib():
    '''For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/'''
    #-- Focal length:
    fc = np.array([525,525.5])
    #-- Principal point:
    cc = np.array([524.5,267])
    #-- Skew coefficient:
    alpha_c = 0.000000000000000
    #-- Distortion coefficients:
    kc = np.array([0.026147836868708 , -0.008281285819487 , -0.000157005204226 , 0.000147699131841 , 0.000000000000000])
    #-- Focal length uncertainty:
    fc_error = np.array([2.164397369394806 , 2.020071561303139 ])
    #-- Principal point uncertainty:
    cc_error = np.array([3.314956924207777 , 2.697606587350414 ])
    #-- Skew coefficient uncertainty:
    alpha_c_error = 0.000000000000000
    #-- Distortion coefficients uncertainty:
    kc_error = np.array([0.005403085916854 , 0.015403918092499 , 0.000950699224682 , 0.001181943171574 , 0.000000000000000 ])
    #-- Image size: nx x ny
    nxy = np.array([960,540])
    return {'fc':fc, 'cc':cc, 'ac':alpha_c, 'kc':kc, 'nxy':nxy,
            'fce':fc_error, 'cce':cc_error, 'ace':alpha_c_error, 'kce':kc_error}
#def Registration(img,depth):
'''
Author: Yunhai Han
Function: compute rotation matrix
'''
def compute_rotation_matrix(wx,wy,wz):
    Rz = np.eye(3)
    Rz[0,0]=Rz[1,1]=np.cos(wz)
    Rz[0,1]=-np.sin(wz)
    Rz[1,0]=-Rz[0,1]
    Ry = np.eye(3)
    Ry[0, 0] = Ry[-1, -1] = np.cos(wy)
    Ry[0, -1] = -np.sin(wy)
    Ry[-1, 0] = -Ry[0, -1]
    Rx = np.eye(3)
    Rx[1,1]=Rx[2,2]=np.cos(wx)
    Rx[1,2]=-np.sin(wx)
    Rx[2,1]=-Rx[1,2]
    return Rz @ Ry @ Rx
'''
Author: Yunhai Han
Function: obtain time-synchronized Camera&motion
'''
def time_corres(motion,rgb):
    motion_list = []
    rgb_list = []
    mini = len(motion) #Always more data in motion
    maxi = len(rgb)
    index = 0
    r_tag = rgb[index]
    for i in range(0, mini):
        m_tag = motion[i]
        if(m_tag < r_tag and index == 0):
            continue;
        if m_tag > r_tag:
            rgb_list.append(index)
            motion_list.append(i)
            index += 1
            if(index > maxi - 1):
                break
            else:
                r_tag = rgb[index]
    return motion_list, rgb_list
'''
Author: Yunhai Han
Function: Draw the occupancy maps from map data
'''
def draw_map(occupancyMap, ProbaMap,new_map):
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131)
    plt.imshow(occupancyMap[::-1],cmap="hot")
    plt.title("Occupancy map")
    ax2 = fig.add_subplot(132)
    plt.imshow(ProbaMap[::-1])
    ax3 = fig.add_subplot(133)
    new_map[new_map < 10] = 0
    new_map[new_map >= 10] = 3
    new_map[occupancyMap == 1] = 1
    new_map[occupancyMap == 2] = 2
    plt.imshow(new_map[::-1], cmap="hot")
    plt.title("Texture map")
    plt.show()
'''
Author: Yunhai Han
Function: select the ground grids
'''
def coloring(Configuration,rgb,camera_y,position):
    xmin = Configuration['xmin']
    xmax = Configuration['xmax']
    ymin = Configuration['ymin']
    ymax = Configuration['ymax']
    resolution = Configuration['res']
    sizex = Configuration['sizex']
    sizey = Configuration['sizey']
    robot_x = position[0] * 1000 #m -> mm
    robot_y = position[1] * 1000
    robot_theta = position[2]
    neck_angle = position[3]
    head_angle = position[4]
    camera_x = np.zeros(camera_y.shape)
    camera_z = np.zeros(camera_y.shape)
    Calib_IR = getIRCalib()
    fx = Calib_IR['fc'][0]
    fy = Calib_IR['fc'][1]
    cx = Calib_IR['cc'][0]
    cy = Calib_IR['cc'][1]
    for i in range(0, len(depth)):
        for j in range(0, len(depth[i])):
            camera_x[i][j] = (j - cx) * camera_y[i][j] / fx
            camera_z[i][j] = (i - cy) * camera_y[i][j] / fy + 70
    coor_x = camera_x.reshape(1,camera_x.shape[0]*camera_x.shape[1]).squeeze()
    coor_z = camera_z.reshape(1,camera_z.shape[0]*camera_z.shape[1]).squeeze()
    coor_y = camera_y.reshape(1,camera_y.shape[0]*camera_y.shape[1]).squeeze()
    coor_c = np.array([coor_x,coor_y,coor_z])
    Rotation_matrix = compute_rotation_matrix(wx=0, wy=head_angle, wz=neck_angle)
    coor_h = Rotation_matrix.dot(coor_c)
    coor_h[2] += 330
    z_ground = -930  # contact points with the ground
    invalid = np.logical_and(coor_h[2] <= z_ground,coor_h[1] <= 6000)
    ground_index = np.where(invalid == True)
    Rotation_matrix = compute_rotation_matrix(wx=0, wy=0, wz=robot_theta)
    world_c = Rotation_matrix.dot(np.array([coor_h[0][invalid],coor_h[1][invalid],coor_h[2][invalid]]).squeeze())
    _position_in_map_x = np.ceil((robot_x - xmin) / resolution).astype(np.int16)  # x
    _position_in_map_y = np.ceil((robot_y - ymin) / resolution).astype(np.int16)  # y
    xis = np.ceil((world_c[0] - xmin) / resolution).astype(np.int16) - 1 + np.ceil(_position_in_map_x - (sizex - 1) / 2).astype(np.int16)
    yis = np.ceil((world_c[1] - ymin) / resolution).astype(np.int16) - 1 + np.ceil(_position_in_map_y - (sizey - 1) / 2).astype(np.int16)
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < sizex)),(yis < sizey))
    xis=xis[indGood]
    yis=yis[indGood]
    return xis,yis
if __name__ == "__main__":
    occupancyMap = np.load("L_occupancyMap3.npy")
    ProbaMap = np.load("L_ProbabilityMap3.npy")
    motion = np.load("motion_data3.npy")
    r0 = get_rgb("cam/RGB_3")  # for coloring the map
    xmin = -30000 #Configuration of map
    xmax = 30000
    ymin = -30000
    ymax = 30000
    resolution = 50
    sizex = int(np.ceil((xmax - xmin) / resolution + 1))  # cells
    sizey = int(np.ceil((ymax - ymin) / resolution + 1))
    Configuration = {"xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax,"res":resolution,"sizex":sizex,"sizey":sizey}
    new_map = np.zeros((Configuration['sizex'], Configuration['sizey']), dtype=np.int8)  # DATA TYPE: char or int8
    rgb_time = np.zeros([1, len(r0)])
    for i in range(0, len(r0)):
        rgb_time[0, i] = r0[i]['t']
    rgb_time = rgb_time[0]
    d0 = get_depth("cam/DEPTH_3")  # for coloring the map
    #The depth image and the rgb have the same time tag.
    motion_time = motion.T[0]
    M_tag, R_tag = time_corres(motion_time,rgb_time) #pick up the closest time tag
    print("correlation Done!")
    motion_Synchro = motion[M_tag].squeeze()
    index = 0
    for position in motion_Synchro:
        position = position[1:6]
        img = r0[index]['image']
        depth = d0[index]['depth']
        index += 1
        xis,yis = coloring(Configuration,img,depth,position)
        new_map[yis,xis] += 1
        print("This is the ",str(index), " image in total ", str(len(motion_Synchro)))
    draw_map(occupancyMap,ProbaMap,new_map)




