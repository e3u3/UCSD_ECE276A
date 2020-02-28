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
'''
Author: Yunhai Han
Function: obtain time-synchronized Lidar&Joint data
'''
def time_corres(l0,j0):
    Lidar_data = []
    Joint_data = []
    #print(len(j0['head_angles'][0]))  0-1 neck_angle or head_angle
    mini = min(len(j0['ts'][0]), len(l0)) #Always much more data in Joint_data
    maxi = max(len(j0['ts'][0]), len(l0))
    index = 0
    for i in range(0, mini):
        lidar_tag = l0[i]['t'][0][0]
        Joint_tag = j0['ts'][0][index]
        if(lidar_tag < Joint_tag and index == 0):
            continue;
        while lidar_tag > Joint_tag:
            index += 1
            if(index > maxi - 1):
                break
            else:
                Joint_tag = j0['ts'][0][index]
        if(index > maxi - 1):
            break
        else:
            Lidar_data.append({'ts':lidar_tag, 'delta_pose': l0[i]['delta_pose'], 'scan': l0[i]['scan']})
            Joint_data.append({'ts':Joint_tag, 'neck_angle':j0['head_angles'][0][index], 'head_angle':j0['head_angles'][1][index]})
    return Lidar_data, Joint_data
'''
Author: Yunhai Han
Function: particle filter
'''
class particle_filter():
    def __init__(self, number = 10, noise_x = 1, noise_y = 1, noise_theta = 1):
        self._particleNumber = number
        self._noiseX = noise_x
        self._noiseY = noise_y
        self._noiseT = noise_theta
        self._particle = np.zeros(shape=(self._particleNumber, 3))
        self._proba = np.array([1/self._particleNumber]*self._particleNumber) #weight for each particle
    def update(self, delta):
        #Assume the noise variables are independent
        noise_x = np.random.normal(0, self._noiseX, self._particleNumber) #creare random noise
        noise_y = np.random.normal(0, self._noiseY, self._particleNumber)
        noise_t = np.random.normal(0, self._noiseT, self._particleNumber)
        Noise_matrix = np.mat([noise_x,noise_y,noise_t])
        self._particle += delta[0]  #update particle_position
        #print("particle:", self._particle)
        self._particle += Noise_matrix.T #add random noise
    def Prob_particle(self): #obtain the position of the particle with the highest probability
        index = np.where(self._proba == np.max(self._proba, axis=0))[0][0]
        return self._particle[index]
'''
Author: Yunhai Han
Function: Map_configuration
'''
class Map:
    def __init__(self, Lidar_data, Joint_data, value = 0, threshold_1 = 0, threshold_2 = 0, satur_u = 0, satur_l = 0):
        self.__iteration = value
        self.__MAP = {}
        self.__Lidar = Lidar_data
        self.__Joint = Joint_data
        self._particles = particle_filter(1,0,0,0)
        self.__threshold_o = math.log(threshold_1)
        self.__threshold_f = math.log(threshold_2)
        self.__saturu = satur_u
        self.__saturl = satur_l
        self.__length = len(Lidar_data)
    def Map_init(self, size, resolution):
        self.__MAP['res'] = resolution  # meters
        self.__MAP['xmin'] = -size  # meters
        self.__MAP['ymin'] = -size
        self.__MAP['xmax'] = size
        self.__MAP['ymax'] = size
        self.__MAP['sizex'] = int(np.ceil((self.__MAP['xmax'] - self.__MAP['xmin']) / self.__MAP['res'] + 1))  # cells
        self.__MAP['sizey'] = int(np.ceil((self.__MAP['ymax'] - self.__MAP['ymin']) / self.__MAP['res'] + 1))
        self.__MAP['map'] = np.zeros((self.__MAP['sizex'], self.__MAP['sizey']), dtype=np.int8)  # DATA TYPE: char or int8
        self.__MAP['Probability'] =  np.zeros((self.__MAP['sizex'], self.__MAP['sizey']), dtype=np.float64)  # DATA TYPE: char or int8
    def load_data(self):
        self._neck = self.__Joint[self.__iteration]['neck_angle']
        self._head = self.__Joint[self.__iteration]['head_angle']
        self._lidar = self.__Lidar[self.__iteration]
    def update_par(self):
        self._particles.update(self._lidar['delta_pose'])
        self.__iteration += 1
        print("This is the ", self.__iteration, "th step in total ", str(self.__length))
    '''
    Author: Yunhai Han
    Function: transform from lidar-frame to world-frame
    '''
    def compute_rotation_matrix(self,wx,wy,wz):
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
    def Transform(self, z_min,z_max):
        self.load_data()
        self._degrees = np.linspace(-135, 135.25, 1081)*np.pi/180.
        indValid = np.logical_and((self._lidar['scan'][0] < z_max), (self._lidar['scan'][0] > z_min))
        self._lidar['scan'] = [self._lidar['scan'][0][indValid]]
        self._degrees = self._degrees[indValid]
        xs0 = np.array([self._lidar['scan'] * np.cos(self._degrees)])
        ys0 = np.array([self._lidar['scan'] * np.sin(self._degrees)])
        zs0 = np.array([0.15]*len(indValid))
        self._lidar['coordinate'] = np.array([xs0[0], ys0[0], zs0[0]])
        neck_angle = self._neck
        head_angle = self._head
        particle_position = self._particles.Prob_particle() #obtain the postive of the most probably particle
        self._position_in_map_x = np.ceil((particle_position[0] - self.__MAP['xmin']) / self.__MAP['res']).astype(np.int16) #x
        self._position_in_map_y = np.ceil((particle_position[1] - self.__MAP['ymin']) / self.__MAP['res']).astype(np.int16)#y
        angle_w2b = particle_position[2] #angle_body_to_world
        Rotation_matrix = self.compute_rotation_matrix(wx = 0, wy = head_angle, wz = neck_angle)
        self._lidar['coordinate_w'] = Rotation_matrix.dot(self._lidar['coordinate'])
        self._lidar['coordinate_w'][2] += 0.33
        Rotation_matrix = self.compute_rotation_matrix(wx = 0, wy = 0, wz = angle_w2b)
        self._lidar['coordinate_w'] = Rotation_matrix.dot(self._lidar['coordinate_w'])
        self._lidar['x'] = self._lidar['coordinate_w'][0]
        self._lidar['y'] = self._lidar['coordinate_w'][1]
        self._lidar['z'] = self._lidar['coordinate_w'][2]
        z_ground = -0.93 #contact points with the ground
        indValid = self._lidar['z'] > z_ground
        xs0 = self._lidar['x'][indValid]
        ys0 = self._lidar['y'][indValid]
        xis = np.ceil((xs0 - self.__MAP['xmin']) / self.__MAP['res']).astype(np.int16) - 1 + np.ceil(self._position_in_map_x - (self.__MAP['sizex'] - 1) / 2).astype(np.int16)
        yis = np.ceil((ys0 - self.__MAP['ymin']) / self.__MAP['res']).astype(np.int16) - 1 + np.ceil(self._position_in_map_y - (self.__MAP['sizey'] - 1) / 2).astype(np.int16)
        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.__MAP['sizex'])),(yis < self.__MAP['sizey']))
        for index in range(0, len(xis)):
            x_point = xis[index]
            y_point = yis[index]
            stack = bresenham2D(self._position_in_map_x, self._position_in_map_y, x_point, y_point).astype(np.int16)
            #stack[0] -> x stack[1] -> y  stack[0][-1] -> point on the boundary
            # saturate to [-10,10]
            if self.__MAP['Probability'][stack[1][-1], stack[0][-1]] <= self.__saturu:
                self.__MAP['Probability'][stack[1][-1], stack[0][-1]] += self.__threshold_o
            invalid = self.__MAP['Probability'][stack[1][0:-1], stack[0][0:-1]] >= self.__saturl
            self.__MAP['Probability'][stack[1][0:-1][invalid], stack[0][0:-1][invalid]] += self.__threshold_f
            #self.__MAP['Probability'][stack[1][0:-1], stack[0][0:-1]] += self.__threshold_f
        self.__MAP['map'][self.__MAP['Probability'] > 0] = 1
        self.__MAP['map'][self._position_in_map_y.astype(np.int16),self._position_in_map_x.astype(np.int16)] = 2
    '''
    Author: Yunhai Han
    Function: compare the first laser scan with the built map
    '''
    def draw_map(self):
         fig = plt.figure(figsize=(18, 6))
         ax2 = fig.add_subplot(131)
         self.__MAP['map'][self.__MAP['Probability']  > 0] = 1
         z_ground = -0.93  # contact points with the ground
         indValid = self._lidar['z'] > z_ground
         xs0 = self._lidar['x'][indValid]
         ys0 = self._lidar['y'][indValid]
         xis = np.ceil((xs0 - self.__MAP['xmin']) / self.__MAP['res']).astype(np.int16) - 1 + np.ceil(
            self._position_in_map_x - (self.__MAP['sizex'] - 1) / 2).astype(np.int16)
         yis = np.ceil((ys0 - self.__MAP['ymin']) / self.__MAP['res']).astype(np.int16) - 1 + np.ceil(
            self._position_in_map_y - (self.__MAP['sizey'] - 1) / 2).astype(np.int16)
         indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.__MAP['sizex'])),
                                 (yis < self.__MAP['sizey']))
         yis = yis[indGood]
         xis = xis[indGood]
         index = range(0, len(xis))
         for i in index:
            cv2.line(self.__MAP['map'], (self._position_in_map_x, self._position_in_map_y), (xis[i], yis[i]), 3, 1)
         self.__MAP['map'][self.__MAP['Probability'] > 0] = 1
         plt.imshow(self.__MAP['map'][::-1],cmap="hot")
         plt.title("Occupancy map")
         ax1 = fig.add_subplot(132)
         plt.plot(self._lidar['x'], self._lidar['y'], '.k')
         plt.scatter(0, 0, s=30, c='r')
         ax3 = fig.add_subplot(133)
         plt.imshow(self.__MAP['Probability'][::-1])
         plt.show()
    def save_map(self, data_i):
        np.save("L_occupancyMap"+str(data_i)+".npy", self.__MAP['map'])
        np.save("L_ProbabilityMap"+str(data_i)+".npy",self.__MAP['Probability'])
if __name__ == "__main__":
    X = 0
    Y = 0
    X_trajectory = [0]
    Y_trajectory = [0]
    theta = 0
    # It determines how much we trust the observation
    threshold_1 = 4  # occupied
    threshold_0 = 1/4  # free
    satur_u = 10 #saturation
    satur_l = -10
    for data_i in range(0, 5):
        joint_dire = "joint/train_joint" + str(data_i)
        j0 = get_joint(joint_dire)
        lidar_dire = "lidar/train_lidar" + str(data_i)
        l0 = get_lidar(lidar_dire)
        r0 = get_rgb("cam/RGB_"+ str(data_i))  # for coloring the map
        d0 = get_depth("cam/DEPTH_" + str(data_i))  # for coloring the map
        [Lidar_data, Joint_data] = time_corres(l0, j0)  # synchronized Lidar&Joint data
        map = Map(Lidar_data, Joint_data, 0, threshold_1, threshold_0, satur_u, satur_l)
        map.Map_init(30, 0.05) #max_size = 30 resolution = 0.05
        for i in range(0, len(Lidar_data)):
            map.Transform(0.1, 20)  # z_min to remove some bad measurements
            map.update_par()
            if i % 1000 == 0:
                map.save_map(data_i)
        map.save_map(data_i)
        map.draw_map()
    # visualize data
    #replay_lidar(l0[:50])
    #replay_rgb(r0[:200])
    #replay_depth(d0[:5])
