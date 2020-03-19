import numpy as np
from utils import *
from tqdm import tqdm
from scipy.linalg import expm
def hat_so(x):
    if x.shape == (3,1):
        return np.array([
            [0    , -x[2][0],  x[1][0]],
            [x[2][0] ,     0, -x[0][0]],
            [-x[1][0],  x[0][0],     0]
        ])
    else:
        return np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ])
def hat_se(x):
    velosity = x[:3]
    omega = x[3:]

    return np.block([
        [hat_so(omega), velosity.reshape((3,1))],
        [np.zeros((1,4))                  ]
    ])
def ren_se(x):
    velosity = x[:3]
    omega = x[3:]

    return np.block([
        [hat_so(omega)   , hat_so(velosity)],
        [np.zeros((3, 3)), hat_so(omega)]
    ])
def circle_se(x):
    s = x[0:3]
    return np.block([
        [np.eye(3) , -hat_so(s)],
        [np.zeros((1, 6))]
    ])
def get_inverse(T):
    R = T[:3, :3] # (3, 3)
    p = T[:3, -1] # (3, )
    return np.block([
        [R.T         , -R.T @ p.reshape((3, 1))],
        [np.array([0, 0, 0, 1]).reshape((1, 4))]
    ])
def steoro_in(K,b):
    return np.block([
        [K[0][0], 0, K[0][2], 0],
        [0, K[1][1], K[1][2], 0],
        [K[0][0], 0, K[0][2], -K[0][0] * b],
        [0, K[1][1], K[1][2], 0],
    ])
def camera_T_world(data, configura, cam_T_imu, world_T_imu):
    baseline = configura["baseline"]
    fx = configura["fx"]
    fy = configura["fy"]
    cx = configura["cx"]
    cy = configura["cy"]
    Z = fx * b / (data[0] - data[2])
    X = (data[0] - cx) * Z / fx
    Y = (data[1] - cy) * Z / fy
    coor = np.array([X,Y,Z,1])
    world_coor = get_inverse(world_T_imu) @ get_inverse(cam_T_imu) @ coor
    return world_coor[0:3]
def map(map_mean):
    I = np.zeros((2,int(len(map_mean)/3)))
    for i in range(0,int(len(map_mean)/3)):
        I[0][i] = map_mean[3*i]
        I[1][i] = map_mean[3*i + 1]
    return I
def world_T_camera(vector,mean,cam_T_imn,steoro_intrinsic):
    m_i = np.zeros([4,1])
    m_i[0] = vector[0]
    m_i[1] = vector[1]
    m_i[2] = vector[2]
    m_i[3] = 1
    tmp = cam_T_imn @ mean @ m_i
    tmp[0] /= tmp[2]
    tmp[1] /= tmp[2]
    tmp[3] /= tmp[2]
    tmp[2] /= tmp[2]
    zxc = steoro_intrinsic @  tmp
    zxc = np.squeeze(zxc)
    return zxc
def compute_der(tmp):
    return 1/tmp[2] * np.block([
        [1, 0,-tmp[0]/tmp[2], 0],
        [0, 1,-tmp[1]/tmp[2], 0],
        [0, 0, 0, 0],
        [0, 0, -tmp[3]/tmp[2], 1],
    ])
def compute_H(vector,mean,cam_T_imn,steoro_intrinsic):
    P = np.concatenate((np.eye(3),np.zeros([3,1])),axis=1).T
    right_matrix = cam_T_imn @ mean @ P
    m_i = np.zeros([4,1])
    m_i[0] = vector[0]
    m_i[1] = vector[1]
    m_i[2] = vector[2]
    m_i[3] = 1
    tmp = cam_T_imn @ mean @ m_i
    deriv = compute_der(tmp)
    return steoro_intrinsic @ deriv @ right_matrix
def compute_H_(vector,mean,cam_T_imn,steoro_intrinsic):
    m_i = np.zeros([4, 1])
    m_i[0] = vector[0]
    m_i[1] = vector[1]
    m_i[2] = vector[2]
    m_i[3] = 1
    tmp = circle_se(mean @ m_i)
    right_matrix = cam_T_imn @ tmp
    middle_matrix = compute_der(cam_T_imn @ mean @ m_i)
    return steoro_intrinsic @ middle_matrix @ right_matrix
if __name__ == '__main__':
    filename = "./data/0022.npz"
    mean = np.eye(4)
    conva = np.zeros(6)
    motion_noise = 0.5 * np.eye(6)
    measurement_noise_con = 4
    t,features,linear_velosity,rotational_velosity,K,b,cam_T_imn = load_data(filename)
    u1 = features[0] #u1[0] wrt time
    u1_t = u1.T
    v1 = features[1]
    u2 = features[2]
    v2 = features[3]
    steoro_intrinsic = steoro_in(K,b)
    iter = len(t[0])
    trajectory = np.zeros((4,4,iter))
    trajectory[:,:,0] = get_inverse(mean)
    map_mean = np.zeros(3 * features.shape[1]) #mean_value for the map
    map_conva = np.eye(3 * features.shape[1]) * 1000 #the initial convariance matrix for the map, set as 1000
    map_unseen = np.zeros(features.shape[1])
    configura = {"baseline":b,"fx":K[0][0],"fy":K[1][1],"cx":K[0][2],"cy":K[1][2]}
    # (a) IMU Localization via EKF Prediction
    for i in tqdm(range(0,iter - 1)):
        time = t[0][i+1] - t[0][i]
        velosity = np.array([linear_velosity[0][i],linear_velosity[1][i],linear_velosity[2][i]])
        omega = np.array([rotational_velosity[0][i], rotational_velosity[1][i], rotational_velosity[2][i]])
        u_t = np.concatenate([velosity,omega]).reshape(6,1)
        hat_u_t = hat_se(u_t)
        ren_u_t = ren_se(u_t)
        mean = expm(-time * hat_u_t) @ mean
        conva = expm(-time * ren_u_t) @ conva @ expm(-time * ren_u_t).T + motion_noise
        u1T = u1_t[i]
        index = np.where(u1T != -1)[0] #all the detected landmarks
        min_i = index[0]
        max_i = index[-1]
        index_ = index - min_i
        zt =  np.zeros([4*len(index)])
        hat_zt = np.zeros([4*len(index)])
        H = np.zeros([4*len(index),3*(max_i-min_i+1)])
        H_ = np.zeros([4*len(index),6])
        for j in range(0,len(index)): #for all features
            coor = np.array([u1[index[j]][i], v1[index[j]][i], u2[index[j]][i], v2[index[j]][i]]) #each feature point's camera coordinate
            zt[4*j:4*(j+1)] = coor
            if map_unseen[index[j]] == 0: #first time for this feature point
                wor_coor = camera_T_world(coor,configura,cam_T_imn,mean)
                map_mean[3*index[j]:3*(index[j]+1)] = wor_coor
                map_unseen[index[j]] = 1
            hat_zt[4*j:4*(j+1)] = world_T_camera(map_mean[3*index[j]:3*(index[j]+1)],mean,cam_T_imn,steoro_intrinsic)
            H[4*j:4*(j+1),3*index_[j]:3*(index_[j]+1)] = compute_H(map_mean[3*index[j]:3*(index[j]+1)],mean,cam_T_imn,steoro_intrinsic)
            H_[4*j:4*(j+1),0:6] = compute_H_(map_mean[3*index[j]:3*(index[j]+1)],mean,cam_T_imn,steoro_intrinsic)
        tmp_mean = map_mean[3*min_i:3*(max_i+1)]
        tmp_conva = map_conva[3*min_i:3*(max_i+1),3*min_i:3*(max_i+1)]
        measurement_noise = np.eye(4*len(index)) * measurement_noise_con
        K_ = conva @ H_.T @ np.linalg.inv(H_ @ conva @ H_.T + measurement_noise)
        mean = expm(hat_se(K_ @ (zt - hat_zt))) @ mean
        conva = (np.eye(6) - K_ @ H_) @ conva
        K = tmp_conva @ H.T @ np.linalg.inv(H @ tmp_conva @ H.T + measurement_noise)
        tmp_mean = tmp_mean + K @ (zt - hat_zt)
        tmp_conva = (np.eye(3*(max_i-min_i+1)) - K @ H) * tmp_conva
        map_mean[3 * min_i:3 * (max_i + 1)] = tmp_mean
        map_conva[3 * min_i:3 * (max_i + 1), 3 * min_i:3 * (max_i + 1)] = tmp_conva
        trajectory[:, :, i + 1] = get_inverse(mean)
    print(conva)
    new_map = map(map_mean)
    visualize_trajectory_2d(new_map,trajectory, show_ori=True)
	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
