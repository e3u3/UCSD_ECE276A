from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math


'''
Author: Yunhai Han
Function: Draw the occupancy maps from map data
'''
def draw_map():
    occupancyMap = np.load("L_occupancyMap0.npy")
    ProbaMap = np.load("L_ProbabilityMap0.npy")
    motion = np.load("motion_data0.npy")
    print(motion)
    fig = plt.figure(figsize=(18, 6))
    ax2 = fig.add_subplot(131)
    plt.imshow(occupancyMap[::-1],cmap="hot")
    plt.title("Occupancy map")
    ax3 = fig.add_subplot(132)
    plt.imshow(ProbaMap[::-1])
    plt.show()

if __name__ == "__main__":
  draw_map()
