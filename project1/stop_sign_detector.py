'''
ECE276A WI20 HW1
Stop Sign Detector
'''
import os, cv2
import numpy as np
from skimage.measure import label, regionprops
class StopSignDetector():
	def __init__(self):
		self.W = np.array([(  7.97998502,-5.52467351,-3.12123777),
 				( -2.29260393,-2.99410614,-2.45656097),
				(  2.64526509,11.27600299,-9.97913918),
				(-12.05650543,-0.04008605,18.02165188),
				(  0.90202575,-1.19887349,-2.44773507),
				(  2.19011698,-1.6565441,-0.59845608),
				(  0.63171651,0.1382803,0.58147718)])  #weight matrix obtained from the training data
		self.W[:,[0,2]] = self.W[:,[2,0]] #rgb->bgr
	def segment_image(self, img):  #segment the image
		height, width, channels = img.shape
		img_new = np.zeros((height,width,3),np.uint8)
		img_new.fill(0)
		img.reshape(height*width,channels)
		for index in range(0,height):
                	img_tmp = img[index].T
                	img_choice = np.dot(self.W,img_tmp)
                	max_index = np.argmax(img_choice,axis=0)
                	for i in range(0,len(max_index)):
                        	if(max_index[i] == 0):
                        	        img_new[index,i] = [255,255,255]
		return img_new
	def geometry_filter(self,img,x,y,w,h,width,height): #filter the stop sign candidates by geometric property
		if(x<=3 or x+w >= width-3): 
			return 2
		if(y<=3 or y+h >= height-3):
			return 2
		ratio = w/h
		if(ratio >= 0.6 and ratio <= 1.4):
			x_ = [0,0,0,0]
			y_ = [0,0,0,0]
			for c_ in range(x,x+w):
				if(img[y+2,c_] == 0):
					x_[0] += 1
				else:
					break
			for r_ in range(y,y+h):
				if(img[r_,x+2] == 0):
					y_[0] += 1
				else:
					break
			for c_ in range(x+w,x,-1):
				if(img[y+2,c_] == 0):
					x_[1] += 1
				else:
					break
			for r_ in range(y,y+h):
				if(img[r_,x+w-2] == 0):
					y_[1] += 1
				else:
					break
			for c_ in range(x,x+w):
				if(img[y+h-2,c_] == 0):
					x_[2] += 1
				else:
					break
			for r_ in range(y+h,y,-1):
				if(img[r_,x+2] == 0):
					y_[2] += 1
				else:
					break
			for c_ in range(x+w,x,-1):
				if(img[y+h-2,c_] == 0):
					x_[3] += 1
				else:
					break
			for r_ in range(y+h,y,-1):
				if(img[r_,x+w-2] == 0):
					y_[3] +=1
				else:
					break
			x_sum = sum(x_)
			y_sum = sum(y_)
			ratio_min = min(x_sum,y_sum)
			ratio_max = max(x_sum,y_sum)
			if(ratio_min > 0):
				ratio_ = ratio_max / ratio_min
				return ratio_
		return 2
	def get_bounding_box(self, img): #get the bounding box
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret,img = cv2.threshold(img,122,255,0) #convert the image into the binary image
		contours,hierarchy = cv2. findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		boxes = []
		box = []
		height_ori,width_ori = img.shape
		height = 480
		width = 600	
		img = cv2.resize(img, (width, height))  #resize the image to (480.600)
		for i in contours:
			area = cv2.contourArea(i)
			if(area > 700): #filter some random noise by the area(prior knowledge:the stop sign should not be too small)
				boxes.append(i)
		boxes = sorted(boxes,key=lambda x:cv2.contourArea(x))
		if(len(boxes) > 0): 
			for i in range(0,len(boxes)):
				x,y,w,h = cv2.boundingRect(boxes[i])
				ratio_ = self.geometry_filter(img,x,y,w,h,width,height) #filter the left boungding box by the geometric property(prior knowledge:the stop sign should be octagon)
				if(ratio_ <= 1.5): #if the geometric requirement is satisfied:
					box_i = cv2.boundingRect(boxes[i])
					box_l = []
					for i in box_i:
						box_l.append(i)
					ratio_width = width_ori / width #convert the coordinates into the original image coordinate
					box_l[0] = box_l[0] * ratio_width
					box_l[2] = box_l[2] * ratio_width
					ratio_height = height_ori / height
					box_l[1] = box_l[1] * ratio_height
					box_l[3] = box_l[3] * ratio_height
					tmp = []
					#compute the bottom_left point and the top_right point
					tmp.append(box_l[0])
					tmp.append(height_ori - box_l[1] - box_l[3]) 
					tmp.append(box_l[0] + box_l[2])
					tmp.append(height_ori - box_l[1])
					box.append(box_l)
		if(len(box) > 0):
			boxes = []
			boxes.append(box[-1]) #select the largest one				
			return boxes
		else:
			boxes = []
			return boxes
if __name__ == '__main__':
	folder = "trainset"
	my_detector = StopSignDetector()
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		mask_img = my_detector.segment_image(img)
		boxes = my_detector.get_bounding_box(mask_img)
		print(boxes)
