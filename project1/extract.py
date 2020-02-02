import pickle
import numpy
import matplotlib.pyplot as plt
import cv2
dic={"stopsign":1,"other_red":2,"green":3,"blue":4,"yellow":5,"gray":6,"white":7}   #define seven different color classes
color_class=["stopsign","other_red","green","blue","yellow","gray","white"]
file_path = './trainset/'
class Kary():
	W = numpy.zeros((7,3)) #initialize the weight matrix 
	loss = 1000000000  #initialize the loss value and update it during optimization
	def update(self,update): 
		self.W += update
	def loss_f(self,loss_x,loss_y):
		forward_matrix = numpy.dot(self.W, loss_x)
		forward_matrix = forward_matrix.T
		max_index = forward_matrix.argmax(axis=1)
		self.loss=sum(max_index==loss_y)
		print(self.loss) 
	def swap(self):  #the weight matrix ordered in rgb, the input image from OpenCV is ordered in bgr
		self.W[:,[0,2]] = self.W[:,[2,0]]
with open('mask.pickle','rb') as f:
	a = pickle.load(f)
	alpha = 0.00001 #parameter for k-ary logistic regression
	y_matrix = numpy.zeros((1,7))
	x_matrix = numpy.zeros((1,3))
	classifier = Kary()
	iteration = 50  #maximum iteration times
	for index in range(88,100):  #training image set
		print(index)
		image_name = str(index)
		image_name += '.jpg'
		try:
			mask = a[0][image_name]
		except KeyError:
			print(image_name+" is invalid! Process next image")
		else:
			image_name = file_path + image_name
			for i in range(0,7):
				color = color_class[i]
				roi = plt.imread(image_name)[mask == dic[color]] #record the all the pixel values corresponging with each class
				roi = roi / 255 #normalization
				tmp = numpy.zeros((len(roi),7))
				tmp[:,i] = 1
				x_matrix = numpy.vstack((x_matrix,roi))
				y_matrix = numpy.vstack((y_matrix,tmp))
	x_matrix = x_matrix[1:]  #remove the zeros(1,3)
	y_matrix = y_matrix[1:]	 #remove the zeros(1,7)
	index_list = numpy.arange(len(x_matrix))
	numpy.random.shuffle(index_list) #shuffle the list for optimization
	x_matrix = x_matrix[index_list,:]
	y_matrix = y_matrix[index_list,:]
	x_matrix = x_matrix.T
	loss_x_matrix = x_matrix[:,0:10000]  #compute the loss using the first 10000 pixels
	loss_y_matrix = y_matrix[0:10000,:]
	loss_y_matrix = loss_y_matrix.argmax(axis=1)
	for i in range(0,iteration): #optimization
		tmp = numpy.dot(classifier.W,x_matrix)
		tmp = numpy.exp(tmp)
		tmp_sum = numpy.sum(tmp,axis = 0)
		tmp_sum = tmp_sum.T
		solution = tmp / tmp_sum
		solution = solution.T
		y_matrix_tmp = y_matrix - solution
		y_matrix_tmp = y_matrix_tmp.T
		x_matrix_tmp = x_matrix.T
		update_matrix = numpy.dot(y_matrix_tmp,x_matrix_tmp)
		update_matrix *= alpha   #update matrix
		classifier.update(update_matrix)
		print(classifier.W)
		classifier.loss_f(loss_x_matrix,loss_y_matrix) #compute the loss 			
classifier.swap() #rgb -> bgr
img_index = 1
#test with image1 - image60
for z in range(0,60):
	test_image = file_path + str(img_index) + ".jpg"
	img = cv2.imread(test_image, 1)
	if img is None: #check the image
		test_image = file_path + str(img_index) + ".png"
		img = cv2.imread(test_image, 1)
		if img is None:
			img_index += 1
			test_image = file_path + str(img_index) + ".jpg"
			img = cv2.imread(test_image, 1)
	height, width, channels = img.shape
	img = cv2.resize(img, (600,480))  #resize the image to (600,480)
	height, width, channels = img.shape
	cv2.imshow('image',img)	
	img_new = numpy.zeros((height,width,channels),numpy.uint8)
	img_new.fill(0)
	cv2.waitKey(3000)
	img.reshape(height*width,channels)
	for index in range(0,height):
		img_tmp = img[index].T
		img_choice = numpy.dot(classifier.W,img_tmp)  
		max_index = numpy.argmax(img_choice,axis=0) #classify each pixel
		for i in range(0,len(max_index)):  
			if(max_index[i] == 0):  
				img_new[index,i] = [255,255,255]   #if it is classified as stop sign(0), set the value as [255,255,255]
	cv2.imshow('new_image',img_new)
	cv2.waitKey(3000)
	img_index += 1
	cv2.destroyAllWindows()
		
	
				

				
				
			
					
				
