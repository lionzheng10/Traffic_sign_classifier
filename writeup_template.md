# Self-Driving Car Engineer Nanodegree

##Deep learning
##Project: Build a Traffic Sign Recognition Project

---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/1x.png "Traffic Sign 1"
[image5]: ./examples/2x.png "Traffic Sign 2"
[image6]: ./examples/3x.png "Traffic Sign 3"
[image7]: ./examples/5x.png "Traffic Sign 5"
[image8]: ./examples/6x.png "Traffic Sign 6"
[image9]: ./examples/sampleClasses.jpg "smple classes"
[image10]: ./examples/ClassesImage.jpg "Classes image"
[image11]: ./examples/grayscale2.jpg "grayscale2"
[image12]: ./examples/graph-run.png "graph-run"
[image13]: ./examples/8x.png "Traffic Sign 8"
[image14]: ./examples/9x.png "Traffic Sign 9"
[image15]: ./examples/10x.png "Traffic Sign 10"
[image16]: ./examples/analyzeImageFromWeb.png "analyzeImageFromWeb"
[image17]: ./examples/softmaxProbability.png "softmaxProbability"
[image18]: ./examples/conv1feature.png "conv1feature"
[image19]: ./examples/conv2feature.png "conv2feature"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
# Writeup / README

### Step0.Load the data.  
my code:
 
	# Load pickled data
	import pickle
	import numpy as np
	from tensorflow.examples.tutorials.mnist import input_data
	import random
	import numpy as np
	import tensorflow as tf
	import matplotlib.pyplot as plt
	import cv2
	%matplotlib inline
	
	# TODO: Fill this in based on where you saved the training and testing data
	training_file = 'dataset/train.p'
	validation_file = 'dataset/valid.p'
	testing_file = 'dataset/test.p'
	
	with open(training_file, mode='rb') as f:
	    train = pickle.load(f)
	with open(validation_file, mode='rb') as f:
	    valid = pickle.load(f)
	with open(testing_file, mode='rb') as f:
	    test = pickle.load(f)
	    
	X_train, y_train = train['features'], train['labels']
	X_valid, y_valid = valid['features'], valid['labels']
	X_test, y_test = test['features'], test['labels']
	
	print(X_train.shape, y_train.shape)
	print(X_valid.shape, y_valid.shape)
	print(X_test.shape, y_test.shape)

output

	(34799, 32, 32, 3) (34799,)
	(4410, 32, 32, 3) (4410,)
	(12630, 32, 32, 3) (12630,)

start tensorGraph

	sess = tf.Session()
	graph = tf.get_default_graph()
	graph.get_operations()
	operations = graph.get_operations()
	summary_writer = tf.summary.FileWriter('log_lenet_graph', sess.graph)

### Step1:Dataset Summary & Exploration
#### 1.Provide a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

	### Replace each question mark with the appropriate value. 
	### Use python, pandas or numpy methods rather than hard coding the results
	
	# TODO: Number of training examples
	n_train = X_train.shape[0]
	
	# TODO: Number of validation examples
	n_validation = X_valid.shape[0]
	
	# TODO: Number of testing examples.
	n_test = X_test.shape[0]
	
	# TODO: What's the shape of an traffic sign image?
	image_shape = X_train[0].shape
	
	# TODO: How many unique classes/labels there are in the dataset.
	n_classes = np.unique(y_train).size

	print("Number of training examples =", n_train)
	print("Number of validation examples =",n_validation)
	print("Number of testing examples =", n_test)
	print("Image data shape =", image_shape)
	print("Number of classes =", n_classes)

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43


#### 2.Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution by classes.I use np.bincount to count the lable frequency and plt.bar to show the bar graph.

	### Data exploration visualization goes here.
	### Feel free to use as many code cells as needed.
	import matplotlib.pyplot as plt
	#import random
	#import scipy.ndimage
	
	# Visualizations will be shown in the notebook.
	%matplotlib inline
	label_freq = np.bincount(y_train)
	max_inputs = np.max(label_freq)
	plt.bar(np.arange(n_classes), label_freq, align='center', alpha=0.4)
	
	plt.xlabel('# samples')
	plt.title('Sign count')
	plt.show()

output

![alt text][image9]

#### 3.I load signnames.csv into program, and display an sample image for every class, including sign name.  
my code:    

	# retrieve class labels 
	import csv
	with open('signnames.csv', 'r') as file:
	    reader = csv.reader(file)
	    class_names = dict(reader)
	
	# take a random sample per class
	sample_image_per_class = []
	for n in range(n_classes):
	    sample_image_per_class.append(np.random.choice(np.where(y_train==n)[0]))
	        
	show_images = X_train[sample_image_per_class,:,:,:]#show sign per class
	def show_signs(image_array, width, height):
	    fig = plt.figure(figsize=(12,20))
	    for j in range(image_array.shape[0]):
	        ax = fig.add_subplot(height, width, j+1)
	        ax.imshow(image_array[j], cmap='gray')
	        title = str(j) +': '+ class_names[str(j)]
	        plt.title(title) #add
	        plt.xticks(np.array([]))
	        plt.yticks(np.array([]))
	    plt.tight_layout()
	    
	show_signs(show_images, 3,15)

output:

![alt text][image10]

### Step 2: Design and Test a Model Architecture

#### 1.Describe how you preprocessed the image data. 
What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I shuffle the training set, define the EPOCHS and BATCH_SIZE.

	### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
	from sklearn.utils import shuffle
	X_train, y_train = shuffle(X_train, y_train)
	
	EPOCHS = 50
	BATCH_SIZE = 128
	### Feel free to use as many code cells as needed.

As second step, I decided to convert the images to grayscale because the Lenet5 need input shape 32x32x1. I use cv2.cvtColor to convert the images, after convert the shape is 32x32, so i add an newaxis to the data.

	#gray scale
	from copy import deepcopy
	
	X_train_gray = np.zeros((X_train.shape[0],32,32),dtype = float)
	X_valid_gray = np.zeros((X_valid.shape[0],32,32),dtype = float)
	X_test_gray = np.zeros((X_test.shape[0],32,32),dtype = float)
	
	for i in range(len(y_train)):
	    X_train_gray[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)
	    
	for i in range(len(y_valid)):
	    X_valid_gray[i] = cv2.cvtColor(X_valid[i], cv2.COLOR_RGB2GRAY)    
	
	for i in range(len(y_test)):
	    X_test_gray[i] = cv2.cvtColor(X_test[i], cv2.COLOR_RGB2GRAY)
	    
	X_train_gray = X_train_gray[:,:,:,np.newaxis]
	X_valid_gray = X_valid_gray[:,:,:,np.newaxis]
	X_test_gray = X_test_gray[:,:,:,np.newaxis]
	print(X_train.shape)
	print(X_train_gray.shape)

output
	(34799, 32, 32, 3)
	(34799, 32, 32, 1)

Here is an example of a traffic sign image before and after grayscaling.

	### plot a picture
	index = random.randint(0, len(X_train))
	image = X_train[index]
	plt.figure(figsize=(2,2))
	plt.imshow(image, cmap="gray")
	
	image_gray = X_train_gray[index].squeeze()
	plt.figure(figsize=(2,2))
	plt.imshow(image_gray, cmap="gray")

output:

![alt text][image11]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


	### Define your architecture here.
	from tensorflow.contrib.layers import flatten

	def LeNet(x,drop_out):    
	    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
	    mu = 0
	    sigma = 0.1
	    
	    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
	    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma),name='conv1_W')
	    conv1_b = tf.Variable(tf.zeros(6),name='conv1_b')
	    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID',name = 'conv1') + conv1_b
	
	    # SOLUTION: Activation.
	    #conv1 = tf.nn.relu(conv1)
	    conv1 = tf.nn.elu(conv1) 
	    
	    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
	    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	
	    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
	    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma),name='conv2_W')
	    conv2_b = tf.Variable(tf.zeros(16),name='conv2_b')
	    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID',name = 'conv2') + conv2_b
	    
	    # SOLUTION: Activation.
	    # conv2 = tf.nn.relu(conv2)
	    conv2 = tf.nn.elu(conv2) 
	    
	    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
	    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	
	    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
	    fc0   = flatten(conv2)
	    
	    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
	    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma),name = 'fc1_W')
	    fc1_b = tf.Variable(tf.zeros(120),name = 'fc1_b')
	    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
	    
	    # SOLUTION: Activation.
	    #fc1    = tf.nn.relu(fc1)
	    fc1    = tf.nn.elu(fc1)
	    
	    # dropout 1
	    fc1 = tf.nn.dropout(fc1, drop_out)
	
	    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
	    #Output = 120
	    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma),name = 'fc2_W')
	    fc2_b  = tf.Variable(tf.zeros(84),name = 'fc2_b')
	    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
	    
	    # SOLUTION: Activation.
	    #fc2    = tf.nn.relu(fc2)
	    fc2    = tf.nn.elu(fc2)
	    
	    # dropout 2
	    fc2 = tf.nn.dropout(fc2, drop_out)
	    print("drop_out =",drop_out)
	
	    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes.
	    #Input = 120
	    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma),name = 'fc3_W')
	    fc3_b  = tf.Variable(tf.zeros(n_classes),name = 'fc3_b')
	    logits = tf.matmul(fc2, fc3_W) + fc3_b
	    
	    return logits

My final model consisted of the following layers:

LayerDescription

1) Convolution  5x5 1x1 stride, valid padding, inputs 32x32x1, outputs 28x28x6  
2) ELU				activation  
3) Max pooling	      2x2 stride, outputs 14x14x6  
4) Convolution 5x5	    1x1 stride, valid padding, inputs 14x14x6 outputs 10x10x16  
5) ELU				activation  
6) Max pooling	     2x2 stride, outputs 5x5x16  
7) flattern		outputs 400  
8) Fully connected	outputs 120  
9) ELU				activation  
10) Fully connected	outputs 84  
11) ELU				activation  
12) Fully connected	outputs 42

open an command window, active carnd-term1 enviroment, carnd-term1 is my conda enviroment.Then active tensorboard,my logdir is log_lenet_graph.

write tensorGraph file

	summary_writer = tf.summary.FileWriter('log_lenet_graph', sess.graph)
activate tensorboard in command window

	$ activate carnd-term1
	$ tensorboard --logdir=log_lenet_graph

show tensorboard in the browser

	http://localhost:6006/

![alt text][image12]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

1) To train the model, I used an structure based on original Lenet5 , add drop_out in the Lenet function.  
2) EPOCHS is 50  
3) BATCHSIZE is 128  
4) learningrate is 0.0005  
I add an interface in the tensorflow session , I tried  decrease learing rate depending the accuracy, but it seems no big help.  So I select fixed learning rate. 

define the learning pipeline

	### Feel free to use as many code cells as needed.
	x = tf.placeholder(tf.float32, (None, 32, 32, 1),name = 'x')
	y = tf.placeholder(tf.int32, (None),name = 'y')
	one_hot_y = tf.one_hot(y, n_classes)
	
	rate = tf.placeholder(tf.float32, name = 'rate')
	drop_out = tf.placeholder(tf.float32)
	
	logits = LeNet(x,drop_out)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate = rate)
	training_operation = optimizer.minimize(loss_operation)
	
define the evaluation function

	###Module Evaluation
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()
	
	def evaluate(X_data, y_data, drop):
	    num_examples = len(X_data)
	    total_accuracy = 0
	    sess = tf.get_default_session()
	    for offset in range(0, num_examples, BATCH_SIZE):
	        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
	        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y ,drop_out :drop})
	        total_accuracy += (accuracy * len(batch_x))
	    return total_accuracy / num_examples
	


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

my code:

	### Train your model here.
	
	X_train_augment = X_train_gray
	y_train_augment = y_train
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    num_examples = len(X_train_augment)
	    
	    print("Training...")
	    print()
	    traning_accuracy = 0
	    for i in range(EPOCHS):
	        X_train_augment, y_train_augment = shuffle(X_train_augment, y_train_augment)
	        learning_rate = 0.0005
	        for offset in range(0, num_examples, BATCH_SIZE):
	            ### Traning pipeline
	            end = offset + BATCH_SIZE
	            batch_x, batch_y = X_train_augment[offset:end], y_train_augment[offset:end]
	            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y , rate:learning_rate ,drop_out:0.6})
	        
	        traning_accuracy = evaluate(X_train_augment, y_train_augment, drop = 1.0 )
	        validation_accuracy = evaluate(X_valid_gray, y_valid, drop = 1.0 )
	        print("EPOCH {} ...".format(i+1))
	        print("traning_accuracy = {:.3f}".format(traning_accuracy))
	        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
	        print()
	            
	    saver.save(sess, './lenet')
	    print("Model saved")

output:

	Training...
	
	EPOCH 1 ...
	traning_accuracy = 0.568
	Validation Accuracy = 0.487
	
	EPOCH 2 ...
	traning_accuracy = 0.761
	Validation Accuracy = 0.678
	
	EPOCH 3 ...
	traning_accuracy = 0.847
	Validation Accuracy = 0.766
	
	EPOCH 4 ...
	traning_accuracy = 0.890
	Validation Accuracy = 0.805
	
	EPOCH 5 ...
	traning_accuracy = 0.914
	Validation Accuracy = 0.840
	
	EPOCH 6 ...
	traning_accuracy = 0.933
	Validation Accuracy = 0.851
	
	EPOCH 7 ...
	traning_accuracy = 0.948
	Validation Accuracy = 0.873
	
	EPOCH 8 ...
	traning_accuracy = 0.956
	Validation Accuracy = 0.878
	
	EPOCH 9 ...
	traning_accuracy = 0.961
	Validation Accuracy = 0.900
	
	......
	......
	
	EPOCH 41 ...
	traning_accuracy = 0.998
	Validation Accuracy = 0.940
	
	EPOCH 42 ...
	traning_accuracy = 0.998
	Validation Accuracy = 0.941
	
	EPOCH 43 ...
	traning_accuracy = 0.998
	Validation Accuracy = 0.943
	
	EPOCH 44 ...
	traning_accuracy = 0.998
	Validation Accuracy = 0.946
	
	EPOCH 45 ...
	traning_accuracy = 0.999
	Validation Accuracy = 0.945
	
	EPOCH 46 ...
	traning_accuracy = 0.999
	Validation Accuracy = 0.945
	
	EPOCH 47 ...
	traning_accuracy = 0.999
	Validation Accuracy = 0.943
	
	EPOCH 48 ...
	traning_accuracy = 0.999
	Validation Accuracy = 0.946
	
	EPOCH 49 ...
	traning_accuracy = 0.999
	Validation Accuracy = 0.945
	
	EPOCH 50 ...
	traning_accuracy = 0.999
	Validation Accuracy = 0.942
	
	Model saved

evaluate the test set

	# Now (drumroll) evaluate the accuracy of the model on the test dataset

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    saver2 = tf.train.import_meta_graph('./lenet.meta')
	    saver2.restore(sess, "./lenet")
	    test_accuracy = evaluate(X_test_gray, y_test, drop = 1.0 )
	    print("Test Set Accuracy = {:.3f}".format(test_accuracy))
output  

	Test Set Accuracy = 0.930

My final model results were:  

* training set accuracy of 0.999  
* validation set accuracy of 0.942  
* test set accuracy of 0.930

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?  
I use Lenet5, because by now this architecture is the only one i know. If i learned other architecture, i hope i can tell the difference.
* What were some problems with the initial architecture?
Using Lenet5, with out dropout, i found when the training accuracy is 0.99, the validation accuracy is only 0.91. I think it is overfitting. 
* How was the architecture adjusted and why was it adjusted?  
  I add two dropout layer in the architecture. When the dropout=1.0, the validation accuray is 0.91, when the dropout=0.7,the validation accuray is 0.93. 
* Which parameters were tuned? How were they adjusted and why?  
I change Relu to ELU. After that, the validation accuray is incresed from 0.93 to 0.94. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
---

### Step3: Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]  
![alt text][image8] ![alt text][image13] ![alt text][image14] ![alt text][image15]

	### Load the images and plot them here.
	### Feel free to use as many code cells as needed.
	
	#reading in an image
	import glob
	import matplotlib.image as mpimg
	
	fig, axs = plt.subplots(2,4, figsize=(6, 3))
	fig.subplots_adjust(hspace = .2, wspace=.001)
	axs = axs.ravel()
	
	my_images = []
	
	for i, img in enumerate(glob.glob('./my-found-traffic-signs/*x.png')):
	    image = cv2.imread(img)
	    axs[i].axis('off')
	    axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	    my_images.append(image)
	
	my_images = np.asarray(my_images)
	
	my_images_gray = np.zeros((my_images.shape[0],32,32),dtype = float)
	#my_images_gray = np.sum(my_images/3, axis=3, keepdims=True)
	
	for i in range(len(my_images)):
	    my_images_gray[i] = cv2.cvtColor(my_images[i], cv2.COLOR_RGB2GRAY)
	my_images_gray = my_images_gray[:,:,:,np.newaxis]    


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

my code: 

	### Calculate the accuracy for these 5 new images. 
	### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
	### Visualize the softmax probabilities here.
	### Feel free to use as many code cells as needed.
	
	softmax_logits = tf.nn.softmax(logits)
	top_k = tf.nn.top_k(softmax_logits, k=3)
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    saver = tf.train.import_meta_graph('./lenet.meta')
	    saver.restore(sess, "./lenet")
	    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_gray, drop_out: 1.0})
	    my_top_k = sess.run(top_k, feed_dict={x: my_images_gray, drop_out: 1.0})
	
	    fig, axs = plt.subplots(len(my_images),4, figsize=(12, 14))
	    fig.subplots_adjust(hspace = .4, wspace=.2)
	    axs = axs.ravel()
	
	    for i, image in enumerate(my_images):
	        axs[4*i].axis('off')
	        axs[4*i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	        #axs[4*i].set_title('input')
	        axs[4*i].set_title(str(my_labels[i]) + ' ' + class_names[str(my_labels[i])])
	        
	        guess1 = my_top_k[1][i][0]
	        index1 = np.argwhere(y_valid == guess1)[0]
	        axs[4*i+1].axis('off')
	        axs[4*i+1].imshow(X_valid[index1].squeeze(), cmap='gray')
	        axs[4*i+1].set_title('top guess: {} ({:.0f}%)'.format(guess1, 100*my_top_k[0][i][0]))
	        
	        guess2 = my_top_k[1][i][1]
	        index2 = np.argwhere(y_valid == guess2)[0]
	        axs[4*i+2].axis('off')
	        axs[4*i+2].imshow(X_valid[index2].squeeze(), cmap='gray')
	        axs[4*i+2].set_title('2nd guess: {} ({:.0f}%)'.format(guess2, 100*my_top_k[0][i][1]))
	        
	        guess3 = my_top_k[1][i][2]
	        index3 = np.argwhere(y_valid == guess3)[0]
	        axs[4*i+3].axis('off')
	        axs[4*i+3].imshow(X_valid[index3].squeeze(), cmap='gray')
	        axs[4*i+3].set_title('3rd guess: {} ({:.0f}%)'.format(guess3, 100*my_top_k[0][i][2]))

output:  
![alt text][image16]

Here are the results of the prediction:

Image Prediction  
Speed limit(60km/h) , Speed limit(60km/h)  
Right-of-way , Right-of-way  
Speed limit(30km/h) , Speed limit(70km/h)  
Priority road , Priority road  
Keep right , Keep right  
Turn left ahead , Turn left ahead  
General caution , General caution  

The accuracy of predictions is 75%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions:

	### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
	### Feel free to use as many code cells as needed.
	
	fig, axs = plt.subplots(8,2, figsize=(9, 19))
	axs = axs.ravel()
	
	for i in range(len(my_softmax_logits)*2):
	    if i%2 == 0:
	        axs[i].axis('off')
	        axs[i].imshow(cv2.cvtColor(my_images[i//2], cv2.COLOR_BGR2RGB))
	    else:
	        axs[i].bar(np.arange(n_classes), my_softmax_logits[(i-1)//2]) 
	        print(max(my_softmax_logits[(i-1)//2]))
	        axs[i].set_ylabel('Softmax probability')

output:  
![alt text][image16]  
1.0  
1.0  
0.803198  
1.0  
1.0  
1.0  
1.0  
0.896039  

Probability Prediction

Image Prediction  
1.00 Speed limit(60km/h)  
1.00 Right-of-way  
0.80 Speed limit(70km/h)  
1.00 Priority road  
1.00 Keep right  
1.00 Turn left ahead  
0.89 General caution


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

my code, define a function to dispaly feature map.

	### Visualize your network's feature maps here.
	### Feel free to use as many code cells as needed.
	
	# image_input: the test image being fed into the network to produce the feature maps
	# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
	# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
	# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
	
	def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
	    # Here make sure to preprocess your image_input in a way your network expects
	    # with size, normalization, ect if needed
	    # image_input =
	    # Note: x should be the same name as your network's tensorflow data placeholder variable
	    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
	    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
	    featuremaps = activation.shape[3]
	    plt.figure(plt_num, figsize=(15,15))
	    #plt.suptitle('this is the figure title', fontsize = 5 ,horizontalalignment = 'right' )
	    for featuremap in range(featuremaps):
	        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
	        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
	        if activation_min != -1 & activation_max != -1:
	            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
	        elif activation_max != -1:
	            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
	        elif activation_min != -1:
	            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
	        else:
	            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")  

my code, display convolution layer1 

	with tf.Session() as sess:
	    saver.restore(sess, "./lenet")
	    print('convolution layer1 28x28x6')
	    conv_layer_2_visual = sess.graph.get_tensor_by_name('conv1:0')
	    outputFeatureMap(my_images_gray,conv_layer_2_visual,plt_num=1)

output:  
this layer find the edge of the traffic sign.  
![alt text][image18] 

my code, display convolution layer2  

	with tf.Session() as sess:
	    saver.restore(sess, "./lenet")
	    print('Convolutional layer2 10x10x16.')
	    conv_layer_2_visual = sess.graph.get_tensor_by_name('conv2:0')
	    outputFeatureMap(my_images_gray,conv_layer_2_visual,plt_num=2)  

output:  
![alt text][image19] 

---

###log  
1. use cv2.cvtColor to prayscale the dataset
	Traning accuracy = 0.95 , valid accuracy = 0.85
2. change learning rate to 0.0003
	Traning accuracy = 0.99 , valid accuracy = 0.91
3. change activation function from Relu to ELU
	Traning accuracy = 0.99 , valid accuracy = 0.92
4. Drop_out = 0.7
	Traning accuracy = 0.99 , valid accuracy = 0.94

### Suggest possible improvements to the program
1. I try gamma,  skew image, rot image on the dataset, but haven't improve the accuracy.
   I think my functions is something wrong. I will test them later.
2. I have not do augmentation on the dataset, I know the balance of the dataset is important.
   I need find some good solution.
3. Some pictures are dark. I think make these picture brighter maybe a good solution.

### Thanks
Thank you Aki saitoh, your kindly support help me a lot. 
Some suggestions i can't understand, but i think i can understand later.