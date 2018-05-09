
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:


# Load pickled data
import pickle
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

print(len(X_train),len(X_valid),len(X_test))
print(np.array(X_train[0]).shape)
print(len(set(y_train)))


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0] #34799 

# TODO: Number of validation examples
n_validation = X_valid.shape[0] #4410

# TODO: Number of testing examples.
n_test = X_test.shape[0] #12630

# TODO: What's the shape of a
get_ipython().set_next_input('n traffic sign image');get_ipython().run_line_magic('pinfo', 'image')
image_shape = list(X_train[0].shape) #[2, 32, 3]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# In[ ]:


n traffic sign image


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[3]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

f,(ax1) = plt.subplots(1)

ax1.imshow(image)
print(y_train[index])


f,(ax1, ax2, ax3 ) = plt.subplots(1, 3, figsize=(20, 5))

ax1.set_title('class of train')
nd = pd.Series(y_train)
nd.hist(ax= ax1, bins=n_classes, range=(0,n_classes))

ax2.set_title('class of valid')
nd = pd.Series(y_valid)
nd.hist(ax= ax2,bins=n_classes, range=(0,n_classes))

ax3.set_title('class of test')
nd = pd.Series(y_test)
nd.hist(ax= ax3,bins=n_classes, range=(0,n_classes))


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[4]:


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import tensorflow as tf
import cv2


#change the train valid and test images to gray, and normalize the image to (pixel - 128)/ 128
#cv2.cvtColor(X_train[0],cv2.COLOR_RGB2GRAY) # change the image to gray
#.astype(np.int16) # change the type of the pixel value from uint8 to int16 
#((cv2.cvtColor(X_train[0],cv2.COLOR_RGB2GRAY)).astype(np.int16) - 128) / 128) # normalize by (pixel - 128)/ 128
#(((cv2.cvtColor(X_train[0],cv2.COLOR_RGB2GRAY)).astype(np.int16) - 128) / 128).reshape([32,32,1])  # reshape the [32,32] image to [32,32,1]

X_train_gray = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
for i in range(len(X_train)):
    X_train_gray[i] = (((cv2.cvtColor(X_train[i],cv2.COLOR_RGB2GRAY)).astype(np.int16) - 128) / 128).reshape([32,32,1]) 

X_valid_gray = np.zeros((X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1))
for i in range(len(X_valid)):
    X_valid_gray[i] = (((cv2.cvtColor(X_valid[i],cv2.COLOR_RGB2GRAY)).astype(np.int16) - 128) / 128).reshape([32,32,1]) 

X_test_gray = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
for i in range(len(X_test)):
    X_test_gray[i] = (((cv2.cvtColor(X_test[i],cv2.COLOR_RGB2GRAY)).astype(np.int16) - 128) / 128).reshape([32,32,1]) 



# ### Model Architecture

# In[5]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten


def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name = 'layer1_W')
    conv1_b = tf.Variable(tf.zeros(6), name = 'layer1_b')
    conv1   = tf.nn.bias_add(tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID', name = 'layer1_conv'), conv1_b, name = 'layer1_conv1')

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1, name = 'layer1_relu')

    # SOLUTION: Pooling. Input = 28x28x6. Output = 27x17x6.
    conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name = 'layer1_avg_pool')

    # SOLUTION: Layer 2: Convolutional. Input = 27x17x6 Output = 23x23x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name = 'layer2_W')
    conv2_b = tf.Variable(tf.zeros(16), name = 'layer2_b')
    conv2   = tf.nn.bias_add(tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', name = 'layer2_conv'), conv2_b, name = 'layer1_conv2')

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2, name = 'layer2_relu')

    # SOLUTION: Pooling. Input = 23x23x16. Output = 22x22x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name = 'layer2_max_pool')

    
    
    # SOLUTION: Layer 3: Convolutional. Input = 22x22x16 Output = 18x18x32.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean = mu, stddev = sigma), name = 'layer3_W')
    conv3_b = tf.Variable(tf.zeros(32), name = 'layer3_b')
    conv3   = tf.nn.bias_add(tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID', name = 'layer3_conv'), conv3_b, name = 'layer1_conv3')

    # SOLUTION: Activation.
    conv3 = tf.nn.relu(conv3, name = 'layer3_relu')

    # SOLUTION: Pooling. Input = 18x18x32. Output = 9x9x32.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name = 'layer3_max_pool')

    # SOLUTION: Flatten. Input = 11x11x32. Output = 400.
    fc0   = flatten(conv3)

    # SOLUTION: Layer 3: Fully Connected. Input = 2592. Output = 1024.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(2592, 1024), mean = mu, stddev = sigma), name = 'fc1_W')
    fc1_b = tf.Variable(tf.zeros(1024), name = 'fc1_b')
    fc1   = tf.matmul(fc0, fc1_W, name = 'fc1_matmul') + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1, name = 'fc1_relu')

    # SOLUTION: Layer 4: Fully Connected. Input = 1024. Output = 512.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 512), mean = mu, stddev = sigma), name = 'fc2_W')
    fc2_b  = tf.Variable(tf.zeros(512), name = 'fc2_b')
    fc2    = tf.matmul(fc1, fc2_W, name = 'fc2_matmul') + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2, name = 'fc2_relu')

    # SOLUTION: Layer 5: Fully Connected. Input = 512. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, 43), mean = mu, stddev = sigma), name = 'fc3_W')
    fc3_b  = tf.Variable(tf.zeros(43), name = 'fc3_b')
    logits = tf.matmul(fc2, fc3_W, name = 'fc3_matmul') + fc3_b

    return logits


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[6]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
from sklearn.utils import shuffle

X_train_gray, y_train = shuffle(X_train_gray, y_train)

#EPOCHS = 20
EPOCHS = 20
BATCH_SIZE = 64


x = tf.placeholder(tf.float32, (None, 32, 32, 1), name = 'x')
y = tf.placeholder(tf.int32, (None), name = 'y')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')  
one_hot_y = tf.one_hot(y, n_classes)


rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



##Train model
max_accuracy = 0.0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_gray)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_gray, y_train = shuffle(X_train_gray, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_gray[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
                
        train_accuracy = evaluate(X_train_gray, y_train)
        
        ##Validate
        validation_accuracy = evaluate(X_valid_gray, y_valid)
        

        
        print("EPOCH {} ...".format(i+1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
        ##Save model
        if validation_accuracy > max_accuracy :
            max_accuracy = validation_accuracy
            saver.save(sess, './lenet')
            print("Model saved")
                    
        print()
        
##Test model   
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_gray, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[7]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.

#index = random.randint(0, len(X_valid_gray))
image_1 = plt.imread('./examples/01_4.jpg') #3
image_2 = plt.imread('./examples/02_35.jpg') 
image_3 = plt.imread('./examples/03_14.jpg') #
image_4 = plt.imread('./examples/04_17.jpg')
image_5 = plt.imread('./examples/05_18.jpg')

image_1 = cv2.resize(image_1,(32,32))
image_2 = cv2.resize(image_2,(32,32))
image_3 = cv2.resize(image_3,(32,32))
image_4 = cv2.resize(image_4,(32,32))
image_5 = cv2.resize(image_5,(32,32))


f,(ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 5))

ax1.imshow(image_1)
ax2.imshow(image_2)
ax3.imshow(image_3)
ax4.imshow(image_4)
ax5.imshow(image_5)


# ### Predict the Sign Type for Each Image

# In[8]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

batch = np.array([image_1, image_2, image_3, image_4, image_5])
batch_x = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], 1))

for i in range(len(batch)):
    batch_x[i] = (((cv2.cvtColor(batch[i],cv2.COLOR_RGB2GRAY)).astype(np.int16) - 128) / 128).reshape([32,32,1])

logits_argmax = tf.argmax(logits,1)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    pred_y = sess.run(logits_argmax, feed_dict={x: batch_x})
    print(pred_y)
    


# ### Analyze Performance

# In[10]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

batch_y = np.array([ 4, 35, 14, 17, 18 ])

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    #v_accuracy = sess.run(validation_accuracy, feed_dict={x: batch_x, y: batch_y})
    #validation_accuracy = evaluate(batch_x, batch_y)
    accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
    
    print(accuracy)


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[11]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

logits_softmax = tf.nn.softmax(logits)
pred_y = None

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    pred_y = sess.run(logits_softmax, feed_dict={x: batch_x})
    pred_y_top5 = sess.run(tf.nn.top_k(tf.constant(pred_y), k=5))

print (pred_y_top5)


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[12]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

# def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
#     # Here make sure to preprocess your image_input in a way your network expects
#     # with size, normalization, ect if needed
#     # image_input =
#     # Note: x should be the same name as your network's tensorflow data placeholder variable
#     # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
#     activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
#     featuremaps = activation.shape[3]
#     plt.figure(plt_num, figsize=(15,15))
#     for featuremap in range(featuremaps):
#         plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
#         plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
#         if activation_min != -1 & activation_max != -1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
#         elif activation_max != -1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
#         elif activation_min !=-1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
#         else:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            
            

batch = np.array([image_1, image_2, image_3, image_4, image_5])
batch_x = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], 1))

for i in range(len(batch)):
    batch_x[i] = (((cv2.cvtColor(batch[i],cv2.COLOR_RGB2GRAY)).astype(np.int16) - 128) / 128).reshape([32,32,1])


f,(ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(20, 20))


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    x = tf.get_default_graph().get_tensor_by_name("x:0")
    layer1_conv1 = tf.get_default_graph().get_tensor_by_name("layer1_conv1:0")
    layer1_conv1 = sess.run(layer1_conv1, feed_dict={x:batch_x})
    #print(layer1_conv1[0])
    layer1_image = layer1_conv1[0]
    #print(layer1_image.transpose()[0])
    
    
    
    ax1.set_title('fileter0')
    ax1.imshow(layer1_image.transpose()[0], cmap='gray')
    
    ax2.set_title('fileter1')
    ax2.imshow(layer1_image.transpose()[1], cmap='gray')
    
    ax3.set_title('fileter2')
    ax3.imshow(layer1_image.transpose()[2], cmap='gray')
    
    ax4.set_title('fileter3')
    ax4.imshow(layer1_image.transpose()[3], cmap='gray')
    
    ax5.set_title('fileter4')
    ax5.imshow(layer1_image.transpose()[4], cmap='gray')
    
    ax6.set_title('fileter5')
    ax6.imshow(layer1_image.transpose()[5], cmap='gray')
    

