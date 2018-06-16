#How to Use Tensorflow for Time Series (Live)
#https://www.youtube.com/watch?v=hhJIztWR_vo&t=8s
#https://github.com/llSourcell/How-to-Use-Tensorflow-for-Time-Series-Live-/blob/master/demo_full_notes.ipynb


from __future__ import print_function, division
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from IPython.display import Image
#from IPython.core.display import HTML 
import datetime



#Image(url= "https://cdn-images-1.medium.com/max/1600/1*UkI9za9zTR-HL8uM15Wmzw.png")
#hyperparams

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length



#Step 1 - Collect data
#Now generate the training data, 
#the input is basically a random binary vector. The output will be the 
#“echo” of the input, shifted echo_step steps to the right.

#Notice the reshaping of the data into a matrix with batch_size rows, e.g. (5, 10000),
#Neural networks are trained by approximating the gradient of loss function 
#with respect to the neuron-weights, by looking at only a small subset of the data, 
#also known as a mini-batch. The reshaping takes the whole dataset and puts it into 
#a matrix, that later will be sliced up into these mini-batches.

#for overview of mini-batch read 
#    https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
#    The updates at the end of the training epoch require the additional complexity of accumulating prediction errors across all training examples.
#    I see what it means. simple error calculation of "layer_2_error = y - layer_2" in siraj_rnn_youtube cannot be accumucated because positive and negative errors would cancel out if they are added together. 



def generateData():
    #0,1, 50K samples, 50% chance each chosen
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    #shift 3 steps to the left
    y = np.roll(x, echo_step)
    #padd beginning 3 values with 0
    y[0:echo_step] = 0
    #Gives a new shape to an array without changing its data.
    #The reshaping takes the whole dataset and puts it into a matrix, 
    #that later will be sliced up into these mini-batches.
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

xx,yy = generateData()

print("xx[:, :10] ", xx[:, 0:10])
print("xx.shape ", xx.shape )
print("yy[:, :10] ", yy[:, 0:10])
print("yy.shape ", yy.shape )
# xx  [
#  [1 1 0 ... 0 0 1]
#  [1 0 0 ... 1 1 1]
#  [0 0 1 ... 1 0 0]
#  [1 0 0 ... 1 0 1]
#  [0 0 1 ... 1 1 0]]
# xx.shape  (5, 10000)


#Schematic of the reshaped data-matrix, arrow curves shows adjacent time-steps that ended up on different rows. 
#Light-gray rectangle represent a “zero” and dark-gray a “one”.
#Image(url= "https://cdn-images-1.medium.com/max/1600/1*aFtwuFsboLV8z5PkEzNLXA.png")

#TensorFlow works by first building up a computational graph, that 
#specifies what operations will be done. The input and output of this graph
#is typically multidimensional arrays, also known as tensors. 
#The graph, or parts of it can then be executed iteratively in a 
#session, this can either be done on the CPU, GPU or even a resource 
#on a remote server.

#operations and tensors

#The two basic TensorFlow data-structures that will be used in this 
#example are placeholders and variables. On each run the batch data 
#is fed to the placeholders, which are “starting nodes” of the 
#computational graph. Also the RNN-state is supplied in a placeholder, 
#which is saved from the output of the previous run.

#Step 2 - Build the Model

#datatype, shape (5, 15) 2D array or matrix, batch size shape for later
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

#and one for the RNN state, 5,4 
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

#The weights and biases of the network are declared as TensorFlow variables,
#which makes them persistent across runs and enables them to be updated
#incrementally for each batch.

#batch1 1 bit -> 4 bit in hidden state
#batch2 1 bit -> 4 bit in hidden state
#batch3 1 bit -> 4 bit in hidden state
#batch4 1 bit -> 4 bit in hidden state
#batch5 1 bit -> 4 bit in hidden state

#Then, 4 bits are fed back to the next time and since 1 bit is input , total 5 bits coming in. 
#The hidden stage should produce 4 bit output, thus 5 x 4 = 20 weights are needed for each batch.

#3 layer recurrent net, one hidden state
#randomly initialize weights
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
#anchor, improves convergance, matrix of 0s 
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

#Image(url= "https://cdn-images-1.medium.com/max/1600/1*n45uYnAfTDrBvG87J-poCA.jpeg")



#Now it’s time to build the part of the graph that resembles the actual RNN computation, 
#first we want to split the batch data into adjacent time-steps.

# Unpack columns
#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#so a bunch of arrays, 1 batch per time step
#unstack will turn columns into the separate array

inputs_series = tf.unstack(batchX_placeholder, axis=1)
# node1 = tf.unstack(batchX_placeholder, axis=1)
# inputs_series = tf.Print( node1, [node1], "#iinputs_series: ")
labels_series = tf.unstack(batchY_placeholder, axis=1)

# In one epoch, there are 666 batches. 
# Each batch is a list of 5 elements (rows) and each row is 15 binary numbers
# the length of 15 is defined by truncated_backprop_length
# batchX [[0 0 0 1 0 0 1 1 1 1 1 0 0 0 0]
#  [0 1 1 0 0 0 1 1 0 1 1 0 1 1 1]
#  [1 0 1 0 1 1 0 0 0 0 1 1 1 0 0]
#  [0 0 1 0 0 1 0 0 0 1 1 0 0 0 1]
#  [0 1 0 0 0 1 0 1 1 0 1 1 0 0 1]]
# and batchY [[0 0 0 0 0 0 1 0 0 1 1 1 1 1 0]
#  [1 0 1 0 1 1 0 0 0 1 1 0 1 1 0]
#  [0 1 0 1 0 1 0 1 1 0 0 0 0 1 1]
#  [0 1 1 0 0 1 0 0 1 0 0 0 1 1 0]
#  [1 0 1 0 1 0 0 0 1 0 1 1 0 1 1]]
# and 
# _inputs_serise [array([0., 0., 1., 0., 0.], dtype=float32), array([0., 1., 0., 0., 1.], dtype=float32), array([0., 1., 1., 1., 0.], dtype=float32), array([1., 0., 0., 0., 0.], dtype=float32), array([0., 0., 1., 0., 0.], dtype=float32), array([0., 0., 1., 1., 1.], dtype=float32), array([1., 1., 0., 0., 0.], dtype=float32), array([1., 1., 0., 0., 1.], dtype=float32), array([1., 0., 0., 0., 1.], dtype=float32), array([1., 1., 0., 1., 0.], dtype=float32), array([1., 1., 1., 1., 1.], dtype=float32), array([0., 0., 1., 0., 1.], dtype=float32), array([0., 1., 1., 0., 0.], dtype=float32), array([0., 1., 0., 0., 0.], dtype=float32), array([0., 1., 0., 1., 1.], dtype=float32)]
# _labels_series [array([0, 1, 0, 0, 1], dtype=int32), array([0, 0, 1, 1, 0], dtype=int32), array([0, 1, 0, 1, 1], dtype=int32), array([0, 0, 1, 0, 0], dtype=int32), array([0, 1, 0, 0, 1], dtype=int32), array([0, 1, 1, 1, 0], dtype=int32), array([1, 0, 0, 0, 0], dtype=int32), array([0, 0, 1, 0, 0], dtype=int32), array([0, 0, 1, 1, 1], dtype=int32), array([1, 1, 0, 0, 0], dtype=int32), array([1, 1, 0, 0, 1], dtype=int32), array([1, 0, 0, 0, 1], dtype=int32), array([1, 1, 0, 1, 0], dtype=int32), array([1, 1, 1, 1, 1], dtype=int32), array([0, 0, 1, 0, 1], dtype=int32)]


#Image(url= "https://cdn-images-1.medium.com/max/1600/1*f2iL4zOkBUBGOpVE7kyajg.png")
#Schematic of the current batch split into columns, the order index is shown on each data-point 
#and arrows show adjacent time-steps.

dummy_variable = 1231212312123121231212312
# In a very simple RNN, the state is a vector holding values for the nodes in the layer, thus 
# a one dimensional array is enough.
# Then, why the state is a matrix in this code? The reason is 5 bits are fed simultaneously and 
# there are virtually 5 mini-RNN within this structure. In each mini-RNN, 
# there are 4 nodes in the hidden layer. 

# I am still puzzled with this structure. What is the reason behind this 5 batch structure? 
# Is this only way or it has been done for some performance reason? 
# How to design the RNN surely depend on how inputs are generated in real applcation. 
# This model assumes that inputs are fed 5 bits at a given time. 


#Forward pass
#state placeholder
current_state = init_state
#series of states through time
# TensorFlow will convert many different Python objects into tf.Tensor objects when they are passed as arguments to TensorFlow operators.
# (I googled up 'tensor vs python variable' and hit the 
#  https://stackoverflow.com/questions/39512276/tensorflow-simple-operations-tensors-vs-python-variables
#)
# That makes sense. My python variables got transformed to tensors to form the graph and got emptied out on session.run().
states_series = []

#question is then, why weights are not cleared up over different batch run or different epoch?
#question is then, why weights are not cleared up over different batch run or different epoch?
#question is then, why weights are not cleared up over different batch run or different epoch?
#question is then, why weights are not cleared up over different batch run or different epoch?
#how does tensorflow knows which tensors are cleared up for each session.run and others are kept? 
#how does tensorflow knows which tensors are cleared up for each session.run and others are kept? 
#how does tensorflow knows which tensors are cleared up for each session.run and others are kept? 
#how does tensorflow knows which tensors are cleared up for each session.run and others are kept? 
#how does tensorflow knows which tensors are cleared up for each session.run and others are kept? 




#for each set of inputs
#forward pass through the network to get new state value
#store all states in memory and used for producing outputs and loss
# Why all states are stored in states_series ? 
# I beleive it could be accomplished within the same for loop, 
# but I think that storing all states are easier to code and debug later.


#looping 15 times
for current_input in inputs_series:
    #print(  current_input )
    #format input
    current_input = tf.reshape(current_input, [batch_size, 1])
    #mix both state and input data 
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns
    #perform matrix multiplication between weights and input, add bias
    #squash with a nonlinearity, for probabiolity value
    
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    #node1 = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    #next_state = tf.Print( node1, [node1], "#next_state: ")
    #store the state in memory
    states_series.append(next_state)
    
    #set current state to next one
    current_state = next_state

    print( datetime.datetime.time(datetime.datetime.now()) )
                                          


# _input_and_state_concatenated 
# [[1.         0.99347264 0.93316406 0.99679774 0.94608176]
#  [1.         0.9929322  0.9304558  0.99653095 0.94201547]
#  [1.         0.9930358  0.93088526 0.9965813  0.94282305]
#  [1.         0.9840871  0.9148303  0.9947946  0.92397386]
#  [0.         0.9839264  0.9145133  0.9947451  0.92321426]]
# _W                              
# [[ 0.45119816  0.09884373  0.24129266  0.15632254]
#  [ 0.48506612  0.8974638   0.8999862  -0.00489075]
#  [ 0.6240867   0.2837278   0.29742885  0.77622265]
#  [ 0.5533387   0.18204261  0.95141745  0.05035832]
#  [ 0.84840906  0.18208294  0.89953744  0.8533669 ]]


#The key idea of Parallel Training of Recurrent Neural Networks
#    http://eknight7.github.io/ParallelRNN/report/parallelRNNReport.pdf
#    Each pipeline is a fully connected network of 5 inputs (1 input + 4 prev.states) 
#    and 4 outputs (next.states) -- requires 20 weights in W matrix.
#    All pipelines share the same Weights.


# The meaning of matrix multipliation?
# matmul is the operation of linear system written in a very simple math format. 
# input * weight  --> multiplication of two values in one dimension
# sum( inputs * weights ) --> matrix multiplication

# row i of _input_and_state_concatenated is i-th input+prev.state  (i = 0,1,2,3,4)
# column j of W are weights to the j-th state (j = 0,1,2,3) 
# matmul(i, j) means i-th input+prev.state passes through network of jth previous weights and summed to form current jth state.

# for example, the first input row is multipled by the first column of 
# W to produce the output of the first value of the firist pipeline.
# [1.         0.99347264 0.93316406 0.99679774 0.94608176] (matmul) 
#  [[ 0.45119816  ]
#   [ 0.48506612  ]
#   [ 0.6240867   ]
#   [ 0.5533387   ]
#   [ 0.84840906  ]]                                   

# for example, the first input row is multipled by the second column of 
# W to produce the output of the second value of the firist pipeline.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
# for example, the second input row is multipled by the first column of 
# W to produce the output of the first value of the second pipeline.


# _next_state 
# [[0.9935295  0.93338263 0.99682385 0.9465517 ]
#  [0.99345773 0.9331087  0.9967906  0.9459609 ]
#  [0.9934711  0.9331586  0.996797   0.9460698 ]
#  [0.9930511  0.9309469  0.996589   0.9429407 ]
#  [0.98286486 0.9116447  0.994393   0.9186698 ]]
# _current_state 
# [[0.9935295  0.93338263 0.99682385 0.9465517 ]
#  [0.99345773 0.9331087  0.9967906  0.9459609 ]
#  [0.9934711  0.9331586  0.996797   0.9460698 ]
#  [0.9930511  0.9309469  0.996589   0.9429407 ]
#  [0.98286486 0.9116447  0.994393   0.9186698 ]]
# _W2 [[ 0.532441   -0.02658456]
#  [ 0.3360257   0.46598867]
#  [ 0.20406891  0.04165077]
#  [ 0.42936248  0.4821142 ]]
# _logits_series [array([[0., 0.],
#        [0., 0.],
#        [0., 0.],
#        [0., 0.],
#        [0., 0.]], dtype=float32), 
# _predictions_series [array([[0.5, 0.5],
#        [0.5, 0.5],
#        [0.5, 0.5],
#        [0.5, 0.5],
#        [0.5, 0.5]], dtype=float32),  
# _losses [array([0.6931472, 0.6931472, 0.6931472, 0.6931472, 0.6931472],
#       dtype=float32), array([0.6931472, 0.6931472, 0.7028808, 0.6931472, 0.7028808],
#       dtype=float32), array([0.6931472, 0.7028808, 0.5956159, 0.6835074, 0.8012292],
#       dtype=float32), array([0.6931472, 0.8179193, 0.8652193, 0.8179193, 0.8652193],
#       dtype=float32), array([0.6931472 , 0.9107408 , 0.50087476, 0.9107408 , 0.5008749 ],
#       dtype=float32), array([0.7028808, 0.4878094, 0.9589313, 0.4878094, 0.9589313],
#       dtype=float32), array([0.8179193 , 0.48016733, 0.96569514, 0.48016733, 0.9656951 ],
#       dtype=float32), array([0.9277378 , 0.96677494, 0.97489405, 0.97464263, 0.9670932 ],
#       dtype=float32), array([0.48306358, 0.9750641 , 0.9764621 , 0.9764299 , 0.97511375],
#       dtype=float32), array([0.47421685, 0.9764838 , 0.96932733, 0.9693221 , 0.96911323],
#       dtype=float32), array([0.47271472, 0.9693308 , 0.47790343, 0.47324446, 0.97541916],
#       dtype=float32), array([0.9693054 , 0.47790298, 0.47338235, 0.47259027, 0.4770694 ],
#       dtype=float32), array([0.473246  , 0.47808096, 0.9691312 , 0.9766671 , 0.9754267 ],
#       dtype=float32), array([0.47259042, 0.9751787 , 0.97542185, 0.47694853, 0.47706866],
#       dtype=float32), array([0.9766671, 0.9764984, 0.4725924, 0.4732409, 0.9677684],
#       dtype=float32)]
#_total_loss 0.6902163

#Image(url= "https://cdn-images-1.medium.com/max/1600/1*fdwNNJ5UOE3Sx0R_Cyfmyg.png")

#calculate loss
#second part of forward pass
#logits short for logistic transform
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
#apply softmax nonlinearity for output probability
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]


#erp029 what is softmax?
# https://stackoverflow.com/questions/34240703/whats-the-difference-between-softmax-and-softmax-cross-entropy-with-logits
# Logits simply means that the function operates on the unscaled output of earlier layers and that the relative scale to understand the units is linear. It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities (you might have an input of 5).
# tf.nn.softmax produces just the result of applying the softmax function to an input tensor. The softmax "squishes" the inputs so that sum(input) = 1; it's a way of normalizing. The shape of output of a softmax is the same as the input - it just normalizes the values. The outputs of softmax can be interpreted as probabilities.
# a = tf.constant(np.array([[.1, .3, .5, .9]]))
# print s.run(tf.nn.softmax(a))
# [[ 0.16838508  0.205666    0.25120102  0.37474789]]
# In contrast, tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function (but it does it all together in a more mathematically careful way). It's similar to the result of:
# sm = tf.nn.softmax(x)
# ce = cross_entropy(sm)
# The cross entropy is a summary metric - it sums across the elements. The output of tf.nn.softmax_cross_entropy_with_logits on a shape [2,5] tensor is of shape [2,1] (the first dimension is treated as the batch).


#measure loss, calculate softmax again on logits, then compute cross entropy
#measures the difference between two probability distributions
#this will return A Tensor of the same shape as labels and of the same type as logits 
#with the softmax cross entropy loss.

#zip- Iterate over two lists in parallel
#https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
#A common use case is to have logits of shape [batch_size, num_classes] and labels of shape [batch_size]. But higher dimensions are supported.

# #Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy  
# #Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy 
# #Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy 
# #Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy 
# #https://www.youtube.com/watch?v=tRsSi_sqXjI

# # This is the label of the output (expected output) -- I am sure it will go through one hot encoding in the sparse_softmax_cross_entropy_with_logits function 

# _labels_series [array([0, 1, 1, 1, 1], dtype=int32), 14 more arrays)

# # This is the logits output of the RNN 
# _logits_series [array([[0.74856526, 0.3401068 ],
#        [0.74856526, 0.3401068 ],
#        [0.        , 0.        ],
#        [0.74856526, 0.3401068 ],
#        [0.74856526, 0.34010682]], dtype=float32), 14 more arrays)

# # This is the softmax of logits --- considered as the probability of 0 and 1
# _predictions_series [array([[0.6007182 , 0.3992818 ],
#        [0.6007182 , 0.3992818 ],
#        [0.5       , 0.5       ],
#        [0.6007182 , 0.3992818 ],
#        [0.6007182 , 0.39928183]], dtype=float32),  14 more arrays)

#loss values are:

# _losses [array([0.50962937, 0.91808784, 0.6931472 , 0.91808784, 0.91808784],
#       dtype=float32),  14 more arrays)

# Formula for calculating the loss function based on cross entropy (basically based on the concept of the correlation)
# D(S,L) = - sum ( Li * ln (Si)), Li are labels and Si are softmax of output of NN, where the number of Si is equal to that of Li.
# see https://www.youtube.com/watch?v=tRsSi_sqXjI

# Example for the first element of the above data:
# L = 1, 0 (the labels for the category 0 and 1)
# S = 0.6007182 , 0.3992818 (softmax values for category 0 and 1)
# D(S,L) = -1 * ( 1 * ln(0.6007182) +  0 * ln(0.3992818)) = 0.51 ===== which is the value of losses

# Example for the second element of the above data:
# L = 0, 1 (the labels for the category 0 and 1)
# S = 0.6007182 , 0.3992818 (softmax values for category 0 and 1)
# D(S,L) = -1 * ( 0 * ln(0.6007182) +  1 * ln(0.3992818)) = 0.918 
# A large loss value means the difference between expected and output of RNN. 
# For example, expected outcome is 1 and the probabilty of output 0 is high. 
# This error will backprop and adjust wegits accordingly so that the loss  will be small.

# #Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy  
# #Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy 
# #Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy 
# #Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy Cross Entropy 



losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits = logits, name="sparce") for labels, logits  in zip(labels_series, logits_series)]
#computes average, one value
#Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true, the reduced dimensions are retained with length 1.
#If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
total_loss = tf.reduce_mean(losses)
#use adagrad to minimize with .3 learning rate
#minimize it with adagrad, not SGD
#One downside of SGD is that it is sensitive to
#the learning rate hyper-parameter. When the data are sparse and features have
#different frequencies, a single learning rate for every weight update can have
#exponential regret.
#Some features can be extremely useful and informative to an optimization problem but 
#they may not show up in most of the training instances or data. If, when they do show up, 
#they are weighted equally in terms of learning rate as a feature that has shown up hundreds 
#of times we are practically saying that the influence of such features means nothing in the 
#overall optimization. it's impact per step in the stochastic gradient descent will be so small 
#that it can practically be discounted). To counter this, AdaGrad makes it such that features 
#that are more sparse in the data have a higher learning rate which translates into a larger 
#update for that feature
#sparse features can be very useful.
#Each feature has a different learning rate which is adaptable. 
#gives voice to the little guy who matters a lot
#weights that receive high gradients will have their effective learning rate reduced, 
#while weights that receive small or infrequent updates will have their effective learning rate increased. 
#great paper http://seed.ucsd.edu/mediawiki/images/6/6a/Adagrad.pdf
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


#visualizer
# def plot(loss_list, predictions_series, batchX, batchY):
#     plt.subplot(2, 3, 1)
#     plt.cla()
#     plt.plot(loss_list)

#     for batch_series_idx in range(5):
#         one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
#         single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

#         plt.subplot(2, 3, batch_series_idx + 2)
#         plt.cla()
#         plt.axis([0, truncated_backprop_length, 0, 2])
#         left_offset = range(truncated_backprop_length)
#         plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
#         plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
#         plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

#     plt.draw()
#     plt.pause(0.0001)

#Step 3 Training the network
with tf.Session() as sess:
    #we stupidly have to do this everytime, it should just know
    #that we initialized these vars. v2 guys, v2..
    
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables()) OLD CODE

    #interactive mode
    #plt.ion()
    #initialize the figure
    #plt.figure()
    #show the graph
    #plt.show()
    #to show the loss decrease
    loss_list = []

    for epoch_idx in range(num_epochs):
        #generate data at eveery epoch, batches run in epochs
        x,y = generateData()
        #initialize an empty hidden state
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx, "shape of x", x.shape)
        #New data, epoch 0 shape of x (5, 10000)
        #each batch
        for batch_idx in range(num_batches):
            #starting and ending point per batch
            #since weights reoccuer at every layer through time
            #These layers will not be unrolled to the beginning of time, 
            #that would be too computationally expensive, and are therefore truncated 
            #at a limited number of time-steps
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]
                        #run the computation graph, give it the values
            #we calculated earlier
            # _total_loss, _train_step, _current_state, _predictions_series, _next_state, _input_and_state_concatenated, _W, _W2,_inputs_series, _labels_series, _logits_series, _losses, _states_series = sess.run(
            #     [total_loss, train_step, current_state, predictions_series, next_state, input_and_state_concatenated, W, W2, inputs_series, labels_series, logits_series, losses, states_series],
            #     feed_dict={
            #         batchX_placeholder:batchX,
            #         batchY_placeholder:batchY,
            #         init_state:_current_state
            #     })
            _total_loss, _train_step, _current_state   = sess.run(
                [ total_loss, train_step , current_state],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

#           if batch_idx < 1:
#                 print( "Num_batches", num_batches, "Step", batch_idx, "batchX",  batchX )
#                 print( "Num_batches", num_batches, "Step", batch_idx, "batchY",  batchY )
#                 print( "_inputs_serise", _inputs_series)
#                 print( "_labels_series", _labels_series)
#                 print( "_input_and_state_concatenated", _input_and_state_concatenated)
#                 print( "_W", _W)
#                 print( "_next_state", _next_state)
#                 print( "_current_state", _current_state) 
#                 print( "_W2", _W2)
#                 print( "_logits_series", _logits_series)
# #                print( "_predictions_series", _predictions_series)
#                 print( "_losses", _losses)
#                 print( "_total_loss", _total_loss)
# #            if batch_idx < 2:
#                 print( "_states_series shape", np.shape(_states_series), " type ", _states_series) # 15 of 5 by 4 state matrix values
#                 print( "states_series shape", np.shape(states_series), " type ", states_series) # 15 of 5 by 4 state matrix values
#_states_series shape (15, 5, 4)
#states_series shape (15,)  type  [<tf.Tensor 'Tanh:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_1:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_2:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_3:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_4:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_5:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_6:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_7:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_8:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_9:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_10:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_11:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_12:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_13:0' shape=(5, 4) dtype=float32>, <tf.Tensor 'Tanh_14:0' shape=(5, 4) dtype=float32>]

#                print( "dummy_variable ", dummy_variable, " _dummy_variable ", _dummy_variable) # 15 of 5 by 4 state matrix values


            loss_list.append(_total_loss)
            #print("Step",batch_idx, "Loss", _total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                #plot(loss_list, _predictions_series, batchX, batchY)

#plt.ioff()
#plt.show()

#Notice the reshaping of the data into a matrix with batch_size rows, e.g. (5, 10000),
#Neural networks are trained by approximating the gradient of loss function 
#with respect to the neuron-weights, by looking at only a small subset of the data, 
#also known as a mini-batch. The reshaping takes the whole dataset and puts it into 
#a matrix, that later will be sliced up into these mini-batches.
 


# >>> a= [[1, 0, 1, 1], [1,1,1,1]]
# >>> a
# [[1, 0, 1, 1], [1, 1, 1, 1]]
# >>> a[1,1]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: list indices must be integers or slices, not tuple
# >>> a[1]
# [1, 1, 1, 1]
# >>> a[1][1]
# 1
# >>> a[0][1]
# 0
# >>> a[:,:]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: list indices must be integers or slices, not tuple
# >>> a[:,1:2]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: list indices must be integers or slices, not tuple
# >>> a[:,0:1]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: list indices must be integers or slices, not tuple
# >>> a.shape
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: 'list' object has no attribute 'shape'
# >>> a
# [[1, 0, 1, 1], [1, 1, 1, 1]]
# >>> b = np.array(a)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'np' is not defined
# >>> import numpy as np
# >>> b = np.array(a)
# >>> b
# array([[1, 0, 1, 1],
#        [1, 1, 1, 1]])
# >>> b.shape
# (2, 4)
# >>> b[:,:]
# array([[1, 0, 1, 1],
#        [1, 1, 1, 1]])
# >>> b[:]
# array([[1, 0, 1, 1],
#        [1, 1, 1, 1]])
# >>> a
# [[1, 0, 1, 1], [1, 1, 1, 1]]
# >>> b
# array([[1, 0, 1, 1],
#        [1, 1, 1, 1]])
# >>> 2*a
# [[1, 0, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
# >>> 2*b
# array([[2, 0, 2, 2],
#        [2, 2, 2, 2]])
# >>> type(a)
# <class 'list'>
# >>> type(b)
# <class 'numpy.ndarray'>
# >>> 


#
# when the current_state is not inside sess.run(); it won't carry over the current state across batches
# saw the loss never decreases
#
# New data, epoch 0 shape of x (5, 10000)
# Step 0 Loss 0.72985166
# Step 100 Loss 0.69420236
# Step 200 Loss 0.6896452
# Step 300 Loss 0.6947153
# Step 400 Loss 0.6949766
# Step 500 Loss 0.6909782
# Step 600 Loss 0.71106017
# ...
# New data, epoch 98 shape of x (5, 10000)
# Step 0 Loss 0.13841936
# Step 100 Loss 0.13968574
# Step 200 Loss 0.14810446
# Step 300 Loss 0.14402653
# Step 400 Loss 0.13453431
# Step 500 Loss 0.13569207
# Step 600 Loss 0.1413332
# New data, epoch 99 shape of x (5, 10000)
# Step 0 Loss 0.1410306
# Step 100 Loss 0.14106669
# Step 200 Loss 0.13692199
# Step 300 Loss 0.13940886
# Step 400 Loss 0.13490462
# Step 500 Loss 0.14523473
# Step 600 Loss 0.14085202

#
#
# when the current_state is specifically evaluted and carried over to the next batches; we got much better performance
#
#
# New data, epoch 0 shape of x (5, 10000)
# Step 0 Loss 0.69334286
# Step 100 Loss 0.4238033
# Step 200 Loss 0.022217385
# Step 300 Loss 0.00889777
# Step 400 Loss 0.005028118
# Step 500 Loss 0.0037809892
# Step 600 Loss 0.0027132034
# ...
# New data, epoch 98 shape of x (5, 10000)
# Step 0 Loss 0.14602518
# Step 100 Loss 2.4035251e-05
# Step 200 Loss 2.400029e-05
# Step 300 Loss 2.8738286e-05
# Step 400 Loss 2.8086652e-05
# Step 500 Loss 2.362836e-05
# Step 600 Loss 2.0594205e-05
# New data, epoch 99 shape of x (5, 10000)
# Step 0 Loss 0.14024259
# Step 100 Loss 2.6846537e-05
# Step 200 Loss 2.2727067e-05
# Step 300 Loss 2.686878e-05
# Step 400 Loss 3.222829e-05
# Step 500 Loss 2.8361434e-05
# Step 600 Loss 2.7777838e-05
