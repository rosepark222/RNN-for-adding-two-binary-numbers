# erp029:
#Build a Recurrent Neural Net in 5 Min
#https://www.youtube.com/watch?v=cdLUzrjnlr4

import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# training dataset generation
int2binary = {}
binary_dim = 8
#binary_dim = 10

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# erp029:
#numpy.ndarray.T
#    Same as self.transpose(), except that self is returned if self.ndim < 2.   
# >>> x
# array([[ 1.,  2.],
#        [ 3.,  4.]])
# >>> x.T
# array([[ 1.,  3.],
#        [ 2.,  4.]])


# >>> int2binary
# array([[0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 0, 1, 0],
#        [0, 0, 0, 0, 0, 0, 1, 1],
#        [0, 0, 0, 0, 0, 1, 0, 0],
#        [0, 0, 0, 0, 0, 1, 0, 1],
#        [0, 0, 0, 0, 0, 1, 1, 0],
#        [0, 0, 0, 0, 0, 1, 1, 1],
#        [0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 1, 0, 0, 1]], dtype=uint8)

#>>> binary[1]
#array([0, 0, 0, 0, 0, 0, 0, 1], dtype=uint8)

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16 #orignal value
#hidden_dim = 6
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000+1):
    
    # generate a simple addition problem (a + b = c)
    # erp029: make sure c does not blow up, so make a and b smaller than half of the max value
    a_int = np.random.randint(largest_number/2) # int version
    b_int = np.random.randint(largest_number/2) # int version

    a = int2binary[a_int] # binary encoding
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

# erp029:
# X shape (1, 2)
# synapse_0 shape (2, 16)
# layer_1 shape (1, 16)

# synapse_1 shape (16, 1)
# layer_2 shape (1, 1)
# y shape (1, 1)


        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        if(j == 100): 
            print("X shape " + str( np.shape(X) ))
            print("y shape " + str( np.shape(y) ))
            print("layer_1 shape " + str( np.shape(layer_1) ))            
            print("layer_2 shape " + str( np.shape(layer_2) ))
            print("synapse_0 shape " + str( np.shape(synapse_0) ))            
            print("synapse_1 shape " + str( np.shape(synapse_1) ))



# >>> a = np.array([[1,2]])
# >>> a
# array([[1, 2]])
# >>> b = np.array([1,2])
# >>> b
# array([1, 2])
# >>> a.shape
# (1, 2)
# >>> b.shape
# (2,)

# 

#The (length,) array is an array where each element is a number 
#    and there are length elements in the array. 
#The (length, 1) array is an array which also has length elements, 
#    but each element itself is an array with a single element. 

# >>> import numpy as np
# >>> a = np.array( [[1],[2],[3]] )
# >>> a.shape
# >>> (3, 1)
# >>> b = np.array( [1,2,3] )
# >>> b.shape
# >>> (3,)



        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2  #this is actaully the derivate of 1/2(y-y^)^2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    #last stage does not have future layer 1 
    future_layer_1_delta = np.zeros(hidden_dim)

#   http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html 
#   http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html 
#   http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html 
#   http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html 
#   http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html 
#   http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html 
#   http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html 

# erp029:
# Data feedforward occurred from LSB to MSB and layer values are stored accordingly
# Backpropagation is easier to perform from MSB to LSB 


    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]

#erp029
#        layer_2_error = y - layer_2
#        layer_2_deltas.append((layer_2_error)*
#            sigmoid_output_to_derivative(layer_2))

#erp029
#this is the most challening component of RNN       
#error propagation is like a time travel (reverse of the time)
#backwardprop is the reverse of forward prop. 
# if you see the backward prop, forward prop can be inferred. 
# layer_1 affects future layer_1 through synapse_h and 
#                 layer 2 through synapse_1

#
#at each time step, future_layer_1_delta is multiplied by same synapse_h after its 
#updated using below calculation. This is reverse of the feedforward, and synapse_h 
#is hold constant for given 8 bit inputs (each data). 
#Also, I see vanishing gradient because this 
#recursive multiplication of synapse_h to the layer_1_delta, 
# future hidden layer = (w (w (w (w previous hidden layer)))), specially w < 1.
#the delta close to input got all production of weight*deltas on the way to it.
#the delta close to output gets less production of intermediate deltas.
#LSB got processed first, thus deltas close to LSB got more production of deltas.
#
        # error at  hidden layer hidden layer hidden layer hidden layer hidden layer
        # The forward method of RecurrentStateUnfold iteratively updates the states through time and returns the resulting state tensor. The backward method propagates the gradients at the outputs of each state backwards through time. Note that at each time k the gradient coming from the output Y needs to be added with the gradient coming from the previous state at time k+1. 
        # http://peterroelants.github.io/posts/rnn_implementation_part02/

        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

#erp029
# update for weights in output layer = weight* input 
# are all format of input.T.dot(output layer delta),
# remember layer_1 is the hidden layer of 16 values and layer_2 is the single bit output layer

#http://www.cs.toronto.edu/~rgrosse/csc321/lec10.pdf
#this class material shows why the derivatives are added up for synapse_012

        #code developer decided to program gradient from layer 2 to error (layer_2_delta), and layer 1 to error 
        # (layer_1_delta). 
        # layer_2_delta is straightfoward NN from layer 2 to error. 
        # layer_1_delta is the sum of the two gradients; one from the current layer 1 to future layer 1 and the other 
        #    from layer 1 to error (using layer_2_delta). future layer 1 already includes gradients to 
        #    furture future layer 1 which contains all path to future errors (recursive way)
        # Once layer 1 and 2 delta are ready, synapse update is obtained by multiped by inputs to the layer. 
        #    X -> W -> layer -> Error  (this is the forward path)
        #    d(layer)/dW = layer_delta * X
        #    d(Error)/dW = d(Error)/d(layer) * d(layer)/dW
        #    , where d(Error)/d(layer) is the layer_delta in this code.
        #    X can be input or another layer closer to the input


        #layer_1_delta is common dE/dL1 that sh and s0 pass through
        #Thus, it can be used to update both sh and s0. 
        #layer_1_delta is  gradient from current hidden layer to all future outputs (through future hidden layers)

        #layer_2_delta is gradient from output to error
        #layer_1, pre_layer_1, and X are all layer values prior to weighted by s1, sh and s0 respectively.

        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

#synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
#synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
#synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1        

        future_layer_1_delta = layer_1_delta

#erp029    
#large delta means large contribution to the error
#weight that contributes to the large error will be adjusted more
#--- if you think about it, it is natural.

# The final state gradient at time k=0 is used to optimise the initial state S0 since the gradient of the inital state is ∂ξ/∂S0. 

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print ("J " + str(j))
        print ("Error:" + str(overallError))
        print ("Pred:" + str(d))
        print ("True:" + str(c))
        print ("Synapse_h")
        #print (synapse_h)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print (str(a_int) + " + " + str(b_int) + " = " + str(out))
        print ("------------")

# erp029:
# (Tensorflow) $ python3 rnn.py 
# Error:[3.45638663]
# Pred:[0 0 0 0 0 0 0 1]
# True:[0 1 0 0 0 1 0 1]
# 9 + 60 = 1
# ------------
# Error:[3.63389116]
# Pred:[1 1 1 1 1 1 1 1]
# True:[0 0 1 1 1 1 1 1]
# 28 + 35 = 255
# ------------
# Error:[3.91366595]
# Pred:[0 1 0 0 1 0 0 0]
# True:[1 0 1 0 0 0 0 0]
# 116 + 44 = 72
# ------------
# Error:[3.72191702]
# Pred:[1 1 0 1 1 1 1 1]
# True:[0 1 0 0 1 1 0 1]
# 4 + 73 = 223
# ------------
# Error:[3.5852713]
# Pred:[0 0 0 0 1 0 0 0]
# True:[0 1 0 1 0 0 1 0]
# 71 + 11 = 8
# ------------
# Error:[2.53352328]
# Pred:[1 0 1 0 0 0 1 0]
# True:[1 1 0 0 0 0 1 0]
# 81 + 113 = 162
# ------------
# Error:[0.57691441]
# Pred:[0 1 0 1 0 0 0 1]
# True:[0 1 0 1 0 0 0 1]
# 81 + 0 = 81
# ------------
# Error:[1.42589952]
# Pred:[1 0 0 0 0 0 0 1]
# True:[1 0 0 0 0 0 0 1]
# 4 + 125 = 129
# ------------
# Error:[0.47477457]
# Pred:[0 0 1 1 1 0 0 0]
# True:[0 0 1 1 1 0 0 0]
# 39 + 17 = 56
# ------------
# Error:[0.21595037]
# Pred:[0 0 0 0 1 1 1 0]
# True:[0 0 0 0 1 1 1 0]
# 11 + 3 = 14
# ------------

# Synapse_h
# [[ 2.74598739e-02 -1.57180653e+00  1.57329084e-02  1.28861706e-01
#    5.03705818e-01 -5.63273268e-01 -2.33772829e+00  1.52523229e-01
#   -9.16633564e-02 -7.69017808e-02  1.04502881e-01 -1.19319232e+00
#    5.01068941e-01 -4.19053500e-01  1.59678700e+00 -5.49522326e-01]
#  [-6.72294993e-01  2.14042537e-01  7.68747181e-01 -8.48236376e-01
#    6.66961883e-01 -6.82312038e-01  2.69683890e+00 -2.22215370e+00
#    1.58277487e+00 -1.73177722e-01 -4.32448197e-01 -4.56050305e-01
#   -7.88585107e-01 -1.42175247e+00 -1.71424698e+00 -6.93729439e-01]
#  [-3.03800957e-01 -1.58229374e-01 -9.68397975e-01  2.26242874e-01
#   -1.02116222e-02 -3.52791837e-01  9.01268567e-01 -1.57453878e+00
#    6.67019944e-01  4.04329046e-01 -9.76086794e-01  2.23255300e-01
#   -7.95626402e-01 -7.22896199e-02 -5.76236480e-01 -7.88544667e-01]
#  [ 1.63301780e-01 -1.34488801e+00  5.79230835e-01 -8.21198026e-01
#    1.80802975e-01 -4.16374757e-01 -6.24821364e-01  1.11642256e+00
#   -5.01178112e-01  2.17927230e-01 -3.23651331e-02  6.85361134e-02
#   -1.34993681e-01  9.78104883e-01  7.39450627e-01  7.28965230e-01]
#  [ 3.41262860e-01 -2.62169457e-01  6.17714155e-01 -2.94401350e-01
#    7.00877497e-01  2.80912600e-01  1.63846167e+00 -8.66285740e-01
#    1.00282411e+00 -4.55494022e-01  9.12813145e-02  5.38445172e-01
#   -4.54885720e-01 -4.06719646e-01 -1.38451112e+00 -4.59200533e-01]
#  [ 1.18719807e-01 -1.48699832e-01  1.93632452e-01 -7.22585012e-02
#   -5.57600822e-01 -1.53912218e-01  4.86677665e-01 -7.58646693e-01
#    5.06467826e-01 -1.22057693e-01 -2.57779577e-01 -7.26987548e-02
#    5.29462675e-01 -8.21394730e-01 -6.95345401e-01  7.72623015e-01]
#  [ 1.19497051e+00 -9.35808361e-01 -1.20857693e+00  1.03764585e+00
#    4.55260556e-03  9.54822342e-01 -2.37639863e+00  1.56104868e+00
#   -8.97003256e-01  5.83204709e-01 -2.64593286e-01 -6.57107412e-01
#    1.36570641e+00  7.00788127e-01  1.66908221e+00 -7.74870168e-01]
#  [ 1.11615819e+00 -1.02014175e+00  4.41823288e-02  7.81079190e-01
#    2.10517467e-01  3.97748942e-01 -1.17596346e+00  7.90130829e-01
#    1.08692855e-01  5.82677274e-02 -7.16064547e-02 -1.10716751e+00
#    6.67683363e-01 -5.90272038e-01  5.27623966e-01 -4.14657978e-01]
#  [-3.35546832e-01  9.17492142e-01  4.07134545e-01 -9.51501775e-01
#   -7.61587216e-01  6.46363147e-01  1.82438717e+00 -2.10877414e+00
#    1.33461956e+00 -3.49767468e-01 -6.54617503e-01  1.29553377e-01
#    3.26627816e-01 -1.25642783e+00 -1.44081912e+00 -5.11415993e-01]
#  [-2.77968281e-01 -2.56825865e-01  3.37394057e-01  2.47188347e-02
#   -1.10475357e+00 -4.92171365e-01 -2.36216366e+00  8.40821524e-01
#   -5.24930489e-01  7.19428503e-01 -1.39888921e-01 -1.19304790e+00
#    2.34669140e-02 -4.76026534e-01  1.53491884e+00 -1.26035120e-01]
#  [ 4.31551768e-01 -6.02162907e-01  8.43613231e-01 -3.32023688e-01
#   -1.71868367e-01 -2.20532779e-01 -2.18866680e+00  1.22306774e+00
#   -7.61776332e-01  6.93948719e-02  4.56471894e-01 -2.40604540e-01
#    1.54961006e+00  9.93335820e-01  6.22341768e-01  8.27792784e-01]
#  [-4.85611279e-01  1.25073100e+00  2.58187517e-01 -7.27772403e-01
#    8.30475727e-01  3.16391884e-01  8.47703994e-01 -1.66662048e+00
#    5.47332549e-01 -1.11220612e+00 -1.15100446e+00 -9.18001432e-02
#   -5.56896554e-01 -4.10003409e-01 -1.05039242e+00  8.98655247e-02]
#  [ 1.10982458e+00 -1.84322584e+00 -2.47330920e-01 -5.60029913e-01
#    8.64155038e-02 -5.03499001e-02 -1.46722212e+00  6.89229749e-01
#   -1.09161540e+00  3.30437974e-01  4.12120008e-02  1.26334370e-01
#    1.65796540e+00  8.48208763e-01  1.51607830e+00 -7.95429450e-01]
#  [ 5.90281438e-01 -1.12531160e+00  6.94935659e-01 -2.20869405e-01
#   -1.07684001e+00 -7.54312381e-01 -2.29228342e+00  1.85601012e+00
#   -1.19793638e-01  1.12136778e+00 -7.33570741e-02 -6.62029056e-01
#    4.53597738e-01  4.61257016e-01  2.00697963e+00  9.34523558e-01]
#  [-6.85716331e-01  1.08432060e+00 -4.55559845e-01  6.73923027e-01
#    1.08979953e+00  7.98142193e-01  3.19146732e-03  1.21568294e+00
#   -2.08586384e-03  1.56895923e-01 -7.34851913e-04 -7.21624482e-01
#   -5.11618551e-01 -1.19586016e+00  3.09055381e-01 -2.93125439e-01]
#  [ 1.21842196e-01 -2.69359224e-01 -3.46938749e-01 -5.34566330e-01
#    4.40564731e-01 -5.81688451e-01 -1.24131490e+00 -8.29451230e-02
#   -8.84586822e-01  9.03771164e-01  9.19912599e-01  5.85778597e-01
#    1.41675373e+00  7.53419377e-01  9.34428393e-01 -7.49361677e-01]]


#Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients
#http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
#http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
#http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
#http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
#http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
#http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
#http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/


# Neural Networks Demystified [Part 4: Backpropagation]
# very insightful clip by Welch Labs. I have not watched it fully yet.
# He is very serious for the production of clips. 
# Visual presentation of imaginary number is impressive.
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek
# https://www.youtube.com/watch?v=GlcnxUlrtek