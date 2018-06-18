# 1.0         young.park         this code implement a RNN that adds two 16bit numbers.


import copy, numpy as np
import tensorflow as tf
import datetime


np.random.seed(0)

# training dataset generation
int2binary = {}
binary_dim = 16   #number of bits of variable that will be added
input_dim = 2
hidden_dim = 16
output_dim = 2

# data generation. int2binary[x] holds binary for x; 
#.  e.g., int2binary[2] = np.array([0,0,0,0,0,0,1,0])
#.  e.g., int2binary[4] = np.array([0,0,0,0,0,1,0,0])

largest_number = pow(2,binary_dim)
binary8 = np.unpackbits(np.array([range(pow(2,8))],dtype=np.uint8).T,axis=1)
int16 =  np.array(range(pow(2,16)),dtype=np.uint16) 
#dec_16 = np.array([range(pow(2,16))],dtype=np.uint16).T
#dec_8  = np.array([range(pow(2,8 ))],dtype=np.uint16).T
def bits2int(bitlist):
     out = 0
     for bit in bitlist:
         out = (out << 1) | bit
     return out

def int2bits(val): #16 bits
     binary16 = []
     div, mod = np.divmod( val, pow(2,8)) # 256 
     #pow(2,16)
     b_div = binary8[div]
     b_mod = binary8[mod]
     binary16 = np.concatenate((b_div, b_mod))    
     return np.array(binary16)

# print ("int2bits " , int2bits(3)) 
# a = [1,2,3]
# b = np.array(a, dtype = np.uint8)    
# print("" , a)
# print(" " , b)
 
int2binary = {}
for i in int16:
    #print ("aaa ", i)
    int2binary[i] = int2bits(i)
#for i in range(largest_number):
#    int2binary[i] = binary[i]

# RNN has 2 input cells, 16 cells in the hidden layer and 2 output cells
# parameters to be learned are:
#   w0      input -> hidden :  2 by 16 
#   h       hidden -> hidden : 16 by 16
#   w1      hidden -> output : 16 by 2
#      they are initizlied with random values between -1.0 and 1.0

W0= tf.Variable(2*np.random.rand(input_dim, hidden_dim)-1, dtype=tf.float32)
H = tf.Variable(2*np.random.rand(hidden_dim, hidden_dim)-1, dtype=tf.float32)
W1= tf.Variable(2*np.random.rand(hidden_dim, output_dim)-1, dtype=tf.float32)


X_placeholder = tf.placeholder(tf.float32, [input_dim, binary_dim]) #2 by 8; each colume is a 2 bit input,  
C_placeholder = tf.placeholder(tf.int32, [1, binary_dim]) #expected bits of the sum
# the RNN should learn to calcualte C_placeholder[i] from
#   X_placeholder[0,i] + X_placeholder[1,i]  

#split inputs 
inputs_series = tf.unstack(X_placeholder, axis=1)
labels_series = tf.unstack(C_placeholder, axis=1) 

#For each new input values, the hidden state values are carried over from the previous learning
#For example, after RNN learns from, 2+3 = 5, 
#   the leanred hidden layer values (learned feature of the RNN) will be used 
#   at the beginning of the next learning, e.g., 4+7 = 11
init_state = tf.placeholder(tf.float32, [1, hidden_dim])
current_state = init_state

#list holding the hidden layer values; they are used to calculate outputs and outputs are 
#compared to expected values to calculate errors between actual and expected output
states_series = []

#main forward network
for input_data in inputs_series:
    X = tf.reshape(input_data, [1,2])
    #print( X )   
    #print (labels_series) 

    #this is heart of RNN -- 
    #hidden layer is calculated based on the input values and the values of hidden layer of previous time step
    next_state = tf.sigmoid( tf.matmul(X, W0) + tf.matmul(current_state, H) )  # Broadcasted addition
    states_series.append(next_state) #remember the state for the output calculation
    current_state = next_state
    #print( datetime.datetime.time(datetime.datetime.now()) )
    #print( "length of states_series ", len(states_series))


#this block is the training algorithm -- learn 'cross entropy' for predicting categorical output
#in summary, RNN produces probability of 0/1 and the probability is compared to the expected output bit
logits_series = [tf.matmul(state, W1) for state in states_series] 
#apply softmax nonlinearity for output probability
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits = logits, name="sparce") 
    for labels, logits  in zip(labels_series, logits_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

#the following is standard Tensorflow (TF) steps to train the model
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    loss_list = []

    _current_state = np.zeros((1, hidden_dim))
    print("_current_state", _current_state, "shape", _current_state.shape)

    for j in range(10000+1):

        a_int = np.random.randint(largest_number/2) # rand out of 0 - 2^7
        b_int = np.random.randint(largest_number/2) # rand out of 0 - 2^7
        a = int2binary[a_int] 
        b = int2binary[b_int]  
        c_int = a_int + b_int
        c = int2binary[c_int]
        #print("c  is ", c)        
        #rint("C ", C, "C shape ", C.shape )

        #flip the binary within each list so that the least significant bit (LSB) could be fed first.
        X = np.vstack((np.flip(a, axis=0),np.flip(b, axis=0)))
        C = np.reshape(np.flip(c, axis=0), (1, binary_dim)) 

        #training model
        _total_loss, _train_step, _current_state, _logits_series, _predictions_series, _W0, _H, _W1 , _states_series  = sess.run(
            [ total_loss, train_step , current_state, logits_series, predictions_series, W0, H, W1, states_series],
            feed_dict={
                X_placeholder:X,
                C_placeholder:C,
                init_state:_current_state
            })

        loss_list.append(_total_loss)
        
        #infomration about the progress
        if j%100 == 0:
            #print("_current_state ",  _current_state)
            #print("_logit_series ",  _logits_series)
            #print("len(states_series) ",  len(_states_series))

            
            actual = []
            out_series = np.flip(_predictions_series, axis=0)
            for k in out_series:
                #print("k ", k)
                if( k[0][0] < k[0][1]): 
                	actual.append(1) # = actual + 1 #np.append(actual, 1)
                else: 
                	actual.append(0) # = actual + 0 #np.append(actual, 0)

            expected = bits2int(c)
            actually = bits2int(actual)
            delta = expected - actually

            #print("a ", a, "+b ", b, "= expected ", c,  "actual ", actual.astype(int))
            #print(" expected ", c,  "actual ", actual) #actual.astype(int))
            #print("Step",j, "Total Loss", _total_loss, ": expected ", expected,  "actual ", actually, "delta ", delta) #actual.astype(int))
            #sprintf("Step",j, "Total Loss", _total_loss, ": expected ", expected,  "actual ", actually, "delta ", delta) #actual.astype(int))
            msg = "Step %5d Total Loss  %10f : expected %7d actual %7d delta  %7d" % ( j, _total_loss, expected, actually, delta ) #actual.astype(int))
            print(msg)


#  ÛÛÛÛÛ                ÛÛÛÛÛ                                        
# °°ÛÛÛ                °°ÛÛÛ                                         
#  °ÛÛÛÛÛÛÛ   ÛÛÛÛÛÛ   ÛÛÛÛÛÛÛ   ÛÛÛÛÛÛÛÛÛÛÛÛÛ    ÛÛÛÛÛÛ   ÛÛÛÛÛÛÛÛ  
#  °ÛÛÛ°°ÛÛÛ °°°°°ÛÛÛ °°°ÛÛÛ°   °°ÛÛÛ°°ÛÛÛ°°ÛÛÛ  °°°°°ÛÛÛ °°ÛÛÛ°°ÛÛÛ 
#  °ÛÛÛ °ÛÛÛ  ÛÛÛÛÛÛÛ   °ÛÛÛ     °ÛÛÛ °ÛÛÛ °ÛÛÛ   ÛÛÛÛÛÛÛ  °ÛÛÛ °ÛÛÛ 
#  °ÛÛÛ °ÛÛÛ ÛÛÛ°°ÛÛÛ   °ÛÛÛ ÛÛÛ °ÛÛÛ °ÛÛÛ °ÛÛÛ  ÛÛÛ°°ÛÛÛ  °ÛÛÛ °ÛÛÛ 
#  ÛÛÛÛÛÛÛÛ °°ÛÛÛÛÛÛÛÛ  °°ÛÛÛÛÛ  ÛÛÛÛÛ°ÛÛÛ ÛÛÛÛÛ°°ÛÛÛÛÛÛÛÛ ÛÛÛÛ ÛÛÛÛÛ
# °°°°°°°°   °°°°°°°°    °°°°°  °°°°° °°° °°°°°  °°°°°°°° °°°° °°°°° 
#Or, maybe you reached a local minimum, that's why the value of the cost function 
# oscillates. Try to add a small random noise to avoid the local minima or try 
# another method instead of gradient descent - stochastic gradient descent or 
# use a Radial Basis Function neural network.
#https://www.researchgate.net/post/Why_MLP_is_not_converging

#Although the behaviour shown by your model is rare, yet I think it's somehow explainable. You are right that stochastic gradient serves as some noise during learning and may help escape local minima. However, the samples for each class contained in successive batches may be somewhat inconsistent (so noisy) such that weights updates is erratic and therefore causes the value of your loss function to oscillate in the error-weight space. As you pointed out in your case, increasing the batch size exposes the neural network to more samples per class every iteration; and hence, there's lesser influence of highly noisy samples on weights update, and consequently the minimized loss experience little or no oscillate. i.e. consistent decrease in the loss. I hope this helps.
#https://www.researchgate.net/post/Why_MLP_is_not_converging


