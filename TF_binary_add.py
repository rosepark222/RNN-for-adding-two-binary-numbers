# erp029:
#Build a Recurrent Neural Net in 5 Min
#https://www.youtube.com/watch?v=cdLUzrjnlr4

import copy, numpy as np
import tensorflow as tf
import datetime


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
input_dim = 2
hidden_dim = 16
output_dim = 2

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

#coefficient = np.array([[1.], [-10.], [25.]])
#XY  = np.array([[1,2,3],[4,5,6]])
#print( "dimension: ", XY.shape)
#print( "XY[1,0] should be 4: ", XY[1,0])
#print( "XY[1,1] should be 5: ", XY[1,1])
#ZZ  = np.array([[1,2,3], [4,5,6], [1,1,1]])
#print ( np.matmul(XY, ZZ) )

# so there are 2 bit inputs fed to 16 cells and 2 bit outputs

# w0      input -> hidden :  2 by 16 
# h       hidden -> hidden : 16 by 16
# w1      hidden -> output : 16 by 2
# init them with random values between -1.0 and 1.0


W0= tf.Variable(2*np.random.rand(input_dim, hidden_dim)-1, dtype=tf.float32)
H = tf.Variable(2*np.random.rand(hidden_dim, hidden_dim)-1, dtype=tf.float32)
W1= tf.Variable(2*np.random.rand(hidden_dim, output_dim)-1, dtype=tf.float32)


X_placeholder = tf.placeholder(tf.float32, [input_dim, binary_dim])
#2 by 8 (2 dim matrix)
#print( X_placeholder )  
C_placeholder = tf.placeholder(tf.int32, [1, binary_dim])
#1 by 8 
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




inputs_series = tf.unstack(X_placeholder, axis=1) # 8 arrays of 2 bits each
labels_series = tf.unstack(C_placeholder, axis=1) 
#tf.split(C_placeholder, binary_dim) #tf.unstack(C_placeholder, axis=1)

init_state = tf.placeholder(tf.float32, [1, hidden_dim])

current_state = init_state
states_series = []

for input_data in inputs_series:
    X = tf.reshape(input_data, [1,2])
    #print( X )   
    #print (labels_series) 
    next_state = tf.sigmoid( tf.matmul(X, W0) + tf.matmul(current_state, H) )  # Broadcasted addition
    states_series.append(next_state)
   
    #set current state to next one
    current_state = next_state
    #print( datetime.datetime.time(datetime.datetime.now()) )
    print( "length of states_series ", len(states_series))


logits_series = [tf.matmul(state, W1) for state in states_series] #Broadcasted addition
#apply softmax nonlinearity for output probability
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits = logits, name="sparce") 
    for labels, logits  in zip(labels_series, logits_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


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

    _current_state = np.zeros((1, hidden_dim))
    print("_current_state", _current_state, "shape", _current_state.shape)

    for j in range(10000+1):

        a_int = np.random.randint(largest_number/2) # int version
        b_int = np.random.randint(largest_number/2) # int version

        #print("a_int is ", a_int)
        #print("b_int is ", b_int)

        a = int2binary[a_int] # binary encoding
        b = int2binary[b_int] # binary encoding
        #print("a is ", a)
        #print("b is ", b)

        c_int = a_int + b_int
        #print("c_int is ", c_int)        

        c = int2binary[c_int]
        #print("c  is ", c)        
        #rint("C ", C, "C shape ", C.shape )

        #LSB should be fed to the RNN first.
        X = np.vstack((np.flip(a, axis=0),np.flip(b, axis=0)))
        C = np.reshape(np.flip(c, axis=0), (1, binary_dim)) #  [C]

        #X = np.flip(X, axis=0)
        #C = np.flip(C, axis=0)
        #print("vstack \n", X, "X shape ", X.shape)
        #print("C ", C, "C shape ", C.shape )

        _total_loss, _train_step, _current_state, _logits_series, _predictions_series, _W0, _H, _W1 , _states_series  = sess.run(
            [ total_loss, train_step , current_state, logits_series, predictions_series, W0, H, W1, states_series],
            feed_dict={
                X_placeholder:X,
                C_placeholder:C,
                init_state:_current_state
            })

        loss_list.append(_total_loss)
        

        if j%100 == 0:
            print("Step",j, "Total Loss", _total_loss)
            #print("_current_state ",  _current_state)
            #print("_logit_series ",  _logits_series)
            #print("len(states_series) ",  len(_states_series))

            
            actual = []
            out_series = np.flip(_predictions_series, axis=0)
            for k in out_series:
                #print("k ", k)
                if( k[0][0] < k[0][1]): 
                	actual = np.append(actual, 1)
                else: 
                	actual = np.append(actual, 0)


            print("a ", a, "+b ", b, "= ", c,  "actual ", actual)


                #plot(loss_list, _predictions_series, batchX, batchY)

#  ÛÛÛÛÛ                ÛÛÛÛÛ                                        
# °°ÛÛÛ                °°ÛÛÛ                                         
#  °ÛÛÛÛÛÛÛ   ÛÛÛÛÛÛ   ÛÛÛÛÛÛÛ   ÛÛÛÛÛÛÛÛÛÛÛÛÛ    ÛÛÛÛÛÛ   ÛÛÛÛÛÛÛÛ  
#  °ÛÛÛ°°ÛÛÛ °°°°°ÛÛÛ °°°ÛÛÛ°   °°ÛÛÛ°°ÛÛÛ°°ÛÛÛ  °°°°°ÛÛÛ °°ÛÛÛ°°ÛÛÛ 
#  °ÛÛÛ °ÛÛÛ  ÛÛÛÛÛÛÛ   °ÛÛÛ     °ÛÛÛ °ÛÛÛ °ÛÛÛ   ÛÛÛÛÛÛÛ  °ÛÛÛ °ÛÛÛ 
#  °ÛÛÛ °ÛÛÛ ÛÛÛ°°ÛÛÛ   °ÛÛÛ ÛÛÛ °ÛÛÛ °ÛÛÛ °ÛÛÛ  ÛÛÛ°°ÛÛÛ  °ÛÛÛ °ÛÛÛ 
#  ÛÛÛÛÛÛÛÛ °°ÛÛÛÛÛÛÛÛ  °°ÛÛÛÛÛ  ÛÛÛÛÛ°ÛÛÛ ÛÛÛÛÛ°°ÛÛÛÛÛÛÛÛ ÛÛÛÛ ÛÛÛÛÛ
# °°°°°°°°   °°°°°°°°    °°°°°  °°°°° °°° °°°°°  °°°°°°°° °°°° °°°°° 
