Jun 17th, added a code performing the 16 bit addition.

    TF_binary_add_16bits.py


Two different implementations of adding two 8 bit binary numbers using RNN.

TF_binary_add.py  
	tensorflow implementation, stochastic gradient descent
siraj_rnn_binary_add.py 
	numpy implementation (no extra library)
siraj_rnn_tensorflow_echo_gen.py 
	I learned to use specific training method in this example. 
	It is based on minibatch gradient descent.

We also use the backpropagation algorithm, but with a little twist. Because the parameters are shared by all time steps in the network, the gradient at each output depends not only on the calculations of the current time step, but also the previous time steps. For example, in order to calculate the gradient at t=4 we would need to backpropagate 3 steps and sum up the gradients. This is called Backpropagation Through Time (BPTT). 

<img width="225" alt="screen shot 2018-11-16 at 6 11 38 pm" src="https://user-images.githubusercontent.com/38844805/48655398-1d0fae80-e9cb-11e8-9d2f-a7eb878ed8be.png">

Why dE3/dW depends on dS0/dW, dS1/dW, dS2/dW and dS3/dW? The reason is that W are shared by all time steps. dE3/dW means what's the change of E3 by the change of W? Chaning W would also change S2, this would change S3 and impact E3. Thus, E3 can be changed partially by direct change in W through S3 or partially by W through S2 through S3, and so on. This partial effects should be summed to calculate total change of E3 by W. This is the best intuitive explanation I can come up with.
 
<img width="800" alt="backprop 2018-11-16 at 6 09 40 pm" src="https://user-images.githubusercontent.com/38844805/48655375-dfab2100-e9ca-11e8-8d95-ed1ba7c29f88.png">


Reference:
http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/

