<!-- Jun 17th, added a code performing the 16 bit addition.

    TF_binary_add_16bits.py


Two different implementations of adding two 8 bit binary numbers using RNN.

TF_binary_add.py  
	tensorflow implementation, stochastic gradient descent
siraj_rnn_binary_add.py 
	numpy implementation (no extra library)
siraj_rnn_tensorflow_echo_gen.py 
	I learned to use specific training method in this example. 
	It is based on minibatch gradient descent.
--> 



# Backpropagation through time in RNN
 

[WildML-RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) says:
"We also use the backpropagation algorithm, but with a little twist. Because the parameters are shared by all time steps in the network, the gradient at each output depends not only on the calculations of the current time step, but also the previous time steps. For example, in order to calculate the gradient at t=4 we would need to backpropagate 3 steps and sum up the gradients. This is called Backpropagation Through Time (BPTT)." 

Then, [WindML-backpropagation-through-time](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/) elaborates how BPTT works. 

If we see the dE3/dW, we see the summation of four terms, k=0 to 3,

<img width="225" alt="screen shot 2018-11-16 at 6 11 38 pm" src="https://user-images.githubusercontent.com/38844805/48667833-26188280-ea94-11e8-850f-90fa092eb3e5.png">

Why do we need to add four terms? Why dE3/dW depends on dS0/dW, dS1/dW, dS2/dW and dS3/dW? Let's talk over the following diagram:

<img width="800" alt="screen shot 2018-11-16 at 6 09 40 pm" src="https://user-images.githubusercontent.com/38844805/48667837-33357180-ea94-11e8-9b1b-c6db50fdb36c.png">

It is obvious that dE3/dW depends on dS3/dW because E3 is function of S3 and S3 is function of W. When we want to know the impact of small perturbance of W to S3, we have to remember that S2 also change, because S2 is function of W as well. How do we account for S2 change as well? S3 is function of W and S2 and W and S2 are function of W. We can use Einstein summation explained in [This article](https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf
), which is the df/dr is expressed as df/dx dx/dr + df/dy dy/dr, when f is a function of both x and y, i.e., f(x,y), and x and y are functions of r. In other words, the change of f due to small change or r is the total change (sum) due to the the change of f on dr through x and the change of f on dr through y. 

Based on this relatioship, change of S3 on change of W is sum of change of S3 on W plus change of S3 on W through S2 (dS3/dS2 dS2/dW). Now, we have to apply the same rule for the dS2/dW because it also depends on dS2/dS1 due to the shared W. 

<!-- The reason is that W are shared by all time steps. dE3/dW means what's the change of E3 by the change of W? Chaning W would also change S2, this would change S3 and impact E3. Thus, E3 can be changed partially by direct change in W through S3 or partially by W through S2 through S3, and so on. This partial effects should be summed to calculate total change of E3 by W. This is the best intuitive explanation I can come up with. Then, why the gradients are summed rather than 
Why do we need to add four terms? 

the key that I got from study group was the f(x,y) and x and y are depend on r use Einstein summation for the dF/dr. If we believe this relationship df/dr = df/dx dx/dr + df/dy dy/dr, then this can be applied to the backprop of RNN. 
S3 in 
http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
depends on W and S2, because S2 depends on W. then ds3/dw -> ds3/dw + ds2/dw (very strange notation of ->).
--> 

# In plain English, perturbance in W affects S3 both 1) directly and 2) through S2.

<!-- 
Reference:
http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/

This one has a nice rigorous mathematical derivation -- Einstein summation, chain rule, and matrix derivatives.
--> 
