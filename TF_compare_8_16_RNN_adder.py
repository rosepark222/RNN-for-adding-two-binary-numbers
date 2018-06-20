# 1.0         young.park         this code implement a RNN that adds two 8bit numbers.
#                                it build model using Tensorflow.
#                                it was originally motivated by Siraj's youtube.                                  
#                                   Build a Recurrent Neural Net in 5 Min
#                                   https://www.youtube.com/watch?v=cdLUzrjnlr4

import copy, numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML
import csv

loss8  = []
loss16  = []


with open('loss_8bit.csv', 'r') as csvfile:
   spamreader = csv.reader(csvfile, delimiter=',')
   for row in spamreader:
      row = list(map(float, row))
      loss8 = row

#how to convert to number
#print("loss8 is", loss8)

plt.plot(loss8, 'ro', alpha=0.5)
#plt.show()

with open('loss_16bit.csv', 'r') as csvfile:
   spamreader = csv.reader(csvfile, delimiter=',')
   for row in spamreader:
      row = list(map(float, row))
      loss16 = row

plt.plot(loss16, alpha=0.5)
plt.show()

