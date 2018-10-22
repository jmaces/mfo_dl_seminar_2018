#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import target_functions as tfunc
import os

###############################################################################
# CLUSTER ARRAY-JOB RELATED SETTINGS
###############################################################################
# get id of array-job instance
jobid = os.environ.get('SGE_TASK_ID')
if not jobid: 
    jobid = 1
else:
    jobid = int(jobid)
# define options for all varying parameters
W_OPTS = np.arange(5, 101, 5)      # width [5, 10, 15, ..., 100]
D_OPTS = np.arange(3, 21, 1)       # depth [3, 4, 5, ..., 20]
A_OPTS = np.asarray([tf.nn.relu, tf.nn.sigmoid])  # activation relu or sigmoid
# map job id to parameter choice
idw, idd, ida = np.unravel_index(
    jobid-1,
    [W_OPTS.size, D_OPTS.size, A_OPTS.size]
)
    

###############################################################################
# GENERAL SETUP
###############################################################################
# general settings
DOMAIN = [0.0, 1.0, 0.0, 1.0]   # unit square [0, 1] x [0, 1]
RESOLUTION = 200                # grid resolution for testing the trained model

# network settings
WIDTH = W_OPTS[idw]            # number of neurons per hidden layer
DEPTH = D_OPTS[idd]            # number of hidden layers
ACTIVATION = A_OPTS[ida]       # hidden layer activation function

# training settings
INIT_L_RATE = 2e-1
FINAL_L_RATE = 1e-3
NUM_ITER = 100000
BATCH_SIZE = 4096

# setup target function
REGULARITY = [2, 3]
def target_func(x, y):
    smooth_part = 2*tfunc.bernstein2d(x, y, [2, 3], [3, 4])
    nonsmooth_part = 2*np.math.factorial(REGULARITY[0]) \
        * np.math.factorial(REGULARITY[1]) \
        * tfunc.signpoly2d(x-0.4, y-0.6, REGULARITY)
    return smooth_part + nonsmooth_part


###############################################################################
# DEFINE TENSORFLOW MODEL
###############################################################################
# define input and output layer
input = tf.placeholder(tf.float32, [None, 2])
target = tf.placeholder(tf.float32, [None, 1])

# define hidden layers with forward skip connections
hidden = (DEPTH-1)*[None]
hidden[0] = input
for l in range(DEPTH-2):
    if l>0:
        hidden[l+1] = tf.layers.dense(
            tf.concat(hidden[:l+1], axis=1),
            WIDTH,
            activation=ACTIVATION
        )
    else:
        hidden[l+1] = tf.layers.dense(
            hidden[l],
            WIDTH,
            activation=ACTIVATION
        )

# final layer without ReLU activation
prediction = tf.layers.dense(tf.concat(hidden, axis=1), 1, activation=None)

###############################################################################
# DEFINE TRAINING PROCEDURE
###############################################################################
# use decaying learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    INIT_L_RATE,
    global_step,
    1,
    np.exp(np.log(FINAL_L_RATE/INIT_L_RATE) / NUM_ITER),
    staircase=True
)

# L2 loss function
# (we want L_inf error, but use smooth L2 error for optimization instead)
loss = 1/2*tf.reduce_mean(tf.square(prediction-target))

# Linf error (we can however still monitor this)
error = tf.reduce_max(tf.abs(prediction-target))

# use gradient descent during optimization
step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss,
    global_step=global_step
)

###############################################################################
# PRINT OUT SOME PARAMETERS FOR BOOKKEEPING
###############################################################################
print('\n----------------------------------------------------') 
print(' RUNNING EXPERIMENT WITH THE FOLLOWING PARAMETERS: ')
print('----------------------------------------------------\n')
print('depth:\t\t\t{}'.format(DEPTH))
print('width:\t\t\t{}'.format(WIDTH))
print('number of neurons:\t{}'.format(2+(DEPTH-2)*WIDTH))
print('number of connections:\t{}'.format(
    2 + (DEPTH-2)*WIDTH*3+WIDTH*WIDTH*(DEPTH-3)*(DEPTH-2)/2
))
print('activation:\t\t{}'.format(ACTIVATION.__name__))
print('learning rate:\t\t{} to {}'.format(INIT_L_RATE, FINAL_L_RATE))
print('iterations:\t\t{}'.format(NUM_ITER))
print('batch size:\t\t{}'.format(BATCH_SIZE))
print('regularity:\t\t{}'.format(REGULARITY))
print('\n\n')

###############################################################################
# RUN THE TRAINING
###############################################################################
# start Tensorflow session and initialize all network variables
session = tf.Session(config=tf.ConfigProto(
    device_count = {'GPU': 0}
))
session.run(tf.global_variables_initializer())

# run gradient descent steps
print('\nStarted training...')
print('{:8s}\t{:8s}\t{:8s}'.format('iter', 'l2-loss', 'linf-err'))
print('{:8s}\t{:8s}\t{:8s}'.format(*(3*[8*'-'])))
for iter in range(NUM_ITER):
    # generate random batch of inputs and corresponding target values
    input_batch = [DOMAIN[1]-DOMAIN[0], DOMAIN[3]-DOMAIN[2]] \
                  * np.random.rand(BATCH_SIZE, 2) \
                  + [DOMAIN[0], DOMAIN[2]]
    target_batch = np.reshape(
        target_func(input_batch[:, 0], input_batch[:, 1]),
        [-1, 1]
    )

    # take gradient descent step and compute loss & error
    loss_val, error_val, _ = session.run(
        [loss, error, step],
        feed_dict={input: input_batch, target: target_batch}
    )
    if iter % 100 == 0:
        print('{:8d}\t{:1.2e}\t{:1.2e}'.format(iter, loss_val, error_val))
print('...finished training.\n')

###############################################################################
# EVALUATE THE TRAININED MODEL
###############################################################################
# generate full sample grid of input domain
xrange = np.linspace(DOMAIN[0], DOMAIN[1], num=RESOLUTION)
yrange = np.linspace(DOMAIN[2], DOMAIN[3], num=RESOLUTION)
xgrid, ygrid = np.meshgrid(xrange, yrange)
input_test_batch = np.stack([xgrid.flatten(), ygrid.flatten()], axis=1)

# get model predictions
prediction_test_batch = np.reshape(
    session.run(
        prediction,
        feed_dict={input: input_test_batch}
    ),
    xgrid.shape
)

# get actual target values and compare with predictions
target_test_batch = target_func(xgrid, ygrid)
l2_err = 1/2*np.mean(np.square(prediction_test_batch-target_test_batch))
linf_err = np.max(np.abs(prediction_test_batch-target_test_batch))
print(
    'Error of predictions after training, evaluated on {}x{} grid:'
    .format(RESOLUTION, RESOLUTION)
)
print('l2:\t{:1.4e}'.format(l2_err))
print('l2inf:\t{:1.4e}'.format(linf_err))
print('\n')

###############################################################################
# CLEANUP & SAVE RESULTS
###############################################################################
np.savez_compressed(
    './results/skip_network{}'.format(jobid),
    jobid=jobid,
    target=target_test_batch,
    prediction=prediction_test_batch,
    l2_error=l2_err,
    linf_error=linf_err,
    grid_resolution=RESOLUTION,
    domain=DOMAIN,
    depth=DEPTH,
    width=WIDTH,
    neurons=2+(DEPTH-2)*WIDTH,
    connections=2+(DEPTH-2)*WIDTH*3+WIDTH*WIDTH*(DEPTH-3)*(DEPTH-2)/2,
    activation=ACTIVATION.__name__,
    iterations=NUM_ITER,
    batch_size=BATCH_SIZE,
    regularity=REGULARITY,
    lrate=(INIT_L_RATE, FINAL_L_RATE),
)
session.close()
