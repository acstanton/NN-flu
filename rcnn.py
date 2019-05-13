#!/usr/bin/env python                                                                               
# Allie Stanton, 2019                                                                               

# Part of the NN flu subtyping project, for 6.874. This script relates to the RCNN model for  
# classification of truncated (150bp) fragments of influenza HA genes.                      
       
# This script does a parameter sweep and outputs the parameter configuration that gives the best    
# validation loss.                                                                                

# The hyperparameters tested here are as follows:                                                   
#      number of rnn units: 32, 48, 64, 128                                                   
#      number of kernels: 32, 48, 64, 128                                                           
#      kernel size: 3, 7, 15, 31                                                                    
#      learning rate: 1e-5, 1e-3                                                                    
#      L2 lambda: 0, 1e-6, 1e-3                                                                     
# Each configuration can take between 15 minutes and 1 hour to test, so I recommend submitting     
# multiple jobs that each test a few configurations as is done here, such that three of the         
# hyperparameters are taken from command line inputs.   

import tensorflow as tf
import numpy as np
import os
import h5py
import itertools
import sys
import shutil
import matplotlib.pyplot as plt
import sklearn.metrics

print(tf.__version__)

# import data
def load_data():
    training_set = h5py.File('truncated_flu_train.h5', 'r')
    validation_set = h5py.File('truncated_flu_val.h5', 'r')
    test_set = h5py.File('truncated_flu_test.h5', 'r')

    return(training_set['features'][:], training_set['labels'][:],
           validation_set['features'][:], validation_set['labels'][:],
           test_set['features'][:], test_set['labels'][:])

x_tr, y_tr, x_va, y_va, x_te, y_te = load_data()
print("training set:", x_tr.shape)
print("validation set:", x_va.shape)
print("test set:", x_te.shape)

# construct an RCNN

def rcnn_model(x_ph, kernel_size, num_kernels, rnn_units):

    conv_filter_height = 1
    conv_filter_width = kernel_size
    in_channels = 4
    out_channels = num_kernels
    max_pool_stride = 2
    unroll_length = 75

    weights = {
        'wc1' : tf.Variable(tf.truncated_normal(
            shape = [conv_filter_height, conv_filter_width, in_channels, out_channels],
            mean = 0.0,
            stddev = np.sqrt(2 / (conv_filter_height * conv_filter_width * in_channels + out_channels)),
            dtype = tf.float32)),
        'wfc1' : tf.Variable(tf.truncated_normal(
            shape = [rnn_units, 16],
            mean = 0.0,
            stddev = np.sqrt(2 / (rnn_units + 16)),
            dtype = tf.float32)),
    }

    biases = {
        'bc1' : tf.Variable(tf.constant(0.0, shape = [out_channels])),
        'bfc1' : tf.Variable(tf.constant(0.0, shape = [16])),
    }

    # convolutional layer 1
    conv = tf.nn.conv2d(
        input = x_ph,
        filter = weights['wc1'],
        strides = [1, 1, 1, 1],
        padding = 'SAME')
    
    conv = tf.nn.bias_add(conv, biases['bc1'])
    conv = tf.nn.relu(conv)

    # max pooling layer 1
    pool = tf.nn.max_pool(
        value = conv,
        ksize = [1, 1, max_pool_stride, 1],
        strides = [1, 1, max_pool_stride, 1],
        padding = 'SAME')

    # prepare tensor for LSTM layer
    input_lstm = tf.squeeze(pool)
    input_lstm = tf.transpose(input_lstm, [1, 0, 2])
    input_lstm = tf.reshape(input_lstm, [-1, out_channels])
    
    # make time points (positions)
    split = tf.split(input_lstm, num_or_size_splits=unroll_length, axis=0)
    
    # LSTM layer
    lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_units)
    outputs, states = tf.nn.static_rnn(lstm_cell, split, dtype=tf.float32)

    # fully connected layer 2 to predict results
    z_op = tf.add(tf.matmul(outputs[-1], weights['wfc1']), biases['bfc1'])
    y_hat_op = tf.nn.softmax(z_op)
    
    weights = list(weights.values())

    return y_hat_op, z_op, weights

# L2-regularized cross entropy loss function

def rcnn_loss(z_op, y_ph, weights, l2_coefficient):
    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_ph, logits = z_op)
    loss_op = loss_op + l2_coefficient * (tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(weights[1]))
    loss_op = tf.reduce_mean(loss_op)
    return loss_op

# training loop

def training(x_train, y_train, x_val, y_val, hyperparam_config, num_epochs, batch_size,
             save_model=True, model_dir='models/best_model'): 

    tf.reset_default_graph()
    
    # define input and output placeholders
    x_ph = tf.placeholder("float", [None, 1, 150, 4], name="x_ph")
    y_ph = tf.placeholder("float", [None, 16], name="y_ph")

    # model
    y_hat_op, z_op, weights = rcnn_model(x_ph, hyperparam_config['kernel_size'], hyperparam_config['num_kernels'], hyperparam_config['rnn_units'])
    y_hat_op_named = tf.identity(y_hat_op, "y_hat_op")
    
    # loss
    loss_op = rcnn_loss(z_op, y_ph, weights, hyperparam_config['l2_lambda'])
    loss_op_named = tf.identity(loss_op, "loss_op")

    # optimizer
    optimizer_op = tf.train.RMSPropOptimizer(hyperparam_config['learning_rate'], 0.9).minimize(loss_op)

    # initialize global variables and training session
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        best_loss = float('inf')

        # loop through epochs
        for epoch in range(num_epochs):
            avg_loss = 0.0
            total_batch = int(x_train.shape[0] / batch_size)
        
            for step in range(total_batch):
                offset = (step * batch_size) % (x_train.shape[0] - batch_size)
                batch_x = x_train[offset:(offset + batch_size), :]
                batch_y = y_train[offset:(offset + batch_size)]
            
                sess.run([optimizer_op, loss_op], feed_dict = {x_ph: batch_x, y_ph: batch_y})

            val_loss = sess.run(loss_op, feed_dict = {x_ph: x_val, y_ph: y_val})

            if val_loss < best_loss:
                best_loss = val_loss
                
                if save_model == True:
                    cwd = os.getcwd()
                    path = os.path.join(cwd, model_dir)
                    shutil.rmtree(path, ignore_errors = True)

                    tf.saved_model.simple_save(
                        session = sess,
                        export_dir = path,
                        inputs={"x_ph": x_ph, "y_ph": y_ph},
                        outputs={"y_hat_op": y_hat_op, "loss_op": loss_op})

    return best_loss

# draw a subsample for hyperparameter search

def subsample_set(x, y, n):
    
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    idx = idx[0 : n]
    
    return x[idx], y[idx]

# grid search function
def grid_search(x_train, y_train, x_val, y_val, hyperparams, num_training, num_validation, num_epochs=5, batch_size=128):
    
    losses = {}
        
    keys = hyperparams.keys()
    values = (hyperparams[key] for key in keys)
    hyperparam_config_list = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    for hyperparam_config in hyperparam_config_list:

        # subsample training and validation sets
        subset_x_train, subset_y_train = subsample_set(x_train, y_train, num_training)
        subset_x_val, subset_y_val = subsample_set(x_val, y_val, num_validation)
                
        print("testing:", hyperparam_config)
                
        best_loss = training(subset_x_train, subset_y_train,
                             subset_x_val, subset_y_val,
                             hyperparam_config,
                             num_epochs,
                             batch_size,
                             save_model = False)
        
        losses[best_loss] = hyperparam_config
    
    return losses

# hyperparameter values to test
hyperparams_to_test = {
    'rnn_units' : [int(sys.argv[1])],
    'l2_lambda': [float(sys.argv[2])],
    'learning_rate' : [1e-03, 1e-05],
    'kernel_size' : [int(sys.argv[3])],
    'num_kernels' : [32, 45, 64, 128]
    }


# static hyperparameters
num_training = 100000
num_validation = 10000
batch_size = 128
num_epochs = 10

param_results = grid_search(x_tr, y_tr, x_va, y_va, hyperparams_to_test, num_training, num_validation, num_epochs, batch_size)

best_loss = min(param_results.keys())
print("best loss:", best_loss)

best_loss_config = param_results[best_loss]
print("best loss config:", best_loss_config)
