#!/usr/bin/env python                                                                               
# Allie Stanton, 2019                                                                               

# Part of the NN flu subtyping project, for 6.874. This script relates to the CNN model for         
# classification of truncated (150bp) fragments of influenza HA genes.                      

# This script trains the network on the complete training set using the hyperparameters input by   
# the user (according to the best configuration found in parameter sweep).                          

# It also outputs several performance metric including some graphs of ROC and precision-recall   
# curves as well as a confusion matrix. It can take up to 12 hours to run.

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

######################################################################
#
#             Input hyperparameters
#
######################################################################

best_hyperparams = {
    'dropout_rate' : 0.2,
    'l2_lambda' : 0,
    'learning_rate' : 1e-03,
    'kernel_size' : 31,
    'num_kernels' : 128,
    'ratio_fc_neurons' : 0.67
    }

num_epochs = 10
batch_size = 128

######################################################################

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

# construct a CNN

def cnn_model(x_ph, dropout_rate, kernel_size, num_kernels, ratio_fc_neurons):
    
    conv_filter_height = 1
    conv_filter_width = kernel_size
    in_channels = 4
    out_channels = num_kernels
    fc_units = round(150 * ratio_fc_neurons)
    max_pool_stride = 2

    weights = {
        'wc1' : tf.Variable(tf.truncated_normal(
            shape = [conv_filter_height, conv_filter_width, in_channels, out_channels],
            mean = 0.0,
            stddev = np.sqrt(2 / (conv_filter_height * conv_filter_width * in_channels + out_channels)),
            dtype = tf.float32)),
        'wfc1' : tf.Variable(tf.truncated_normal(
            shape = [1 * 75 * out_channels, fc_units],
            mean = 0.0,
            stddev = np.sqrt(2 / (out_channels + fc_units)),
            dtype = tf.float32)),
        'wfc2' : tf.Variable(tf.truncated_normal(
            shape = [fc_units, 16],
            mean = 0.0,
            stddev = np.sqrt(2 / (fc_units + 16)),
            dtype = tf.float32))
    }

    biases = {
        'bc1' : tf.Variable(tf.constant(0.0, shape = [out_channels])),
        'bfc1' : tf.Variable(tf.constant(0.0, shape = [fc_units])),
        'bfc2' : tf.Variable(tf.constant(0.0, shape = [16]))
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

    # flatten tensor to prepare for fully connected layer
    fcon1 = tf.reshape(pool, [-1, weights['wfc1'].get_shape().as_list()[0]])

    # construct fully connected layer 1
    fcon1 = tf.add(tf.matmul(fcon1, weights['wfc1']), biases['bfc1'])
    fcon1 = tf.nn.relu(fcon1)

    # dropout
    fcon1 = tf.nn.dropout(fcon1, rate = dropout_rate)

    # fully connected layer 2 to predict results
    z_op = tf.add(tf.matmul(fcon1, weights['wfc2']), biases['bfc2'])
    y_hat_op = tf.nn.softmax(z_op)
    
    weights = list(weights.values())

    return y_hat_op, z_op, weights

# L2-regularized cross entropy loss function

def cnn_loss(z_op, y_ph, weights, l2_coefficient):
    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_ph, logits = z_op)
    loss_op = loss_op + l2_coefficient * (tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(weights[1]) + tf.nn.l2_loss(weights[2]))
    loss_op = tf.reduce_mean(loss_op)
    return loss_op

# training loop

def training(x_train, y_train, x_val, y_val, hyperparam_config, num_epochs, batch_size,
             save_model=True, model_dir='models/truncated_cnn/best_model'): 

    tf.reset_default_graph()
    
    # define input and output placeholders
    x_ph = tf.placeholder("float", [None, 1, 150, 4], name="x_ph")
    y_ph = tf.placeholder("float", [None, 16], name="y_ph")

    # model
    y_hat_op, z_op, weights = cnn_model(x_ph, hyperparam_config['dropout_rate'], hyperparam_config['kernel_size'], hyperparam_config['num_kernels'], hyperparam_config['ratio_fc_neurons'])
    y_hat_op_named = tf.identity(y_hat_op, "y_hat_op")
    
    # loss
    loss_op = cnn_loss(z_op, y_ph, weights, hyperparam_config['l2_lambda'])
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
            print("training epoch", epoch)
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

best_model_dir = "models/truncated_cnn/best_model"

# subsampling function for validation set

def subsample_set(x, y, n):

    idx = np.arange(len(x))
    np.random.shuffle(idx)
    idx = idx[0 : n]

    return x[idx], y[idx]

x_val_sample, y_val_sample = subsample_set(x_va, y_va, 10000)

validation_loss = training(x_tr, y_tr, x_val_sample, y_val_sample, best_hyperparams, num_epochs, batch_size, save_model=True, model_dir=best_model_dir)

print("validation loss:", validation_loss)

def predict(x, model_dir):
    
    graph2 = tf.Graph()
    
    cwd = os.getcwd()
    path = os.path.join(cwd, model_dir)
    
    with tf.Session(graph = graph2) as sess:        
        tf.saved_model.loader.load(
            sess,
            ["serve"],
            export_dir = path
        )

        x_ph = graph2.get_tensor_by_name('x_ph:0')
        y_hat_op = graph2.get_tensor_by_name('y_hat_op:0')
        y_hat = sess.run(y_hat_op, feed_dict = {x_ph: x})

    return y_hat

y_hat_test = predict(x_te, best_model_dir)

def evaluate(x_test, y_test, y_hat):
    
    n_predictions = x_test.shape[0]

    actual = np.argmax(y_test, axis=1)
    predict = np.argmax(y_hat, axis=1)
    
    equal_bool = np.equal(actual, predict)
    n_correct = np.sum(equal_bool)
    
    accuracy = n_correct / n_predictions

    return accuracy

acc = evaluate(x_te, y_te, y_hat_test)

print("Accuracy:", acc)

def plot_roc_curve(y, y_hat):
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(16):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y[:, i], y_hat[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y.ravel(), y_hat.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(16)]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(16):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= 16
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])
    
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = itertools.cycle(['lightcoral', 'lightsalmon', 'peachpuff', 'navajowhite', 'palegoldenrod',
                    'palegreen', 'mediumspringgreen', 'mediumaquamarine', 'turquoise', 'powderblue',
                    'lightskyblue', 'lightsteelblue', 'cornflowerblue', 'thistle', 'orchid', 'palevioletred'])

    for i, color in zip(range(16), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='H{0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
               prop=dict(size=10), mode="expand", borderaxespad=0, ncol=1)
    plt.show()
    plt.savefig("truncated_ROC.png", bbox_inches='tight')

plot_roc_curve(y_te, y_hat_test)

def plot_pr_curve(y, y_hat):
    precision = {}
    recall = {}
    average_precision = {}
    
    for i in range(16):
        precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y[:, i], y_hat[:, i])
        average_precision[i] = sklearn.metrics.average_precision_score(y[:, i], y_hat[:, i])
        
    precision["micro"], recall["micro"], _ = sklearn.metrics.precision_recall_curve(y.ravel(), y_hat.ravel())
    average_precision["micro"] = sklearn.metrics.average_precision_score(y, y_hat, average="micro")
    
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all subtypes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.savefig('truncated_microavg_pr.png')

    colors = itertools.cycle(['lightcoral', 'lightsalmon', 'peachpuff', 'navajowhite', 'palegoldenrod',
                    'palegreen', 'mediumspringgreen', 'mediumaquamarine', 'turquoise', 'powderblue',
                    'lightskyblue', 'lightsteelblue', 'cornflowerblue', 'thistle', 'orchid', 'palevioletred'])
    
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(16), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('H{0} (area = {1:0.2f})'
                      ''.format(i+1, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(lines, labels, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
               prop=dict(size=10), mode="expand", borderaxespad=0, ncol=1)
    plt.savefig('truncated_multiclass_pr.png', bbox_inches='tight')

plot_pr_curve(y_te, y_hat_test)

def plot_confusion_matrix(y, y_hat):
    # Get labels
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(y_hat, axis=1)

    # Compute confusion matrix                                                                                         
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    class_labels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H\
16']

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...                                                                                     
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries                                                       
           xticklabels=class_labels, yticklabels=class_labels,
           ylabel='True subtype',
           xlabel='Predicted subtype')

    # Rotate the tick labels and set their alignment.                                                                  
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.                                                           
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", size=6,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.title("Normalized confusion matrix for truncated HA classification")
    plt.savefig("truncated_confusion.png")

plot_confusion_matrix(y_te, y_hat_test)
