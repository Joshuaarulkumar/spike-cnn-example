m# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:39:30 2017

@author: Janet
"""

#matplotlib inline

from urllib.request import urlretrieve
import zipfile

import nengo
import nengo_dl
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot

mnist = input_data.read_data_sets("fashionMNIST_data/", one_hot=True)

for i in range(6):
    plt.figure()
    plt.imshow(np.reshape(mnist.train.images[i], (28, 28)))
    plt.axis('off')
    plt.title(str(np.argmax(mnist.train.labels[i])));
    
    # lif parameters
lif_neurons = nengo.LIF(tau_rc=0.01, tau_ref=0.001)

# softlif parameters (lif parameters + sigma)
softlif_neurons = nengo_dl.SoftLIFRate(tau_rc=0.01, tau_ref=0.002, sigma=0.001)

# ensemble parameters
ens_params = dict(max_rates=nengo.dists.Choice([100]), intercepts=nengo.dists.Choice([0]))

# plot some example LIF tuning curves
for neuron_type in (lif_neurons, softlif_neurons):
    with nengo.Network(seed=0) as net:
        ens = nengo.Ensemble(10, 1, neuron_type=neuron_type)

    with nengo_dl.Simulator(net) as sim:
        plt.figure()
        plt.plot(*nengo.utils.ensemble.tuning_curves(ens, sim))
        plt.xlabel("input value")
        plt.ylabel("firing rate")
        plt.title(str(neuron_type))
        
def build_network(neuron_type):
    with nengo.Network() as net:
        # we'll make all the nengo objects in the network
        # non-trainable. we could train them if we wanted, but they don't
        # add any representational power so we can save some computation
        # by ignoring them. note that this doesn't affect the internal
        # components of tensornodes, which will always be trainable or
        # non-trainable depending on the code written in the tensornode.
        nengo_dl.configure_settings(trainable=False)

        # the input node that will be used to feed in input images
        inp = nengo.Node(nengo.processes.PresentInput(mnist.test.images, 0.2))

        # add the first convolutional layer
        x = nengo_dl.tensor_layer(
            inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
            kernel_size=3)

        # apply the neural nonlinearity
        x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

        # add another convolutional layer
        x = nengo_dl.tensor_layer(
            x, tf.layers.conv2d, shape_in=(26, 26, 32),
            filters=16, kernel_size=3)
        x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)
        # apply the neural nonlinearity
        x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)
        
#       add another convolutional layer
        x = nengo_dl.tensor_layer( 
                x, tf.layers.conv2d, shape_in=(24, 24, 16),
            filters=8, kernel_size=3)
#         add a pooling layer
        x = nengo_dl.tensor_layer(
            x, tf.layers.max_pooling2d, shape_in=(22, 22, 8),
            pool_size=2, strides=2)
        # add a pooling layer
        x = nengo_dl.tensor_layer(
            x, tf.layers.max_pooling2d, shape_in=(11, 11, 8),
            pool_size=2, strides=2)
        # add a dense layer, with neural nonlinearity.
        # note that for all-to-all connections like this we can use the
        # normal nengo connection transform to implement the weights
        # (instead of using a separate tensor_layer). we'll use a
        # Glorot uniform distribution to initialize the weights.
        x, conn = nengo_dl.tensor_layer(
            x, neuron_type, **ens_params, transform=nengo_dl.dists.Glorot(),
            shape_in=(128,), return_conn=True)
        # we need to set the weights and biases to be trainable
        # (since we set the default to be trainable=False)
        # note: we used return_conn=True above so that we could access
        # the connection object for this reason.
        net.config[x].trainable = True
        net.config[conn].trainable = True

        # add a dropout layer
        x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=0.3)

        # the final 10 dimensional class output
        x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)

    return net, inp, x

# construct the network
net, inp, out = build_network(softlif_neurons)
with net:
    out_p = nengo.Probe(out)
    
# construct the simulator
minibatch_size = 50
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)

# note that we need to add the time dimension (axis 1), which has length 1
# in this case. we're also going to reduce the number of test images, just to
# speed up this example.
train_inputs = {inp: mnist.train.images[:, None, :]}
train_targets = {out_p: mnist.train.labels[:, None, :]}
test_inputs = {inp: mnist.test.images[:minibatch_size*2, None, :]}
test_targets = {out_p: mnist.test.labels[:minibatch_size*2, None, :]}

def objective(x, y):
    return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)

opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,)

def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))
            
print("error before training: %.2f%%" % sim.loss(test_inputs, test_targets,
                                                 classification_error))

do_training = True
if do_training:
    # run training
    sim.train(train_inputs, train_targets, opt, objective=objective, n_epochs=45)

    # save the parameters to file
    sim.save_params("./fashionmnist_paramsbs6")
else:
    # download pretrained weights
#    urlretrieve('https://drive.google.com/uc?export=download&id=0B6DAasV-Fri4WWp0ZFM1XzNfMjA', 'mnist_params.zip')
#    with zipfile.ZipFile("mnist_params.zip") as f:
#        f.extractall()

    # load parameters
    sim.load_params("./fashionmnist_paramsbs6")

print("error after training: %.2f%%" % sim.loss(test_inputs, test_targets,
                                                classification_error))

sim.close()

net, inp, out = build_network(lif_neurons)
with net:
    out_p = nengo.Probe(out, synapse=0.3)
    inp_p = nengo.Probe(inp)
    
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size, unroll_simulation=10)
sim.load_params("./fashionmnist_paramsbs6")

n_steps = 60
test_inputs_time = {inp: np.tile(v, (1, n_steps, 1)) for v in test_inputs.values()}
test_targets_time = {out_p: np.tile(v, (1, n_steps, 1)) for v in test_targets.values()}

print("spiking neuron error: %.2f%%" % sim.loss(test_inputs_time, test_targets_time,
                                                classification_error))
sim.run_steps(n_steps, input_feeds={inp: test_inputs_time[inp][:minibatch_size]})

for i in range(5):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.reshape(mnist.test.images[i], (28, 28)))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(sim.trange(), sim.data[out_p][i])
    plt.legend([str(i) for i in range(10)], loc="upper left")
    plt.xlabel("time")
    
sim.close()

#sim.run(13)
#
#plt.figure()
#plt.plot(sim.trange(), sim.data[out_p], label="output")
#plt.plot(sim.trange(), sim.data[inp_p], 'r', label="Input")
#plt.xlim(0, 100)
#plt.legend()
#sim.close()
# Plot the spiking output of the ensemble
#plt.figure()
#rasterplot(sim.trange(), sim.data[A_spikes])
#plt.xlim(0, 1);