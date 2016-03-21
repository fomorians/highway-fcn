from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

# define flags (note that Fomoro will not pass any flags by default)
flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

# define artifact directories where results from the session can be saved
model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

mnist = input_data.read_data_sets('mnist', one_hot=True)

def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b

def dense_layer(x, W_shape, b_shape, activation):
    W, b = weight_bias(W_shape, b_shape)
    return activation(tf.matmul(x, W) + b)

def highway_layer(x, size, activation, carry_bias=-1.0):
    W, b = weight_bias([size, size], [size])

    with tf.name_scope('transform_gate'):
        W_T, b_T = weight_bias([size, size], bias_init=carry_bias)

    H = activation(tf.matmul(x, W) + b, name='activation')
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
    C = tf.sub(1.0, T, name="carry_gate")

    y = tf.add(tf.mul(H, T), tf.mul(x, C), name='y') # y = (H * T) + (x * C)
    return y

with tf.Graph().as_default(), tf.Session() as sess:
    input_layer_size = 784
    hidden_layer_size = 50 # use ~71 for fully-connected (plain) layers, 50 for highway layers
    output_layer_size = 10

    x = tf.placeholder("float", [None, input_layer_size])
    y_ = tf.placeholder("float", [None, output_layer_size])

    layer_count = 20
    carry_bias_init = -2.0

    prev_y = None
    y = None
    for i in range(layer_count):
        with tf.name_scope("layer{0}".format(i)) as scope:
            if i == 0: # first, input layer
                prev_y = dense_layer(x, input_layer_size, hidden_layer_size, tf.nn.relu)
            elif i == layer_count - 1: # last, output layer
                y = dense_layer(prev_y, hidden_layer_size, output_layer_size, tf.nn.softmax)
            else: # hidden layers
                # prev_y = dense_layer(prev_y, hidden_layer_size, hidden_layer_size, tf.nn.relu)
                prev_y = highway_layer(prev_y, hidden_layer_size, tf.nn.relu, carry_bias=carry_bias_init)

    # define training and accuracy operations
    with tf.name_scope("loss") as scope:
        loss = -tf.reduce_sum(y_ * tf.log(y))
        tf.scalar_summary("loss", loss)

    with tf.name_scope("train") as scope:
        train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

    with tf.name_scope("test") as scope:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.scalar_summary('accuracy', accuracy)

    merged_summaries = tf.merge_all_summaries()

    # create a saver instance to restore from the checkpoint
    saver = tf.train.Saver(max_to_keep=1)

    # initialize our variables
    sess.run(tf.initialize_all_variables())

    # save the graph definition as a protobuf file
    tf.train.write_graph(sess.graph_def, model_path, 'highway.pb', as_text=False)

    # restore variables
    if FLAGS.restore:
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path:
            saver.restore(sess, latest_checkpoint_path)

    if not FLAGS.skip_training:
        summary_writer = tf.train.SummaryWriter(summary_path, sess.graph_def)

        num_steps = 5000
        checkpoint_interval = 100
        batch_size = 50

        step = 0
        for i in range(num_steps):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            if step % checkpoint_interval == 0:
                validation_accuracy, summary = sess.run([accuracy, merged_summaries], feed_dict={
                    x: mnist.validation.images,
                    y_: mnist.validation.labels,
                    keep_prob1: 1.0,
                    keep_prob2: 1.0,
                    keep_prob3: 1.0,
                    keep_prob4: 1.0,
                })
                summary_writer.add_summary(summary, step)
                saver.save(sess, checkpoint_path + 'checkpoint', global_step=step)
                print('step %d, training accuracy %g' % (step, validation_accuracy))

            sess.run(train_step, feed_dict={
                x: batch_xs,
                y_: batch_ys,
                keep_prob1: 0.8,
                keep_prob2: 0.7,
                keep_prob3: 0.6,
                keep_prob4: 0.5,
            })

            step += 1

        summary_writer.close()

    test_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        keep_prob1: 1.0,
        keep_prob2: 1.0,
        keep_prob3: 1.0,
        keep_prob4: 1.0,
    })
    print('test accuracy %g' % test_accuracy)
