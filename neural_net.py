import random
import math
import tensorflow as tf
import numpy as np

NUM_HOURS = 12
NUM_HIDDEN = 8
NUM_OUTPUT = 4
BATCH_SIZE = 128
LEARN_RATE = .05

def normalize(temp):
    return (temp-55)/50

def normalize_array(temps_array):
    return [normalize(x) for x in temps_array[7:19]]

def gen_training_recs(temp_arr):
    total = sum(temp_arr)
    average = total/NUM_HOURS
    if average>.4: return [1,0,0,0] #75 deg F
    elif average>.2: return [0,1,0,0] #65 deg F
    elif average>-.2: return [0,0,1,0] #45 deg F
    else: return [0,0,0,1]

def fizz_buzz(i):
    return ["S/S","L/S","L/L","L/C"][i]

def init_weights(shape, var_name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=var_name)

#This does some stuff I don't relaly understand, but it does some matrix multiplication across the layers
def model(X, w_h, w_o, b_h, b_o):
    hidden_out = tf.nn.relu(tf.add(tf.matmul(X, w_h),b_h))
    return tf.add(tf.matmul(hidden_out, w_o),b_o)

def main(temps_array):
    #Prepare our training data
    normalized_temps = [normalize_array(to_normalize) for to_normalize in temps_array]
    tr_x = np.array(normalized_temps)
    tr_y = np.array([gen_training_recs(tArr) for tArr in normalized_temps])

    #Prepare tf to store our network
    X = tf.placeholder("float", [None, NUM_HOURS])
    Y = tf.placeholder("float", [None, NUM_OUTPUT])

    # Initialize the weights.
    w_h = init_weights([NUM_HOURS, NUM_HIDDEN],"w_h")
    w_o = init_weights([NUM_HIDDEN, NUM_OUTPUT],"w_o")
     # Initialize the biases.
    b_h = init_weights([1, NUM_HIDDEN],"b_h")
    b_o = init_weights([1, NUM_OUTPUT],"b_o")


    # Predict y given x using the model.
    py_x = model(X, w_h, w_o, b_h, b_o)

    #Train the model to minimize the cost function between our actual ouput (py_X) and our intended output (Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(cost)

    #Pick the largest value from our output as the recommendation
    predict_op = tf.argmax(py_x, 1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for epoch in range(100):
            # Shuffle the data before each training iteration.
            p = np.random.permutation(range(len(tr_x)))
            tr_x, tr_y = tr_x[p], tr_y[p]

            # Train in batches of 128 inputs.
            for start in range(0, len(tr_x), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={X: tr_x[start:end], Y: tr_y[start:end]})

            # And print the current accuracy on the training data.
            print(NUM_HIDDEN, epoch, np.mean(np.argmax(tr_y, axis=1) ==
                                 sess.run(predict_op, feed_dict={X: tr_x, Y: tr_y})))

        save_path = saver.save(sess, "saves/model.ckpt")
        print("Model saved in file: %s" % save_path)

def test_load(temps_array):
    #Prepare tf to store our network
    normalized_temps = np.array([normalize_array(temps_array)])
    X = tf.placeholder("float", [None, NUM_HOURS])
    Y = tf.placeholder("float", [None, NUM_OUTPUT])

    # Initialize the weights.
    w_h = init_weights([NUM_HOURS, NUM_HIDDEN],"w_h")
    w_o = init_weights([NUM_HIDDEN, NUM_OUTPUT],"w_o")
     # Initialize the biases.
    b_h = init_weights([1, NUM_HIDDEN],"b_h")
    b_o = init_weights([1, NUM_OUTPUT],"b_o")


    # Predict y given x using the model.
    py_x = model(X, w_h, w_o, b_h, b_o)

    #Train the model to minimize the cost function between our actual ouput (py_X) and our intended output (Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(cost)

    #Pick the largest value from our output as the recommendation
    predict_op = tf.argmax(py_x, 1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # Restore variables from disk.
        saver.restore(sess, "saves/model.ckpt")
        print("Model restored.")
        # Do some work with the model
        teX = np.array(normalized_temps)
        teY = sess.run(predict_op, feed_dict={X: teX})
        output = [fizz_buzz(to_fizz) for to_fizz in teY]
        print(output)