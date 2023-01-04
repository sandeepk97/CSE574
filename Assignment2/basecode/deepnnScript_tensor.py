'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import tensorflow as tf
import numpy as np
import pickle
import time


# Create model
# Add more hidden layers to create deeper networks
# Remember to connect the final hidden layer to the out_layer
def create_multilayer_perceptron():
    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of features
    n_hidden_2 = 256  # 2nd layer number of features
    n_hidden_3 = 256  # 3rd layer number of features
    n_hidden_4 = 256  # 4th layer number of features
    n_hidden_5 = 256  # 5th layer number of features
    n_hidden_6 = 256  # 6th layer number of features
    n_hidden_7 = 256  # 7th layer number of features
    n_input = 2376  # data input
    n_classes = 2

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
        'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
        'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
        'out': tf.Variable(tf.random_normal([n_hidden_7, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'b5': tf.Variable(tf.random_normal([n_hidden_5])),
        'b6': tf.Variable(tf.random_normal([n_hidden_6])),
        'b7': tf.Variable(tf.random_normal([n_hidden_7])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])


    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    # Hidden layer with RELU activation
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
    # Hidden layer with RELU activation
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.relu(layer_6)
    # Hidden layer with RELU activation
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf.nn.relu(layer_7)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
    return out_layer,x,y

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels.T
    train_y = np.zeros(shape=(21100, 2))
    train_l = labels[0:21100]
    valid_y = np.zeros(shape=(2665, 2))
    valid_l = labels[21100:23765]
    test_y = np.zeros(shape=(2642, 2))
    test_l = labels[23765:]
    for i in range(train_y.shape[0]):
        train_y[i, train_l[i]] = 1
    for i in range(valid_y.shape[0]):
        valid_y[i, valid_l[i]] = 1
    for i in range(test_y.shape[0]):
        test_y[i, test_l[i]] = 1

    return train_x, train_y, valid_x, valid_y, test_x, test_y

# Parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 100

# Construct model
pred,x,y = create_multilayer_perceptron()

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# load data
train_features, train_labels, valid_features, valid_labels, test_features, test_labels = preprocess()
# Launch the graph
time1=time.time()
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    time2=time.time()
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_features.shape[0] / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train_features[i * batch_size: (i + 1) * batch_size], train_labels[i * batch_size: (i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
    time3=time.time()
    print("Training Time:" + str(time3-time2))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_features, y: test_labels}))
time4=time.time()
print("Total time:" + str(time4-time1))

