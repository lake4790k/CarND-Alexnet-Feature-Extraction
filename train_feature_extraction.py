import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
from tqdm import tqdm
import os
from urllib.request import urlretrieve

# TODO: Load traffic signs data.
sign_names = pd.read_csv('signnames.csv')
nb_classes = 43


def downloadIfMissing(url, file):
    if not os.path.isfile(file):
        print('downloading', file)
        urlretrieve(url, file)

url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p"
file = 'train.p'
downloadIfMissing(url, file)
url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy"
file = 'bvlc-alexnet.npy'
downloadIfMissing(url, file)

from alexnet import AlexNet

with open("train.p", mode='rb') as f:
    train = pickle.load(f)
X, Y = train['features'], train['labels']

#X = X[0:200, :, :, :]
#Y = Y[0:200]

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.25, random_state=11)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
w = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, w, b)
# probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, nb_classes)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_operation = optimizer.minimize(loss, var_list=[w, b])

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

BATCH_SIZE = 128


def evaluate(sess, x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    with tqdm(total=(num_examples // BATCH_SIZE)) as pbar:
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = x_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y })
            total_accuracy += (accuracy * len(batch_x))
            pbar.update(1)
    return total_accuracy / num_examples


EPOCHS = 100
best_acc = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, './lenet')
    num_examples = len(X_train)

    for i in range(EPOCHS):
        X_prep, y_prep = shuffle(X_train, y_train)
        with tqdm(total= (num_examples // BATCH_SIZE)) as pbar:
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={ x: batch_x, y: batch_y })
                pbar.update(1)

        validation_accuracy = evaluate(sess, X_test, y_test)
        best = ''
        if validation_accuracy > best_acc:
            best_acc = validation_accuracy
            # saver.save(sess, './alex.best')
            best = '*'

        print("EPOCH {} ValAcc = {:.3f} {}".format(i + 1, validation_accuracy, best))
