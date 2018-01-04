import cPickle
import tensorflow as tf
import os
import numpy as np
import random

# tensorboard --logdir=path/to/log-directory

CIFAR_DATA_DIR = 'cifar-10-files/'

LOG_PATH = 'logs/'

NUM_OF_CLASSES = 10

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3

IMAGE_DISTORTION_SIZE = 24

IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH

NUM_EXAMPLES_FOR_TRAINING = 50000
NUM_EXAMPLES_FOR_TESTING = 10000

CONV1_FILTER_SIZE = 5
CONV1_NUM_FILTERS = 64
CONV1_NUM_INPUT_CHANNELS = 3

CONV2_FILTER_SIZE = 5
CONV2_NUM_FILTERS = 64
CONV2_NUM_INPUT_CHANNELS = 64

FC_SIZE = 128

BATCH_SIZE = 64
INPUT_CHANNELS = 3


def read_files(path):
    print("Loading data for path: " + path)
    with open(path, 'rb') as file:
        data = cPickle.load(file)
    return data


def get_files_to_read(train_session=True):
    cifar_data_files = list()
    if train_session:
        for root, dirs, files in os.walk(CIFAR_DATA_DIR):
            for f in files:
                if 'data' in f:
                    cifar_data_files.append(
                        CIFAR_DATA_DIR + f)
    else:
        cifar_data_files = [os.path.join(CIFAR_DATA_DIR, 'test_batch')]
    return cifar_data_files


def load_data_for_training():
    images = np.zeros(
        shape=[NUM_EXAMPLES_FOR_TRAINING, IMAGE_HEIGHT, IMAGE_WIDTH,
               IMAGE_DEPTH], dtype=float)
    classes = np.zeros(shape=[NUM_EXAMPLES_FOR_TRAINING], dtype=int)
    files_read = get_files_to_read(train_session=True)
    begin = 0
    for file in files_read:
        data = read_files(file)
        raw_images = data[b'data']
        raw_cls = np.array(data[b'labels'])
        raw_images = np.array(raw_images, dtype=float) / 255.0
        processed_images = raw_images.reshape([-1, IMAGE_DEPTH, IMAGE_HEIGHT,
                                               IMAGE_WIDTH])
        processed_images = processed_images.transpose([0, 2, 3, 1])
        num_images = len(processed_images)
        end = begin + num_images
        images[begin:end, :] = processed_images
        classes[begin:end] = raw_cls
        begin = end
        one_hot = tf.one_hot(indices=classes, depth=10, on_value=1.0,
                             off_value=0.0, axis=-1)
        with tf.Session() as sess:
            label_numpy = one_hot.eval(session=sess)
            sess.close()
    return images, classes, label_numpy


def load_data_for_eval():
    files_read = get_files_to_read(train_session=False)
    data = read_files(files_read[0])
    raw_images = data[b'data']
    raw_cls = np.array(data[b'labels'])
    raw_images = np.array(raw_images, dtype=float) / 255.0
    processed_images = raw_images.reshape([-1, IMAGE_DEPTH, IMAGE_HEIGHT,
                                           IMAGE_WIDTH])
    processed_images = processed_images.transpose([0, 2, 3, 1])
    return processed_images, raw_cls, tf.one_hot(raw_cls, NUM_OF_CLASSES)


def load_class_names():
    raw = read_files(CIFAR_DATA_DIR + "batches.meta")[b'label_names']
    names = [x.decode('utf-8') for x in raw]
    return names


image_count = 0

images_for_training, classes_for_training, labels_train = \
    load_data_for_training()

images_for_eval, classes_for_eval, labels_eval = load_data_for_eval()

class_names_of_images = load_class_names()

x_input = tf.placeholder(tf.float32,
                         shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
                         name='x-input')

y_true_output = tf.placeholder(tf.float32, shape=[None, NUM_OF_CLASSES],
                               name='y_true_output')

y_true_cls = tf.argmax(y_true_output, dimension=1, name='y_true_cls')


def pre_processing_images(image, train_session=True):
    if train_session:
        distort_image = tf.random_crop(image, size=[IMAGE_DISTORTION_SIZE,
                                                    IMAGE_DISTORTION_SIZE,
                                                    IMAGE_DEPTH])
        distort_image = tf.image.random_flip_left_right(distort_image)
        if random.choice([0, 1]) == 0:
            distort_image = tf.image.random_hue(distort_image, max_delta=0.05)
            distort_image = tf.image.random_brightness(distort_image,
                                                       max_delta=0.2)
            distort_image = tf.image.random_contrast(distort_image, lower=0.3,
                                                     upper=1.0)
            distort_image = tf.image.random_saturation(distort_image, lower=0.0,
                                                       upper=2.0)
        else:
            distort_image = tf.image.random_hue(distort_image, max_delta=0.05)
            distort_image = tf.image.random_contrast(distort_image, lower=0.3,
                                                     upper=1.0)
            distort_image = tf.image.random_brightness(distort_image,
                                                       max_delta=0.2)
            distort_image = tf.image.random_saturation(distort_image, lower=0.0,
                                                       upper=2.0)

            distort_image = tf.minimum(distort_image, 1.0)
            distort_image = tf.maximum(distort_image, 0.0)

    else:
        distort_image = tf.image.resize_image_with_crop_or_pad \
            (image, target_height=IMAGE_DISTORTION_SIZE,
             target_width=IMAGE_DISTORTION_SIZE)
    return distort_image


def pre_process_image(images, training):
    images = tf.map_fn(lambda image: pre_processing_images(image, training),
                       images)
    return images


def generate_weights(shape, name, stddev=0.05):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    tf.summary.histogram(name, weight)
    return weight


def generate_biases(shape, name, initial_value=0.05):
    bias = tf.Variable(tf.constant(initial_value, shape=shape))
    tf.summary.histogram(name, bias)
    return bias


def new_conv_layer(input, num_input_channels, filter_size, num_filters, name,
                   use_pooling=True):
    with tf.name_scope(name):
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = generate_weights(shape, name=name + "_weights")
        biases = generate_biases([num_filters], name=name + "_bias")
        conv_layer = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='SAME',
                                  name=name)
        conv_layer = tf.add(conv_layer, biases)
        if use_pooling:
            conv_layer = tf.nn.max_pool(value=conv_layer,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME')
        conv_layer_relu = tf.nn.relu(conv_layer)
    return conv_layer_relu


def new_fully_connected_layer(input, num_input, num_output, name, use_relu=True,
                              stddev=0.05, biases_value=0.05):
    with tf.name_scope(name):
        weights = generate_weights([num_input, num_output], stddev=stddev,
                                   name=name + "_weights")
        biases = generate_biases([num_output], initial_value=biases_value,
                                 name=name + "_bias")
        fc_layer = tf.matmul(input, weights, name=name)
        fc_layer = tf.add(fc_layer, biases, name=name)
        if use_relu:
            fc_layer = tf.nn.relu(fc_layer)

    return fc_layer


def network(input):
    with tf.variable_scope('network', reuse=not True):
        conv_layer_1 = new_conv_layer(input=input,
                                      num_input_channels=INPUT_CHANNELS,
                                      filter_size=CONV1_FILTER_SIZE,
                                      num_filters=CONV1_NUM_FILTERS,
                                      use_pooling=True,
                                      name='Conv_layer_1')

        conv_layer_2 = new_conv_layer(conv_layer_1,
                                      num_input_channels=CONV1_NUM_FILTERS,
                                      filter_size=CONV2_FILTER_SIZE,
                                      num_filters=CONV2_FILTER_SIZE,
                                      use_pooling=True,
                                      name='Conv_layer_2')

        layer_shape = conv_layer_2.get_shape()

        num_features = layer_shape[1:4].num_elements()

        layer_flat = tf.reshape(conv_layer_2, [-1, num_features])

        fc_layer_1 = new_fully_connected_layer(layer_flat, num_features,
                                               FC_SIZE,
                                               use_relu=True,
                                               name='FC_layer_1')

        fc_layer_2 = new_fully_connected_layer(fc_layer_1,
                                               FC_SIZE,
                                               NUM_OF_CLASSES,
                                               use_relu=False,
                                               name='FC_layer_2')

    return fc_layer_2


def random_batch():
    index = np.random.choice(len(images_for_training),
                             size=train_batch_size,
                             replace=False)
    x_batch_set = images_for_training[index, :, :, :]
    y_batch_set = labels_train[index, :]

    return x_batch_set, y_batch_set


images = x_input

tf.summary.image('image', images)

images = pre_process_image(images, True)

fc_layer_2_output = network(images)

global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

y_pred_output = tf.nn.softmax(fc_layer_2_output)

y_pred_cls = tf.argmax(y_pred_output, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=fc_layer_2_output,
    labels=y_true_output,
    name='cross_entropy')

loss = tf.reduce_mean(cross_entropy, name='loss')

net_work_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

network_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cost_value", loss)

tf.summary.scalar("accuracy", network_accuracy)

tf.summary.histogram('histogram_cross_entropy', cross_entropy)

tf.summary.histogram('histogram_cost', loss)

tf.summary.histogram('histogram_accuracy', network_accuracy)

session = tf.Session()

merged = tf.summary.merge_all()

session.run(tf.global_variables_initializer())

log_writer = tf.summary.FileWriter(LOG_PATH, session.graph)

train_batch_size = BATCH_SIZE

total_iterations = 0

saver = tf.train.Saver()

session = tf.Session()

save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'cifar10_cnn')

try:
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(session, save_path=last_chk_path)
    print("Restored checkpoint:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())


## The below part of the code is used for training the network and 
## displaying the accuracy every 100 epochs and saves a check point 
## every 10000 epochs

def train_netwrok(num_iterations):
    for i in range(num_iterations):

        x_batch, y_true_batch = random_batch()

        feed_dict_train = {x_input: x_batch,
                           y_true_output: y_true_batch}

        session.run(net_work_optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            summary, acc = session.run([merged, network_accuracy],
                                       feed_dict=feed_dict_train)
            log_writer.add_summary(summary, i)

            print "Optimization Iteration: {0}, Training Accuracy: {" \
                  "1:>6.1%}".format(i + 1, acc)

        if (i % 1000 == 1) or (i == num_iterations - 1):
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print "Saved checkpoint."


train_netwrok(100000)

## The below part of the code is used for testing the accuracy of the 
## network

batch_size = 512


def predict_class(images, labels, cls_true):
    cls_pred = np.zeros(shape=len(images), dtype=np.int)

    i = 0

    with tf.Session() as sess:
        label_numpy = labels.eval(session=sess)
        sess.close()

    while i < len(images):
        j = min(i + batch_size, len(images))

        feed_dict = {x_input: images[i:j, :],
                     y_true_output: label_numpy[i:j, :]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred


def print_eval_accuracy():
    correct, cls_prediciton = predict_class(images=images_for_eval,
                                            labels=labels_eval,
                                            cls_true=classes_for_eval)

    print "Accuracy on Test-Set: {0:.1%} ({1} / {2})".format(correct.mean(),
                                                             correct.sum(),
                                                             len(correct))


print_eval_accuracy()
