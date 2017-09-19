import os
import random
import sys
import numpy as np
import tensorflow as tf
import data_provider

reload(sys)
sys.setdefaultencoding('utf8')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', '/tmp/dssm-v5', 'Summaries directory')
flags.DEFINE_integer('embedding_dim', 200+200+500, 'the dim of the fusion of multi word2vec model')
# flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate')
flags.DEFINE_integer('max_epoch', 50000, 'max train steps')
flags.DEFINE_integer('sentence_length', 20, 'sentence max word')
flags.DEFINE_boolean('use_gpu', False, 'use gpu or not')
flags.DEFINE_string('model_version', 'v5', 'model version')

SAMPLE = 10+1
BS = SAMPLE*200
learning_rate = 0.1
learning_rate_decay_factor = 0.95

L1_N = 1000
L2_N = 240
L3_N = 400
OUT_N = 2

conv_kernel_size = [2, 3, 4]
conv_kernel_number = [128, 64, 64]

train_query, train_doc, train_label = data_provider.load_train_dataset()
valid_query, valid_doc, valid_label = data_provider.load_valid_dataset()
test_query, test_doc, test_label = data_provider.load_test_dataset()

def get_train_batch_data(step):
    global train_query, train_doc, train_label
    start = step * BS
    end = (step + 1) * BS
    return train_query[start:end, :], train_doc[start:end, :], train_label[start:end, :]

def get_test_data():
    global test_query, test_doc, test_label
    return test_query, test_doc, test_label

def get_valid_data():
    global valid_query, valid_doc, valid_label
    return valid_query, valid_doc, valid_label

def load_w2v(path, expectDim):
    fp = open(path, "r")
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == expectDim)
    ws = []
    for t in range(total):
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == dim)
        vals = []
        for i in range(0, dim):
            fv = float(ss[i])
            vals.append(fv)
        ws.append(vals)
    fp.close()
    return np.asarray(ws, dtype=np.float32)

def biclass_rate(label, pred):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(label.shape[0]):
        if label[i] == 1:
            if pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred[i] == 0:
                tn += 1
            else:
                fp += 1
    p_precision = tp / (tp + fp + 0.0001)
    p_recall = tp / (tp + fn + 0.0001)
    n_precision = tn / (tn + fn + 0.0001)
    n_recall = tn / (tn + fp + 0.0001)

    return p_precision, p_recall, n_precision, n_recall

def conv2d(name, input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID'), b), name=name)


def full_max_pool(name, input, perm):
    conv1 = tf.transpose(input, perm=perm)
    values = tf.nn.top_k(conv1, 1, name=name).values
    conv2 = tf.transpose(values, perm=perm)
    return conv2


def norm(name, input, lsize=4):
    return tf.nn.local_response_normalization(input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


if FLAGS.use_gpu:
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"
with tf.device(device_name):
    with tf.name_scope('input'):
        query = tf.placeholder(tf.int32, shape=[None, FLAGS.sentence_length], name='QueryData')
        doc = tf.placeholder(tf.int32, shape=[None, FLAGS.sentence_length], name="DocData")
        label = tf.placeholder(tf.float32, shape=[None, 2], name='Label')

    with tf.name_scope('w2v'):
        words = tf.Variable(load_w2v('../model/vec.txt', 900), dtype=tf.float32, name='words')

        query_words = tf.nn.embedding_lookup(words, query)
        doc_words = tf.nn.embedding_lookup(words, doc)

        query_words_out = tf.expand_dims(query_words, -1)
        doc_words_out = tf.expand_dims(doc_words, -1)

    with tf.name_scope('convolution_layer'):
        wc = {}
        bc = {}
        query_conv = {}
        query_pool = {}
        query_pool_list = []
        doc_conv = {}
        doc_pool = {}
        doc_pool_list = []
        for i, size in enumerate(conv_kernel_size):
            #conv kernel size = i
            wc[size] = tf.Variable(tf.random_normal([size, FLAGS.embedding_dim, 1, conv_kernel_number[i]]), 'wc' + str(size))
            bc[size] = tf.Variable(tf.random_normal([conv_kernel_number[i]]), 'bc' + str(size))

            query_conv[size] = conv2d('conv' + str(size), query_words_out, wc[size], bc[size])
            query_pool[size] = full_max_pool('pool' + str(size), query_conv[size], [0, 3, 2, 1])
            query_pool_list.append(query_pool[size])

            doc_conv[size] = conv2d('conv' + str(size), query_words_out, wc[size], bc[size])
            doc_pool[size] = full_max_pool('pool' + str(size), query_conv[size], [0, 3, 2, 1])
            doc_pool_list.append(doc_pool[size])

        query_pool_merge = tf.concat(query_pool_list, 3)
        query_conv_out = tf.nn.l2_normalize(tf.nn.relu(tf.reshape(query_pool_merge, [-1, sum(conv_kernel_number)])), 1)

        doc_pool_merge = tf.concat(doc_pool_list, 3)
        doc_conv_out = tf.nn.l2_normalize(tf.nn.relu(tf.reshape(doc_pool_merge, [-1, sum(conv_kernel_number)])), 1)

    with tf.name_scope('dense_layer_1'):
        l1_par_range = np.sqrt(6.0 / (sum(conv_kernel_number) + L1_N))
        wd1 = tf.Variable(tf.random_uniform([sum(conv_kernel_number), L1_N], -l1_par_range, l1_par_range))
        bd1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))

        query_l1 = tf.matmul(query_conv_out, wd1) + bd1
        doc_l1 = tf.matmul(doc_conv_out, wd1) + bd1

        query_l1_out = tf.nn.l2_normalize(tf.nn.relu(query_l1), 1)
        doc_l1_out = tf.nn.l2_normalize(tf.nn.relu(doc_l1), 1)

    with tf.name_scope('dense_layer_2'):
        l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        wd2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
        bd2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))

        query_l2 = tf.matmul(query_l1_out, wd2) + bd2
        doc_l2 = tf.matmul(doc_l1_out, wd2) + bd2

        query_l2_out = tf.nn.l2_normalize(tf.nn.relu(query_l2), 1)
        doc_l2_out = tf.nn.l2_normalize(tf.nn.relu(doc_l2), 1)

    with tf.name_scope('merge_query_doc'):
        pairwise = tf.concat([query_l2_out, doc_l2_out], axis=1)

    with tf.name_scope('hidden_layer'):
        hl1_par_range = np.sqrt(6.0 / (L2_N*2 + L3_N))
        wh1 = tf.Variable(tf.random_uniform([L2_N*2, L3_N], -hl1_par_range, hl1_par_range), 'wh1')
        bh1 = tf.Variable(tf.random_uniform([L3_N], -hl1_par_range, hl1_par_range), 'bh1')

        hl = tf.matmul(pairwise, wh1) + bh1
        hl_out = tf.nn.l2_normalize(tf.nn.relu(hl), 1)

    with tf.name_scope('mlp_out'):
        out_par_range = np.sqrt(6.0 / (L3_N + OUT_N))
        wo1 = tf.Variable(tf.random_uniform([L3_N,OUT_N], -out_par_range, out_par_range), 'wo1')
        bo1 = tf.Variable(tf.random_uniform([OUT_N], -out_par_range, out_par_range), 'bo1')

        out = tf.matmul(hl_out, wo1) + bo1

    with tf.name_scope('loss'):
        out_softmax = tf.nn.softmax(out)
        pred_y = tf.argmax(tf.nn.softmax(out), 1)
        label_y = tf.argmax(label, 1)
        pred = tf.equal(pred_y, label_y)
        accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=label))

    with tf.name_scope('train'):
        learning_rate = tf.Variable(float(learning_rate), trainable=False)
        learning_rate_decay_op = learning_rate.assign(learning_rate * learning_rate_decay_factor)
#        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
config.allow_soft_placement=True

saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    if os.path.exists('../model/dssm_' + FLAGS.model_version + '.ckpt.index'):
        saver.restore(sess, '../model/dssm_' + FLAGS.model_version + '.ckpt')
    else:
        sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    max_loss = float('INF')
    epoch_steps = train_query.shape[0]/BS
    previous_losses = []
    for epoch in range(FLAGS.max_epoch):
        for step in range(epoch_steps):
            query_batch, doc_batch, label_batch = get_train_batch_data(step)
            sess.run(train_step, feed_dict={query: query_batch, doc:doc_batch, label:label_batch})
            mgd, ls, acc, pred_, label_, l_rate = sess.run([merged, loss, accuracy,pred_y, label_y, learning_rate], feed_dict={query: query_batch, doc:doc_batch, label:label_batch})
            p_p, p_r, n_p, n_r = biclass_rate(label_, pred_)
            train_writer.add_summary(mgd, epoch*epoch_steps + step)
            print('Train Epoch %d, Step %d, l_rate: %f, loss: %f, accuracy: %f, p_precision: %f, p_recall: %f, n_precision: %f, n_recall: %f' % (epoch+1, step+1, l_rate, ls, acc, p_p, p_r, n_p,  n_r))
            sys.stdout.flush()

            if step % 5 == 0:
                valid_query_batch, valid_doc_batch, valid_label_batch = get_valid_data()
                ls, acc, pred_, label_ = sess.run([loss, accuracy, pred_y, label_y], feed_dict={query: valid_query_batch, doc: valid_doc_batch, label: valid_label_batch})
                p_p, p_r, n_p, n_r = biclass_rate(label_, pred_)
                if len(previous_losses) >= 5 and ls > max(previous_losses[-5:]):
                    sess.run(learning_rate_decay_op)
                    print("Learning rate decay Epoch %d, Step %d , learning rate %f" % (
                    epoch + 1, step + 1, learning_rate.eval()))
                previous_losses.append(ls)
                print('Valid Epoch %d, loss: %f, accuracy: %f, p_precision: %f, p_recall: %f, n_precision: %f, n_recall: %f' % (epoch + 1, ls, acc, p_p, p_r, n_p,  n_r))
                sys.stdout.flush()
                if ls < max_loss:
                    saver_path = saver.save(sess, '../model/dssm_' + FLAGS.model_version + '.ckpt')

        test_query_batch, test_doc_batch, test_label_batch = get_test_data()
        ls, acc, pred_, label_ = sess.run([loss, accuracy, pred_y, label_y], feed_dict={query: test_query_batch, doc: test_doc_batch, label: test_label_batch})
        p_p, p_r, n_p, n_r = biclass_rate(label_, pred_)
        print "--------------------------------------------"
        print('Test Epoch %d, loss: %f, accuracy: %f, p_precision: %f, p_recall: %f, n_precision: %f, n_recall: %f' % (epoch + 1, ls, acc, p_p, p_r, n_p, n_r))
        print "--------------------------------------------"
        sys.stdout.flush()
