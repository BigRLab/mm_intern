import random
import numpy as np
import tensorflow as tf
import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', '/tmp/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_integer('embedding_dim', 200+200+500, 'the dim of the fusion of multi word2vec model')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate')
flags.DEFINE_integer('max_epoch', 50000, 'max train steps')
flags.DEFINE_integer('epoch_steps', 30, 'epoch step')
flags.DEFINE_integer('sentence_length', 20, 'sentence_max_word')
flags.DEFINE_boolean('use_gpu', False, 'use gpu or not')

NEG = 50
BS = 100

K2_N = 128
K3_N = 64
K4_N = 64
K_total = K2_N + K3_N + K4_N

L1_N = 400
L2_N = 120
L3_N = 400
OUT_N = 2

input_data = []

vocab_path = '../model/vocab.txt'
vocab = {}

def load_vocab():
    global vocab_path, vocab
    fr = open(vocab_path)
    index = 0
    for row in fr.readlines():
        word = row.strip().decode('utf8')
        if word == '':
            continue
        vocab[word] = index
        index += 1
    fr.close()


def read_onehot_file(filepath):
    data_set = []
    fr = open(filepath)
    for row in fr.readlines():
        temp = []
        row = row.strip()
        if row == '':
            data_set.append(temp)
            continue
        else:
            array = row.split(' ')
            onehot_vec = []
            for i in array:
                onehot_vec.append(int(i))
            temp.append(onehot_vec)
    fr.close()
    return data_set


def get_onehot_vec(filepath):
    data_set = []
    fr = open(filepath)
    for row in fr.readlines():
        temp = []
        try:
            row = row.strip().decode('utf8')
        except:
            continue
        if row == '':
            continue
        for word in row:
            if word in vocab:
                temp.append(vocab[word])
        if len(temp) > FLAGS.sentence_length:
            temp = temp[:FLAGS.sentence_length]
        elif len(temp) < FLAGS.sentence_length:
            temp = temp + [0]*(FLAGS.sentence_length-len(temp))
        data_set.append(temp)
    fr.close()
    return data_set


def load_input_data(query_path, doc_path, normal_path):
    global input_data
    query_train_data = get_onehot_vec(query_path)
    doc_train_data = get_onehot_vec(doc_path)
    normal_data = get_onehot_vec(normal_path)
    assert len(query_train_data) == len(doc_train_data)
    l = len(normal_data)
    for i in range(len(query_train_data)):
        input_data.append(query_train_data[i])
        input_data.append(doc_train_data[i])
        for _ in range(NEG):
            rand_index = random.randint(0,l-1)
            input_data.append(normal_data[rand_index])
    input_data = np.asarray(input_data, dtype=np.int32)
    print input_data.shape


def get_batch_data(step):
    global input_data
    start = step * BS * (NEG+2)
    end = (step + 1) * BS * (NEG+2)

    label = np.zeros([BS*(NEG + 1), 2])
    for i in range(BS*(NEG + 1)):
        if i % (NEG+1) == 0:
            label[i,1] = 1
        else:
            label[i,0] = 1
    return input_data[start:end, :], label


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


def conv2d(name, input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID'), b), name=name)


def full_max_pool(name, input, perm):
    conv1 = tf.transpose(input, perm=perm)
    values = tf.nn.top_k(conv1, 1, name=name).values
    conv2 = tf.transpose(values, perm=perm)
    return conv2


def norm(name, input, lsize=4):
    return tf.nn.local_response_normalization(input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

w2v = load_w2v('../model/vec.txt', 900)
load_vocab()
print len(vocab), w2v.shape[0]
assert len(vocab) == w2v.shape[0]
load_input_data('../train_data/train_query.txt','../train_data/train_doc.txt', '../train_data/normal.txt')


if FLAGS.use_gpu:
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"
with tf.device(device_name):
    with tf.name_scope('input'):
        input_batch = tf.placeholder(tf.int32, shape=[BS*(NEG+2), FLAGS.sentence_length], name='InputBatch')
        input_label = tf.placeholder(tf.float32, shape=[BS*(NEG+1), 2], name='InputLabel')

    with tf.name_scope('w2v'):
        words = tf.Variable(w2v, dtype=tf.float32, name='words')

        input_words = tf.nn.embedding_lookup(words, input_batch)

        words_out = tf.expand_dims(input_words, -1)

    with tf.name_scope('convolution_layer'):
        #conv kernel = 2
        wc1 = tf.Variable(tf.random_normal([2, FLAGS.embedding_dim, 1, K2_N]), 'wc1')
        bc1 = tf.Variable(tf.random_normal([K2_N]), 'bc1')

        conv1 = conv2d('conv1', words_out, wc1, bc1)
        pool1 = full_max_pool('pool1', conv1, [0, 3, 2, 1])

        #conv kernel = 3
        wc2 = tf.Variable(tf.random_normal([3, FLAGS.embedding_dim, 1, K3_N]), 'wc2')
        bc2 = tf.Variable(tf.random_normal([K3_N]), 'bc2')
        conv2 = conv2d('conv2', words_out, wc2, bc2)
        pool2 = full_max_pool('pool2', conv2, [0, 3, 2, 1])

        #conv kernel = 4
        wc3 = tf.Variable(tf.random_normal([4, FLAGS.embedding_dim, 1, K4_N]), 'wc3')
        bc3 = tf.Variable(tf.random_normal([K4_N]), 'bc3')
        conv3 = conv2d('conv3', words_out, wc3, bc3)
        pool3 = full_max_pool('pool3', conv3, [0, 3, 2, 1])

        pool_merge = tf.concat([pool1, pool2, pool3], 3)
#        pool_norm = tf.reshape(norm('conv_norm', pool_merge), [BS*(NEG+2), K_total])
        conv_out = tf.nn.l2_normalize(tf.nn.relu(tf.reshape(pool_merge, [BS*(NEG+2), K_total])), 1)

    with tf.name_scope('dense_layer_1'):
        l1_par_range = np.sqrt(6.0 / (K_total + L1_N))
        wd1 = tf.Variable(tf.random_uniform([K_total, L1_N], -l1_par_range, l1_par_range))
        bd1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))

        l1 = tf.matmul(conv_out, wd1) + bd1

        l1_out = tf.nn.l2_normalize(tf.nn.relu(l1), 1)

    with tf.name_scope('dense_layer_2'):
        l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        wd2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
        bd2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))

        l2 = tf.matmul(l1_out, wd2) + bd2

        l2_out = tf.nn.l2_normalize(tf.nn.relu(l2), 1)

    with tf.name_scope('negative_sampling'):
        all_case = tf.unstack(l2_out, axis=0)
        pairwise_list = []
        for i in range(BS):
            for j in range(NEG+2):
                if j > 0:
                    query_case = all_case[i*(NEG+2)]
                    doc_case = all_case[i*(NEG+2) + j]
                    pairwise_list.append(tf.concat([query_case, doc_case], 0))
        pairwise = tf.reshape(tf.concat(pairwise_list, 0),shape=[BS*(NEG+1), L2_N*2])

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

        out = tf.nn.relu(tf.matmul(hl_out, wo1) + bo1)

    with tf.name_scope('loss'):
        weight = tf.constant(np.array([[1,0],[0,NEG]], dtype=np.float32), dtype=tf.float32)
        weighted_label = tf.matmul(input_label, weight)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=weighted_label))
        pred = tf.equal(tf.argmax(tf.nn.softmax(out), 1), tf.argmax(input_label, 1))
        accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

merged = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
config.allow_soft_placement=True

saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    if os.path.exists('../model/dssm_v3.ckpt.index'):
        saver.restore(sess, '../model/dssm_v3.ckpt')
    else:
        sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    max_loss = float('INF')

    for epoch in range(FLAGS.max_epoch):
        for step in range(FLAGS.epoch_steps):
            batch_data, batch_label = get_batch_data(step)
            acc = sess.run(accuracy, feed_dict={input_batch: batch_data, input_label:batch_label})
            sess.run(train_step, feed_dict={input_batch: batch_data, input_label:batch_label})
            ls = sess.run(loss, feed_dict={input_batch: batch_data, input_label:batch_label})
            print('Epoch %d, Step %d, loss: %f accuracy: %f' % (epoch+1, step+1, ls, acc))
            if ls < max_loss:
                saver_path = saver.save(sess, "../model/dssm_v3.ckpt")
