import random
import numpy as np
import tensorflow as tf
import text_reformer
import sys
reload(sys)
sys.setdefaultencoding('utf8')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', '/tmp/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_integer('embedding_dim', 200+200+500, 'the dim of the fusion of multi word2vec model')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate')
flags.DEFINE_integer('max_epoch', 50000, 'max train steps')
flags.DEFINE_integer('epoch_steps', 20, 'epoch step')
flags.DEFINE_integer('sentence_length', 20, 'sentence_max_word')

NEG = 10
BS = 20

L1_N = 400
L2_N = 120

query_train_data = None
doc_train_data = None

query_test_data = None
doc_test_data = None

vocab_path = 'model/vocab.txt'
vocab = {}

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.squre(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

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
        row = row.strip().decode('utf8')
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

def load_train_data(query_path, doc_path):
    global query_train_data, doc_train_data
    query_train_data = np.asarray(get_onehot_vec(query_path), dtype=np.float32)[:400,:]
    doc_train_data = np.asarray(get_onehot_vec(doc_path), dtype=np.float32)[:400,:]
    assert query_train_data.shape[0] == doc_train_data.shape[0]

def load_test_data(query_path, doc_path):
    global query_test_data, doc_test_data
    query_test_data = np.asarray(get_onehot_vec(query_path), dtype=np.float32)[400:,:]
    doc_test_data = np.asarray(get_onehot_vec(doc_path), dtype=np.float32)[400:,:]
    assert query_test_data.shape[0] == doc_test_data.shape[0]

def get_batch_data(step):
    global query_train_data, doc_train_data
    start = step * BS
    end = step * BS + BS
    return query_train_data[start:end, :], doc_train_data[start:end, :]

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

def max_pool(name, input, k):
    return tf.nn.max_pool(input, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='VALID', name=name)

def full_max_pool(name, input, perm):
    conv1 = tf.transpose(input, perm=perm)
    values = tf.nn.top_k(conv1, 1, name=name).values
    conv2 = tf.transpose(values, perm=perm)
    return conv2

def norm(name, input, lsize=4):
    return tf.nn.local_response_normalization(input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

w2v = load_w2v('model/vec.txt', 900)
load_vocab()
print len(vocab), w2v.shape[0]
assert len(vocab) == w2v.shape[0]
load_train_data('train_data/train_query.txt','train_data/train_doc.txt')
load_test_data('train_data/train_query.txt','train_data/train_doc.txt')

with tf.name_scope('input'):
    query_batch = tf.placeholder(tf.int32, shape=[BS, FLAGS.sentence_length], name='QueryBatch')
    doc_batch = tf.placeholder(tf.int32, shape=[BS, FLAGS.sentence_length], name='DocBatch')

with tf.name_scope('w2v'):
    words = tf.Variable(w2v, dtype=tf.float32, name='words')

    query_words = tf.nn.embedding_lookup(words, query_batch)
    doc_words = tf.nn.embedding_lookup(words, doc_batch)

    query_words_out = tf.expand_dims(query_words, -1)
    doc_words_out = tf.expand_dims(doc_words, -1)

with tf.name_scope('convolution_layer'):
    #conv kernel = 2
    wc1 = tf.Variable(tf.random_normal([2, FLAGS.embedding_dim, 1, 64]), 'wc1')
    bc1 = tf.Variable(tf.random_normal([64]), 'bc1')

    query_conv1 = conv2d('conv1', query_words_out, wc1, bc1)
    query_pool1 = full_max_pool('pool1', query_conv1, [0, 3, 2, 1])
    doc_conv1 = conv2d('conv1', doc_words_out, wc1, bc1)
    doc_pool1 = full_max_pool('pool1', doc_conv1, [0, 3, 2,1])

    #conv kernel = 3
    wc2 = tf.Variable(tf.random_normal([3, FLAGS.embedding_dim, 1, 64]), 'wc2')
    bc2 = tf.Variable(tf.random_normal([64]), 'bc2')
    query_conv2 = conv2d('conv2', query_words_out, wc2, bc2)
    query_pool2 = full_max_pool('pool2', query_conv2, [0, 3, 2, 1])
    doc_conv2 = conv2d('conv2', doc_words_out, wc2, bc2)
    doc_pool2 = full_max_pool('pool2', doc_conv2, [0, 3, 2, 1])

    #conv kernel = 4
    wc3 = tf.Variable(tf.random_normal([4, FLAGS.embedding_dim, 1, 64]), 'wc3')
    bc3 = tf.Variable(tf.random_normal([64]), 'bc3')
    query_conv3 = conv2d('conv3', query_words_out, wc3, bc3)
    query_pool3 = full_max_pool('pool3', query_conv3, [0, 3, 2, 1])
    doc_conv3 = conv2d('conv3', query_words_out, wc3, bc3)
    doc_pool3 = full_max_pool('pool3', doc_conv3, [0, 3, 2, 1])

    query_pool_merge = tf.concat([query_pool1, query_pool2, query_pool3], 3)
    query_pool_norm = tf.reshape(norm('conv_norm', query_pool_merge), [BS, 64*3])

    doc_pool_merge = tf.concat([doc_pool1, doc_pool2, doc_pool3], 3)
    doc_pool_norm = tf.reshape(norm('conv_norm', doc_pool_merge), [BS, 64*3])

with tf.name_scope('dense_layer_1'):
    l1_par_range = np.sqrt(6.0 / (64*3 + L1_N))
    wd1 = tf.Variable(tf.random_uniform([64*3, L1_N], -l1_par_range, l1_par_range))
    bd1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))

    query_l1 = tf.matmul(query_pool_norm, wd1) + bd1
    doc_l1 = tf.matmul(doc_pool_norm, wd1) + bd1

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

with tf.name_scope('negative_sampling'):
    temp = tf.tile(doc_l2_out, [1,1])
    doc_y = tf.tile(doc_l2_out, [1,1])
    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat([doc_y, tf.slice(temp, [rand, 0], [BS - rand, -1]), tf.slice(temp, [0, 0], [rand, -1])], 0)

    query_x = tf.tile(query_l2_out, [NEG + 1, 1])

with tf.name_scope('hidden_layer'):
    hl1_par_range = np.sqrt(6.0 / (120*2 + 300))
    wh1 = tf.Variable(tf.random_uniform([120*2, 300], -hl1_par_range, hl1_par_range), 'wh1')
    bh1 = tf.Variable(tf.random_uniform([300], -hl1_par_range, hl1_par_range), 'bh1')

    hl = tf.matmul(tf.concat([query_x, doc_y], 1), wh1) + bh1
    hl_out = tf.nn.l2_normalize(tf.nn.relu(hl), 1)

with tf.name_scope('mlp_out'):
    out_par_range = np.sqrt(6.0 / (300 + 1))
    wo1 = tf.Variable(tf.random_uniform([300,1], -out_par_range, out_par_range), 'wo1')
    bo1 = tf.Variable(tf.random_uniform([1], -out_par_range, out_par_range), 'bo1')

    out_raw = tf.matmul(hl_out, wo1) + bo1
    out = tf.transpose(tf.reshape(tf.transpose(out_raw), [NEG + 1, BS])) * 20
    out_norm = tf.nn.l2_normalize(out, 1)

with tf.name_scope('loss'):
    np_y = np.zeros([BS, NEG+1])
    np_y[:,0] += 1
    y_ = tf.constant(np_y, dtype=tf.int32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_norm, labels=y_))
    pred = tf.equal(tf.argmax(out_norm, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    #    prob = tf.nn.softmax(out_norm)
    #    hit_prob = tf.slice(prob, [0,0], [-1,1])
    #    loss = -tf.reduce_sum(tf.log(hit_prob)) / BS

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

merged = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = False

saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(query_x)))
    exit()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    max_loss = float('INF')
    for epoch in range(FLAGS.max_epoch):
        for step in range(FLAGS.epoch_steps):
            query_batch_data, doc_batch_data = get_batch_data(step)
            acc = sess.run(accuracy, feed_dict={query_batch: query_batch_data, doc_batch: doc_batch_data})
            sess.run(train_step, feed_dict={query_batch: query_batch_data, doc_batch: doc_batch_data})
            ls = sess.run(loss, feed_dict={query_batch: query_batch_data, doc_batch: doc_batch_data})
            print('Epoch %d, Step %d, loss: %f accuracy: %f' % (epoch+1, step+1, ls, acc))
#            if ls < max_loss:
#                saver_path = saver.save(sess, "model/dssm.ckpt")
