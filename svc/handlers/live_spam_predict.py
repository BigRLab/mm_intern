#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import json
import time
from core import dssm, data_provider, log
from copy import deepcopy

model_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../model/dssm.ckpt')
vocab_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../model/vocab.txt')

vocab = data_provider.load_vocab(vocab_path)

dssmModel = dssm.Model()
sess = tf.Session()
dssmModel.load_model(sess, model_path)

def get_onehot_vec(text_list, sentence_length=20):
    global vocab
    data_set = []
    for row in text_list:
        temp = []
        try:
            row = row.strip().decode('utf8')
        except:
            continue
        if row == '':
            row = u'你好'
        for word in row:
            if word in vocab:
                temp.append(vocab[word])
        if len(temp) > sentence_length:
            temp = temp[:sentence_length]
        elif len(temp) < sentence_length:
            temp = temp + [0]*(sentence_length-len(temp))
        data_set.append(temp)
    return np.asarray(data_set, dtype=np.int32)

def predict(environ, start_response):
    try:
        post_data = environ['post_data']
        querys = post_data['params']['args'][0]['querys']
        spams = post_data['params']['args'][0]['spams']
        if len(querys) == len(spams):
            querys_vec = get_onehot_vec(querys)
            spams_vec = get_onehot_vec(spams)
            try:
                prob = sess.run([dssmModel.prob], feed_dict={dssmModel.query: querys_vec, dssmModel.doc: spams_vec})
                result = prob[0][:, 1].tolist()
                log_data = {'type':'success', 'result':result, 'spams': spams, 'querys': querys}
                start_response('200 OK', [('Content-type', 'text/plain')])
                log.write_info('Response ' + json.dumps(log_data, ensure_ascii=False))
                yield json.dumps({"ec": 0, "em": "ok", "result": result})
            except Exception, e:
                result = "dssm exception: " + repr(e)
                log_data = {'type': 'exception', 'result': result}
                log.write_error('Error ' + json.dumps(log_data, ensure_ascii=False))
                start_response('200 OK', [('Content-type', 'text/plain')])
                yield json.dumps({"ec": 1, "em": "error", "result": result})
        else:
            result = "length doesn't match"
            log_data = {'type':'exception', 'result': result}
            log.write_error('Error ' + json.dumps(log_data, ensure_ascii=False))
            start_response('200 OK', [('Content-type', 'text/plain')])
            yield json.dumps({"ec": 1, "em": "error", "result": result})
    except Exception,e:
        result = 'params error'
        log_data = {'type':'exception', 'result': result}
        log.write_error('Error ' + json.dumps(log_data, ensure_ascii=False))
        start_response('200 OK', [('Content-type', 'text/plain')])
        yield json.dumps({"ec": 1, "em": "error", "result": result})