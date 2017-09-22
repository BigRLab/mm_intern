#coding=utf-8
import random
import numpy as np

train_query_path = '../../dataset/train_query.txt'
train_doc_path = '../../dataset/train_doc.txt'
train_label_path = '../../dataset/train_label.txt'
valid_query_path = '../../dataset/valid_query.txt'
valid_doc_path = '../../dataset/valid_doc.txt'
valid_label_path = '../../dataset/valid_label.txt'
test_query_path = '../../dataset/test_query.txt'
test_doc_path = '../../dataset/test_doc.txt'
test_label_path = '../../dataset/test_label.txt'

def load_train_dataset():
    return load_dataset(train_query_path, train_doc_path, train_label_path)

def load_valid_dataset():
    return load_dataset(valid_query_path, valid_doc_path, valid_label_path)

def load_test_dataset():
    return load_dataset(test_query_path, test_doc_path, test_label_path)

def load_dataset(query_path, doc_path, label_path):
#    global train_query, train_doc, train_label, test_query, test_doc, test_label
    query_data = get_onehot_vec(query_path, sentence_length = 20)
    doc_data = get_onehot_vec(doc_path)
    label_data = get_label(label_path)
    print query_data.shape, doc_data.shape, label_data.shape
    assert query_data.shape == doc_data.shape
    assert query_data.shape[0] == label_data.shape[0]
    return query_data, doc_data, label_data

def get_onehot_vec(filepath, sentence_length=20):
    vocab = load_vocab()
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
        if len(temp) > sentence_length:
            temp = temp[:sentence_length]
        elif len(temp) < sentence_length:
            temp = temp + [0]*(sentence_length-len(temp))
        data_set.append(temp)
    fr.close()
    return np.asarray(data_set, dtype=np.int32)

def get_label(filepath):
    label = []
    fr = open(filepath)
    for row in fr.readlines():
        array = row.strip().split(' ')
        if row.strip() == '' or len(array) != 2:
            continue
        vec = []
        for i in array:
            vec.append(int(i))
        label.append(vec)
    fr.close()
    return np.asarray(label, dtype=np.int32)

def load_vocab():
    vocab_path = '../../model/vocab.txt'
    vocab = {}
    fr = open(vocab_path)
    index = 0
    for row in fr.readlines():
        word = row.strip().decode('utf8')
        if word == '':
            continue
        vocab[word] = index
        index += 1
    fr.close()
    return vocab

def generate_dataset(n_neg=10, n_rand=10, former_rate=0.8, latter_rate = 0.9):
    query = get_file_content('../../train_data/train_query.txt')
    doc = get_file_content('../../train_data/train_doc.txt')
    rand = get_file_content('../../train_data/normal.txt')
    neg_quey = get_file_content('../../train_data/false_train_query.txt')
    neg_doc = get_file_content('../../train_data/false_train_doc.txt')

    assert len(query) == len(doc)
    assert len(neg_quey) == len(neg_doc)

    random.shuffle(query)
    random.shuffle(doc)
    random.shuffle(neg_quey)
    random.shuffle(neg_doc)

    l = len(query)
    l_neg = len(neg_quey)
    l_rand = len(rand)

    query_data = []
    doc_data = []
    label = []

    for i in range(l):
        query_data.append(query[i])
        doc_data.append(doc[i])
        label.append([0,1])
        for _ in range(n_neg):
            idx = random.randint(0, l_neg-1)
            query_data.append(neg_quey[idx])
            doc_data.append(neg_doc[idx])
            label.append([1,0])
        for _ in range(n_rand):
            idx = random.randint(0, l_rand-1)
            query_data.append(query[i])
            doc_data.append(rand[idx])
            label.append([1, 0])

    sample_number = len(query_data)

    former = int(sample_number*former_rate)
    latter = int(sample_number*latter_rate)

    train_query = query_data[:former]
    train_doc = doc_data[:former]
    train_label = label[:former]

    valid_query = query_data[former:latter]
    valid_doc = doc_data[former:latter]
    valid_label = label[former:latter]

    test_query = query_data[latter:]
    test_doc = doc_data[latter:]
    test_label = label[latter:]

    write_file(train_query, train_query_path)
    write_file(train_doc, train_doc_path)
    write_file(train_label, train_label_path, True)

    write_file(valid_query, valid_query_path)
    write_file(valid_doc, valid_doc_path)
    write_file(valid_label, valid_label_path, True)

    write_file(test_query, test_query_path)
    write_file(test_doc, test_doc_path)
    write_file(test_label, test_label_path, True)

def get_file_content(filepath):
    content = []
    fr = open(filepath)
    for row in fr.readlines():
        row = row.rstrip()
        if row == '':
            continue
        content.append(row)
    fr.close()
    return content

def write_file(content, filepath, label=False):
    fw = open(filepath, 'w')
    if label:
        for row in content:
            fw.write(' '.join([str(i) for i in row]) + '\n')
    else:
        for row in content:
            fw.write(row + '\n')
    fw.close()

if __name__=="__main__":
    generate_dataset(10, 0, 0.8)