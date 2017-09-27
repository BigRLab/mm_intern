#coding=utf-8
import pypinyin
import word_part
import numpy as np
import pinyin_utility
pypinyin.load_single_dict(pinyin_dict={0x55ef: u"en"})
pypinyin.load_phrases_dict(phrases_dict={u'嗯哪': [[u'en'], [u'na']], u'人生何处不相逢':[[u'ren'], [u'sheng'], [u'he'], [u'chu'], [u'bu'], [u'xiang'], [u'feng']]})
pinyin_part_dict = pinyin_utility.load_pinyin_part_dict()
wp = word_part.WordPart()

def generate_vocab_vec():
    raw_vocab_list, raw_vocab, size1 = load_w2v('../../model/raw.txt')
    pinyin_vocab_list, pinyin_vocab, size2 = load_w2v('../../model/pinyin.txt')
    part_vocab_list, part_vocab, size3 = load_w2v('../../model/part.txt')
    fw1 = open('../../model/vocab.txt', 'w')
    fw2 = open('../../model/vec.txt', 'w')
    size = size1 + size2 + size3
    fw2.write(str(len(raw_vocab_list)) + ' ' + str(size) + '\n')
    fw1.write('UNK\n')
    unk_list = [0] * size
    unk = np.asarray(unk_list, dtype=np.float32).tolist()
    fw2.write(' '.join([str(i) for i in unk]) + '\n')
    for word in raw_vocab:
        word_pinyin = pypinyin.lazy_pinyin(word, errors=lambda x:u'ng')
        try:
            pinyin_list = pinyin_part_dict[word_pinyin[0]]
        except:
            print word
            continue
        py_res = []
        for py in pinyin_list:
            py_res.append(pinyin_vocab[py])
        np_py = np.asarray(py_res, dtype=np.float32)
        np_py = np_py.mean(0)
        pinyin_vec = np_py.tolist()
        
        part_list = wp.get_word_partition(word)
        part_res = []
        for part in part_list:
            part_res.append(part_vocab[part])
        np_part = np.asarray(part_res, dtype=np.float32)
        np_part = np_part.mean(0)
        part_vec = np_part.tolist()

        vec = raw_vocab[word] + pinyin_vec + part_vec

        fw1.write(word.encode('utf-8') + '\n')
        fw2.write(' '.join([str(i) for i in vec]) + '\n')
    fw1.close()
    fw2.close()



def load_w2v(filepath):
    fr = open(filepath)
    vocab = {}
    vocab_list = []
    row = fr.readline().strip().decode('utf8').split(' ')
    vocab_size = int(row[0])
    vec_size = int(row[1])
    for row in fr.readlines():
        array = row.strip().decode('utf8').split(' ')
        vocab_list.append(array[0])
        vocab[array[0]] = array[1:]
    fr.close()
    return vocab_list, vocab, vec_size

if __name__=='__main__':
    generate_vocab_vec()