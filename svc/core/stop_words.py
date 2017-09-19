#coding=utf-8
import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')

stop_words_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../resource/stop_words.txt')

def load_stop_words_vocab():
    global stop_words_path
    sw = set()
    fr = open(stop_words_path)
    for row in fr.readlines():
        word = row.strip().decode('utf8')
        if word == '':
            continue
        else:
            if word not in sw:
                sw.add(word)
    fr.close()
    return sw