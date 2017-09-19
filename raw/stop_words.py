#coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def load_stop_words_vocab():
    sw = set()
    fr = open('resource/stop_words.txt')
    for row in fr.readlines():
        word = row.strip().decode('utf8')
        if word == '':
            continue
        else:
            if word not in sw:
                sw.add(word)
    fr.close()
    return sw