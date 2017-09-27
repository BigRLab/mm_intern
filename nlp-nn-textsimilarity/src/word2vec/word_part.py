#coding=utf-8
import csv
import re
class WordPart(object):
    _base_lib_path = 'resource/chinese_word_partition.csv'
    _extend_lib_path = 'resource/user_dict_cwp.csv'
    _word_dict = {}
    def __init__(self):
        self._load_lib(self._base_lib_path)
        self._load_lib(self._extend_lib_path)

    def _load_lib(self, lib_path):
        with open(lib_path, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                label = row[0].decode('utf-8')
                value = row[1].decode('utf-8')
                if value == '':
                    continue
                if label not in self._word_dict:
                    part_1 = value[1]
                    part_2 = value[2]
                    self._word_dict[label] = [part_1, part_2]

    def has_word(self, word):
        if word in self._word_dict:
            return True
        else:
            return False

    def isHanzi(self, word):
#        if re.match(u'[\u4e00-\u9fa5]+', word) == None:
        if word >= u'\u4e00' and word<=u'\u9fa5':
            return True
        else:
            return False

    def get_word_partition(self, word):
        if not self.isHanzi(word):
            raise IOError('Input word is not a chinese word')

        if self.has_word(word):
            return self._word_dict[word]
        else:
            return [word]

    def get_sentence_word_partition(self, sentence):
        res = []
        for word in sentence:
            if not self.isHanzi(word):
                continue
            res.append(self.get_word_partition(word))
        return res

if __name__=='__main__':
    wp = WordPart()
    s = u"www.baidu.com我的weixin是xubc_2005"
    res = wp.get_sentence_word_partition(s)
    for i in res:
        for j in i:
            print j
