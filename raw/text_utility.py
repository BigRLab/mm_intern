#coding=utf-8
import text_reformer
import pinyin_utility
import word_part
import pypinyin

class TextMultiRepresentation(object):
    def __init__(self):
        self.textReformer = text_reformer.TextReformer()

        self.pinyin_part = pinyin_utility.load_pinyin_part_dict()

        self.wordPart = word_part.WordPart()

        pypinyin.load_single_dict(pinyin_dict={0x55ef: u"en"})
        pypinyin.load_phrases_dict(phrases_dict={u'嗯哪': [[u'en'], [u'na']], u'人生何处不相逢':[[u'ren'], [u'sheng'], [u'he'], [u'chu'], [u'bu'], [u'xiang'], [u'feng']]})

    def multi_represent(self, text):
        text = self.textReformer.reform_text(text)
        raw_list = list(text)
        wordpart_list = self.wordPart.get_sentence_word_partition(text)
        pinyin_list = pypinyin.lazy_pinyin(text, errors=lambda x:u'ng')
        pinyin_part_list = []
        for pinyin in pinyin_list:
            py_part = self.pinyin_part.get(pinyin, [u'ng'])
            if py_part == [u'ng']:
                print text
                print pinyin
            pinyin_part_list.append(py_part)
        return raw_list, wordpart_list, pinyin_part_list

    def destroyed(self):
        self.textReformer.destroyed()

def main():
    fw = open('data/text_represent.txt', 'w')
    fr = open('data/sc.txt')
    textRepresent = TextMultiRepresentation()
    for row in fr.readlines():
        row = row.rstrip('\n').decode('utf8')
        if row == '' or row == '###---###':
            continue
        raw_list, wordpart_list, pinyin_part_list = textRepresent.multi_represent(row)
        fw.write(' '.join(raw_list) + '\n')
        fw.write(' '.join([' '.join(i) for i in wordpart_list]) + '\n')
        fw.write(' '.join([' '.join(i) for i in pinyin_part_list]) + '\n')
        fw.write('\n')
    fw.close()
    textRepresent.destroyed()

def generate_w2v_corpus():
    fw1 = open('w2v/raw.txt', 'w')
    fw2 = open('w2v/pinyin.txt', 'w')
    fw3 = open('w2v/part.txt', 'w')
    textRepresent = TextMultiRepresentation()

    fr = open('data/sc.txt')
    for row in fr.readlines():
        row = row.rstrip('\n').decode('utf8')
        if row == '' or row == '###---###':
            continue
        raw_list, wordpart_list, pinyin_part_list = textRepresent.multi_represent(row)
        fw1.write(' '.join(raw_list) + '\n')
        fw3.write(' '.join([' '.join(i) for i in wordpart_list]) + '\n')
        fw2.write(' '.join([' '.join(i) for i in pinyin_part_list]) + '\n')
    fr.close()

    fr = open('w2v/live_corpus.txt')
    for row in fr.readlines()[:2000000]:
        row = row.rstrip('\n').decode('utf8')
        content = row.split('\t')[1]
        raw_list, wordpart_list, pinyin_part_list = textRepresent.multi_represent(content)
        fw1.write(' '.join(raw_list) + '\n')
        fw3.write(' '.join([' '.join(i) for i in wordpart_list]) + '\n')
        fw2.write(' '.join([' '.join(i) for i in pinyin_part_list]) + '\n')
    fr.close()
    fw1.close()
    fw2.close()
    fw3.close()

    textRepresent.destroyed()

def generate_normal_corpus():
    fw1 = open('train_data/normal.txt', 'w')
    textRepresent = TextMultiRepresentation()

    fr = open('w2v/live_corpus.txt')
    for row in fr.readlines()[:2000000]:
        row = row.rstrip('\n').decode('utf8')
        content = row.split('\t')[1]
        if len(content) < 4:
            continue
        raw_list, wordpart_list, pinyin_part_list = textRepresent.multi_represent(content)
        fw1.write(''.join(raw_list) + '\n')
    fr.close()
    fw1.close()

    textRepresent.destroyed()



if __name__=='__main__':
#    generate_w2v_corpus()
    generate_normal_corpus()

