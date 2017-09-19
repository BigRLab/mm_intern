#coding=utf-8
from pypinyin import lazy_pinyin, load_single_dict, load_phrases_dict

def get_pinyin(sentence):
    special_mapping = {
        'b': u'bi',
    }
    res = lazy_pinyin(sentence, errors=lambda x:special_mapping.get(x, ''))
    return res

def sentence_to_letter_trigram_hierarchic(sentence):
    pinyin_list = get_pinyin(sentence)
    res = []
    trigrams = []
    for word in pinyin_list:
        if word.isalpha():
            res.append(word)
            trigrams.append(word_hashing(word))
    return trigrams

def sentence_to_letter_trigram_list(sentence):
    pinyin_list = get_pinyin(sentence)
    res = []
    trigrams = []
    for word in pinyin_list:
        if word.isalpha():
            res.append(word)
            trigrams.extend(word_hashing(word))
    return trigrams

def word_hashing(word):
    word = '#' + word + '#'
    l = len(word)
    res = []
    for i in range(1, l-1):
        trigram = word[i-1:i+2]
        res.append(trigram)
    return res


if __name__=='__main__':
    sentence = u'望梅大少嗯嗯看哪福利wo'
    sentence = u'人生何处不相逢'
    load_single_dict(pinyin_dict={0x55ef: u"en"})
    load_phrases_dict(phrases_dict={u'嗯哪':[[u'en'], [u'na']]})
    res = lazy_pinyin(sentence, errors='default')
    t = []
    for i in res:
        t.append([i])
    print t