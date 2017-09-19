#coding=utf-8
import sys
import word_trigram
import word_part
import stop_words
import re
import jpype
import emoji
import numpy as np
from sklearn.cluster import KMeans
reload(sys)
sys.setdefaultencoding('utf8')

wp = word_part.WordPart()
jar_path = '/Users/momo/PycharmProjects/DSSM/lib/antispam-spam-recognition-10.0.1-20170421.043152-34-jar-with-dependencies.jar'
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path="+jar_path)

VxSmallFilter = jpype.JClass("com.momo.spam.filter.VxSmallFilter")
VxSmallFilter.setFortest(True)
vxSmallFilter = VxSmallFilter()

WebUrlFilter = jpype.JClass("com.momo.spam.filter.WebUrlFilter")
webUrlFilter = WebUrlFilter()

emoji_dict = emoji.load_emoji_vocab()
stop_words_set = stop_words.load_stop_words_vocab()

def process_sentence(sentence):
    wp_list = wp.get_sentence_word_partition(sentence)
    trigram = word_trigram.sentence_to_letter_trigram(sentence)
    return wp_list, trigram

def read_file(filepath):
    fr = open(filepath)
    data = []
    for row in fr.readlines():
        row = row.strip().decode('utf-8').replace(' ', '').lower()
        if row == u"":
            continue
        for i in row:
            if i in stop_words_set:
                del i
        data.append(row)
    fr.close()
    return data

def remove_string(string, rm_list):
    for v in rm_list:
        string = string.replace(v, '')
    return string

def get_alpha_string(string):
    pattern = r"([a-zA-Z][a-zA-Z][a-zA-Z]+)"
    return re.findall(pattern, string)

def extract_feature(one):
    # get emoji from text
    emoji_list = []
    for i, uchar in enumerate(one):
        if uchar in [u"\uD83D", u"\uD83C"]:
            emoji_uchar = uchar + one[i + 1]
            i += 1
        else:
            emoji_uchar = uchar
        if emoji_uchar in emoji_dict:
            emoji_list.append(emoji_uchar)
    try:
        vx = list(vxSmallFilter.filterAll(one))
    except Exception, e:
        vx = []
    if len(vx) > 0:
        has_vx = 1
        one = remove_string(one, vx)
    else:
        has_vx = 0
    web = webUrlFilter.normalTarget2weburl(one)
    if web != None:
        has_web = 1
        one = remove_string(one, [web])
    else:
        has_web = 0
    alpha_string = get_alpha_string(one)
    alpha_string_wh = []
    for word in alpha_string:
        alpha_string_wh.extend(word_trigram.word_hashing(word))
    wp_list = wp.get_sentence_word_partition(one)
    trigram = word_trigram.sentence_to_letter_trigram_list(one)
    return emoji_list, alpha_string_wh, wp_list, trigram, has_vx, has_web

def generate_dict():
    emoji_set = set()
    wordhash_set = set()
    wordpart_set = set()
    data = read_file('data/sc.txt')
    for one in data:
        emoji_list, alpha_string_wh, wp_list, trigram, has_vx, has_web = extract_feature(one)
        for v in emoji_list:
            if v not in emoji_set:
                emoji_set.add(v)
        for v in alpha_string_wh + trigram:
            if v not in wordhash_set:
                wordhash_set.add(v)
        for v in wp_list:
            for w in v:
                if w != '' and w not in wordpart_set:
                    wordpart_set.add(w)

    emoji_vocab = {}
    wordhash_vocab = {}
    wordpart_vocab = {}
    emoji_fw = open('vocab/emoji.vocab', 'w')
    wordhash_fw = open('vocab/wordhash.vocab', 'w')
    wordpart_fw = open('vocab/wordpart.vocab', 'w')
    for v in emoji_set:
        emoji_fw.write(v+'\n')

    for v in wordhash_set:
        wordhash_fw.write(v+'\n')

    for v in wordpart_set:
        wordpart_fw.write(v+'\n')

    emoji_fw.close()
    wordhash_fw.close()
    wordpart_fw.close()

def load_vocab(filepath):
    vocab = {}
    fr = open(filepath)
    index = 0
    for row in fr.readlines():
        row = row.strip()
        if row != '':
            vocab[row] = index
            index += 1
    fr.close()
    return vocab

def load_all_data(filepath):
    fr = open(filepath)
    spam = 0
    space = 0
    all_list = []
    raw_list = []
    content = []
    raw_content = []
    for row in fr.readlines():
        raw_row = row.strip('\n')
        row = row.strip().decode('utf-8').replace(' ', '').lower()
        for i in row:
            if i in stop_words_set:
                del i
        if spam == 0:
            spam =1
            continue
        if space == 0:
            space = 1
            continue
        if row == '###---###':
            spam = 0
            space = 0
            all_list.append(list(content))
            raw_list.append(raw_content)
            content = []
            raw_content = []
            continue
        if len(row) > 4 and row not in content:
            content.append(row)
            raw_content.append(raw_row)
    fr.close()
    return all_list, raw_list

def get_vocab_vec(lists, vocab):
    size = len(vocab)
    vec = [0] * size
    for w in lists:
        if w in vocab:
            index = vocab[w]
            vec[index] = 1
    return vec

def generate_vec(all_text, raw_text):
    emoji_vocab = load_vocab('vocab/emoji.vocab')
    wordhash_vocab = load_vocab('vocab/wordhash.vocab')
    wordpart_vocab = load_vocab('vocab/wordpart.vocab')
    all_vec = []
    for text in all_text:
        part_vec = []
        for one in text:
            emoji_list, alpha_string_wh, wp_list, trigram, has_vx, has_web = extract_feature(one)
            emoji_vec = get_vocab_vec(emoji_list, emoji_vocab)
            wordhash_vec = get_vocab_vec(alpha_string_wh+trigram, wordhash_vocab)
            wordpart_vec = get_vocab_vec(wp_list, wordpart_vocab)
            if has_vx:
                vx_vec = [2]
            else:
                vx_vec = [0]
            if has_web:
                web_vec = [2]
            else:
                web_vec = [0]
            vec = emoji_vec + wordhash_vec + wordpart_vec + vx_vec + web_vec
            part_vec.append(vec)
        all_vec.append(part_vec)
    return all_vec, all_text, raw_text

def cluster():
    filepath = 'data/sc.txt'
    all_text, raw_text = load_all_data(filepath)
    all_vec, all_text, raw_text = generate_vec(all_text, raw_text)
    fw = open('data/cluster_raw.txt', 'w')
    for i, text in enumerate(all_text):
        l = len(text)
        vec_array = np.asarray(all_vec[i])
        print i,vec_array.shape, l
        n = l/10
        if n<2:
            print "less data"
            continue
        kmeans = KMeans(n_clusters=n, random_state=0).fit(vec_array)
        res = {}
        for j,label in enumerate(kmeans.labels_):
            if label not in res:
                res[label] = []
            res[label].append(raw_text[i][j])
        for k in res:
            fw.write(str(i) + '__' + str(k) + '\n')
            fw.write('\n')
            for t in res[k]:
                fw.write(t + '\n')
            fw.write('--------')
            fw.write('\n')
    fw.close()

def main():
    data = read_file('data/sc.txt')
    for one in data:
        emoji_list, alpha_string_wh, wp_list, trigram, has_vx, has_web = extract_feature(one)
#        print ' '.join(emoji_list)
#        print ' '.join(alpha_string_wh)
#        print ' '.join([i[0] + ' ' + i[1] for i in wp_list])
#        print ' '.join(trigram)
        print "%s\t%s\t%s\t%s\t%d\t%d" % (' '.join(emoji_list), ' '.join(alpha_string_wh), ' '.join([i[0] + ' ' + i[1] for i in wp_list]), ' '.join(trigram), has_vx, has_web)

def cluster_spam():
    filepath = 'data/spam.txt'
    all_text, raw_text = load_spam_data(filepath)
    all_vec, all_text, raw_text = generate_vec(all_text, raw_text)
    fw = open('data/cluster_spam.txt', 'w')
    for i, text in enumerate(all_text):
        l = len(text)
        vec_array = np.asarray(all_vec[i])
        print i, vec_array.shape, l
        n = l / 10
        if n < 2:
            print "less data"
            continue
        kmeans = KMeans(n_clusters=n, random_state=0).fit(vec_array)
        res = {}
        for j, label in enumerate(kmeans.labels_):
            if label not in res:
                res[label] = []
            res[label].append(raw_text[i][j])
        for k in res:
            fw.write(str(i) + '__' + str(k) + '\n')
            fw.write('\n')
            for t in res[k]:
                fw.write(t + '\n')
            fw.write('--------')
            fw.write('\n')
    fw.close()

def load_spam_data(filepath):
    fr = open(filepath)
    raw_text = []
    all_text = set()
    for row in fr.readlines():
        raw_row = row.strip('\n')
        row = row.strip().decode('utf-8').replace(' ', '').lower()
        if row == '':
            continue
        if row not in all_text:
            all_text.add(row)
            raw_text.append(raw_row)
    fr.close()
    return [list(all_text)], [raw_text]

if __name__=='__main__':
    exit()
    cluster_spam()