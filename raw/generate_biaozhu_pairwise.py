#coding=utf-8
import sys
import json
import random
reload(sys)
sys.setdefaultencoding('utf8')

def generate_cluser_judge_data():
    fr = open('data/cluster_raw.txt')
    title = 0
    space = 0
    content = []
    res = []
    for line in fr.readlines():
        line = line.strip('\n')
        if title == 0:
            title = 1
            index = line
            continue
        if space == 0:
            space = 1
            continue
        if line == '--------':
            space = 0
            title = 0
            if len(content) < 3:
                content = []
                continue
            if len(content) > 5:
                random.shuffle(content)
                content = content[:5]
            content_dict = {}
            content_dict[index] = content
            res.append(json.dumps(content_dict, ensure_ascii=False))
            content = []
            continue
        content.append(line)
    fw = open('data/cluster_biaozhu_t.txt', 'w')
    random.shuffle(res)
    for s in res:
        fw.write(s + '\n')
    fw.close()

def get_spam_cluster_id():
    ids = []
    fr = open('data/201_2447_batch2.txt')
    for row in fr.readlines():
        array = row.strip().split('\t')
        if row.strip() == '' or len(array) != 2:
            continue
        if array[1] == '8':
            continue
        js = json.loads(array[0])
        for k in js:
            ids.append(k)
    fr.close()
    return ids

def load_all_data(filepath):
    ids = get_spam_cluster_id()
    fr = open(filepath)
    spam = 0
    space = 0
    raw_list = {}
    raw_content = []
    for row in fr.readlines():
        raw_row = row.rstrip('\n').decode('utf8')
        if spam == 0:
            spam = 1
            title = raw_row
            continue
        if space == 0:
            space = 1
            continue
        if raw_row == u'--------':
            spam = 0
            space = 0
            if title in ids:
                raw_list[title] = raw_content
            raw_content = []
            continue
        raw_content.append(raw_row)
    fr.close()
    return raw_list

def save_data(data, filepath):
    fw = open(filepath, 'w')
    for key in data:
        fw.write(key + '\n\n')
        for text in data[key]:
            fw.write(text + '\n')
        fw.write('###---###' + '\n')
    fw.close()

def generate_pairwise(data, filepath):
    pairs = []
    for k in data:
        l = len(data[k])
        for i in range(l):
            for j in range(i+1, l):
#                if random.randint(1, 100) > 50:
#                    continue
                d = {}
                d['text_1'] = data[k][i]
                d['text_2'] = data[k][j]
                js = json.dumps(d, ensure_ascii=False)
                pairs.append(js)
    for _ in range(5):
        random.shuffle(pairs)
    fw = open(filepath, 'w')
    for js in pairs:
        fw.write(js + '\n')
    fw.close()

def generate_over_cluster_pairwise(data, filepath):
    new = {}
    for k in data:
        ks = k.strip().split('__')
        if ks[0] not in new:
            new[ks[0]] = {}
        new[ks[0]][ks[1]] = data[k]

    pairs = []
    for p in new:
        for _ in range(3):
            temp = []
            for q in new[p]:
                l = len(new[p][q])
                index = random.randint(0,l-1)
                temp.append(new[p][q][index])
            l = len(temp)
            for i in range(l):
                for j in range(i+1,l):
                    d = {}
                    d['text_1'] = temp[i]
                    d['text_2'] = temp[j]
                    js = json.dumps(d, ensure_ascii=False)
                    pairs.append(js)
    for _ in range(5):
        random.shuffle(pairs)
    fw = open(filepath, 'w')
    for js in pairs:
        fw.write(js + '\n')
    fw.close()


if __name__=='__main__':
    exit()
#    generate_cluser_judge_data()
    data = load_all_data('data/cluster_raw.txt')
#    save_data(data, 'data/cluster_new.txt')
    generate_pairwise(data, 'data/pairwise.txt')
#    generate_over_cluster_pairwise(data, 'data/pairwise_over.txt')