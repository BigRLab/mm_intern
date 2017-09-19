#coding=utf-8
import sys
import jpype
import pypinyin
import text_reformer
import re
import json
import random
reload(sys)
sys.setdefaultencoding('utf8')


def generate_text():
    fr = open('data/sc.txt')
    fw = open('data/test.txt', 'w')
    textRefomer = text_reformer.TextReformer()
    for line in fr.readlines():
        line = line.rstrip('\n').decode('utf8')
        if line == '' or line == '###---###':
            continue
        fw.write(textRefomer.reform_text(line)+'\n')
    textRefomer.destroyed()
    fr.close()
    fw.close()

def jpype_test():
    jar_path = '/Users/momo/PycharmProjects/DSSM/lib/antispam-spam-recognition-10.0.1-20170421.043152-34-jar-with-dependencies.jar'
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path="+jar_path)
    JDclass = jpype.JClass("com.momo.spam.filter.VxSmallFilter")
    #jd = jpype.JPackage("com.momo.spam.filter").VxSmallFilter
    #jd.setFortest(True)
    JDclass.setFortest(True)
    jd = JDclass()
    s = u"本店加工出售海南黄花梨饰品、摆件及雕刻和沉香手串、项链➕VX:HNHHL0518"
    s = s.lower()
    print s
    res = jd.filterAll(s)
    print res

    #web filter
    jd = jpype.JClass("com.momo.spam.filter.WebUrlFilter")
    j = jd()
    s = u"本店www.baidu.com想鲁哥哥加妹妹崴信uu1239i"
    print s
    res = j.normalTarget2weburl(s)
    print res

    d = {u'b':u'bi'}
    s = u'晕群'
    print pypinyin.lazy_pinyin(s,errors=lambda x:d.get(x, ''))

def replace_number():
    text = u'你好123我123手机号15612350986447'
    number_list = re.findall(r'\d+', text)
    print number_list
    for number in number_list:
        if len(number) < 6:
            new = u'潫'
        else:
            new = u'潂'
        text = text.replace(number, new, 1)
    print text

def clear_file():
    for f in ['part.txt', 'raw.txt', 'pinyin.txt']:
        filepath = 'w2v/' + f
        w_filepath = 'w2v/' + f +'.new'
        fr = open(filepath)
        fw = open(w_filepath, 'w')
        for row in fr.readlines():
            if row.strip() == '':
                continue
            fw.write(row)
        fr.close()
        fw.close()

def generate_train_data():
    fr = open('data/biaozhu_false.log')
    fw1 = open('train_data/false_train_query.txt', 'w')
    fw2 = open('train_data/false_train_doc.txt', 'w')
    textReformer = text_reformer.TextReformer()
    a = fr.readlines()
    random.shuffle(a)
    for row in a:
        row = row.strip().decode('utf8')
        if row == '':
            continue
        js = json.loads(row)
        t1 = textReformer.reform_text(js['text_1'])
        t2 = textReformer.reform_text(js['text_2'])
        if len(t1) * len(t2) != 0:
            fw1.write(t1+'\n')
            fw2.write(t2+'\n')
    textReformer.destroyed()
    fr.close()
    fw1.close()
    fw2.close()



if __name__=='__main__':
    #replace_number()
    #generate_text()
    #clear_file()
    generate_train_data()