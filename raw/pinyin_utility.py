#coding=utf-8
'''
生成拼音的声母、韵母字典
'''
def generate_pinyin_part():
    shengmu = ['b','p','m','f','d', 't','n','l','g','k', 'h','j','q','x','zh','ch','sh','r','z','c', 's','y','w']
#    yunmu = ['a','o','e','i','u','v','ai','ei','ui','ao','ou','iu','ie','ve','er','an','en','in','un','ang','eng','ing','ong']
    yunmu = ['i', 'u', 'v', 'a', 'ia', 'ua', 'o', 'uo', 'e', 'ie', 'ue', 've', 'ai', 'uai', 'ei', 'ui', 'ao', 'iao', 'ou', 'iu', 'an', 'ian', 'uan', 'en', 'in', 'un', 'ang', 'iang', 'uang', 'eng', 'ing', 'ueng', 'ong', 'iong']
    fw = open('resource/shengmu_yunmu.txt', 'w')
    for i in shengmu:
        for j in yunmu:
            fw.write(i+j+'\t'+i+'_'+j+'\n')
    for i in yunmu:
        fw.write(i + '\t' + i + '\n')
    fw.close()

def load_pinyin_part_dict():
    pinyin_part_dict = {}
    fr = open('resource/shengmu_yunmu.txt')
    for row in fr.readlines():
        array = row.rstrip('\n').decode('utf8').split('\t')
        if row.rstrip('\n') == '' or len(array) != 2:
            continue
        pinyin = array[0]
        part = array[1].split('_')
        if pinyin not in pinyin_part_dict:
            pinyin_part_dict[pinyin] = part
    fr.close()
    return pinyin_part_dict

def test_dict():
    d = load_pinyin_part_dict()
    print len(d)
    print d[u'shang']

if __name__=='__main__':
    generate_pinyin_part()
    test_dict()