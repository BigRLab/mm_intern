#coding=utf-8
import os
import json
def processLog2traindata(filepath):
    fr = open(filepath)
    fw1 = open('../../train_data/spam.txt', 'w')
    fw2 = open('../../train_data/text.txt', 'w')
    for line in fr.readlines():
        js = json.loads(line.strip())
        fw1.write(js['spam'].encode('utf8') + '\n')
        fw2.write(js['text'].encode('utf8') + '\n')
    fr.close()

if __name__=='__main__':
    filepath = '../../logdata/spam-相似文本-模型数据-标注_training_2.0.0_否.log'
    processLog2traindata(filepath)