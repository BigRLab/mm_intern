#coding=utf-8
import os

special_string_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../resource/special_string_mapping.txt')

def load_special_string_mapping():
    global special_string_path
    mapping_dict = {}
    fr = open(special_string_path)
    for row in fr.readlines():
        array = row.strip().decode('utf8').split(' ')
        if row.strip() == '' or len(array) != 2:
            continue
        if array[0] not in mapping_dict:
            mapping_dict[array[0]] = array[1]
    fr.close()
    return mapping_dict