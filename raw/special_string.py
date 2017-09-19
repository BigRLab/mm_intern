#coding=utf-8
def load_special_string_mapping():
    mapping_dict = {}
    filepath = 'resource/special_string_mapping.txt'
    fr = open(filepath)
    for row in fr.readlines():
        array = row.strip().decode('utf8').split(' ')
        if row.strip() == '' or len(array) != 2:
            continue
        if array[0] not in mapping_dict:
            mapping_dict[array[0]] = array[1]
    fr.close()
    return mapping_dict