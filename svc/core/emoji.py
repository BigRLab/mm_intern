#coding=utf-8
import os

emoji_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../resource/emoji.txt')
emoji_mapping_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../resource/emoji_mapping.txt')

def load_emoji_vocab():
    fr = open(emoji_path)
    index = 0
    emoji_dict = {}
    for row in fr.readlines():
        array = row.strip().split('\t')
        emoji = array[0].decode('utf8')
        if emoji not in emoji_dict:
            emoji_dict[emoji] = index
            index += 1
    fr.close()
    return emoji_dict

def load_emoji_mapping_dict():
    fr = open(emoji_mapping_path)
    emoji_dict = {}
    for row in fr.readlines():
        array = row.strip().decode('utf8').split('\t')
        if len(array) != 2:
            continue
        emoji = array[0]
        if emoji not in emoji_dict:
            emoji_dict[emoji] = array[1]
    fr.close()
    return emoji_dict

def isEmoji(word):
    if not word:
        return False
    if u"\U0001F600" <= word and word <= u"\U0001F64F":
        return True
    elif u"\U0001F300" <= word and word <= u"\U0001F5FF":
        return True
    elif u"\U0001F680" <= word and word <= u"\U0001F6FF":
        return True
    elif u"\U0001F1E0" <= word and word <= u"\U0001F1FF":
        return True
    else:
        return False

if __name__=='__main__':
    s = u'➕'
    print isEmoji(s)