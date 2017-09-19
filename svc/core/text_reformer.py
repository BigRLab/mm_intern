#coding=utf-8
import sys
import os
import jpype
import stop_words
import emoji
import re
import special_string
reload(sys)
sys.setdefaultencoding('utf8')

class TextReformer(object):
    def __init__(self):
        #use jar class by jpype
        jar_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../jar/antispam-spam-recognition-10.0.1-20170421.043152-34-jar-with-dependencies.jar')
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jar_path)

        #define weixin number filter
        VxSmallFilter = jpype.JClass("com.momo.spam.filter.VxSmallFilter")
        VxSmallFilter.setFortest(True)
        self.vxSmallFilter = VxSmallFilter()

        #define web url filter
        WebUrlFilter = jpype.JClass("com.momo.spam.filter.WebUrlFilter")
        self.webUrlFilter = WebUrlFilter()

        #load stop words dict
        self.stop_words_vocab = stop_words.load_stop_words_vocab()

        #load emoji vocab and emoji mapping
        self.emoji_vocab = emoji.load_emoji_vocab()
        self.emoji_mapping = emoji.load_emoji_mapping_dict()

        #load special string mapping
        self.special_string_mapping = special_string.load_special_string_mapping()

    def replace_weixin(self, text):
        try:
            vx_list = self.vxSmallFilter.filterAll(text)
        except Exception, e:
            vx_list = []
        if vx_list != None and len(vx_list) > 0:
            for vx in vx_list:
                text = text.replace(vx, u'灎')
        return text

    def replace_weburl(self, text):
        try:
            weburl = self.webUrlFilter.normalTarget2weburl(text)
        except Exception, e:
            weburl = None
        if weburl != None:
            text = text.replace(weburl, u'陹')
        return text

    def remove_stop_words(self, text):
        res = []
        for word in text:
            if word not in self.stop_words_vocab and word !=' ':
                res.append(word)
        return ''.join(res)

    def replace_emoji(self, text):
        res = []
        l = len(text)
        i = 0
        while i<l:
            if text[i] in [u"\uD83D", u"\uD83C"] and i < l-1:
                emoji_uchar = text[i] + text[i + 1]
                i += 1
            else:
                emoji_uchar = text[i]
            if emoji.isEmoji(emoji_uchar) or emoji_uchar in self.emoji_vocab:
                if emoji_uchar in self.emoji_mapping:
                    res.append(self.emoji_mapping[emoji_uchar])
            else:
                res.append(text[i])
            i += 1
        return ''.join(res)

    def replace_special_string(self, text):
        for key in self.special_string_mapping:
            if key in text:
                text = text.replace(key, self.special_string_mapping[key])
        return text

    def replace_number_string(self, text):
        pattern = r'\d+'
        number_list = re.findall(pattern, text)
        for number in number_list:
            if len(number) < 6:
                new = u'潫'
            else:
                new = u'潂'
            text = text.replace(number, new, 1)
        return text

    def remove_not_chinease(self,text):
        res = []
        for word in text:
            if word >= u'\u4e00' and word <= u'\u9fa5':
                res.append(word)
        return ''.join(res)

    def reform_text(self, text):
        text = text.lower()
        text = self.replace_weburl(text)
        text = self.remove_stop_words(text)
        text = self.replace_emoji(text)
        text = self.replace_weixin(text)
        text = self.replace_special_string(text)
        text = self.replace_number_string(text)
        text = self.remove_not_chinease(text)
        return text

    def destroyed(self):
        jpype.shutdownJVM()

if __name__=='__main__':
    s = u'本店abs123加🅱工出售🏀海,南,黄,花,梨饰品、摆hello件及雕刻和沉香手串、项链➕VX:H_NHHL0518'
    s = u'陌陌weixin很少在线，麻烦加微18087469444，电话同步！'
    s = u'+v欣有福利：灎'
    s = u'欢迎月儿又来送福利来啦😄😄😄'
    s = u'觉得播漂亮的扣一个①扣完有福利'
    s = u'本店abs123加🅱工出售🏀海,南,黄,花,梨饰品、摆hello件及雕刻和沉香手串、项链➕VX:H_NHHL0518'
    s = u'裸窷薇pkw5796'
    textReformer = TextReformer()
    print textReformer.reform_text(s)
    textReformer.destroyed()