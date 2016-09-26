# -*- coding: utf-8 -*-
__author__ = 'liushuman'

import re

# 按标点和数字切分句子
# 1gram 2gram 3gram
# 按阈值／次数挑一批出来
# 按频度生成query

CHAR_PATTERN_LIST = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                     'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                     'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']

class QueryGenerator(object):
    def __init__(self):
        self.f_source = open("../Static/data.en")
        self.frq_counter = {}

    def sent_spliter(self, sent):
        sent_list = []

        temp_sent = ''
        for i in range(0, len(sent)):
            char = sent[i]
            if char not in CHAR_PATTERN_LIST:
                if len(temp_sent) > 0:
                    sent_list.append(temp_sent)
                temp_sent = ""
            else:
                temp_sent += char
        if len(temp_sent) > 0:
            sent_list.append(temp_sent)

        return sent_list

    def gram_generator(self, sent_list):
        for sent in sent_list:
            word_list = sent.split(' ')
            for i in range(0, len(word_list))[::-1]:
                if word_list[i] == '':
                    del word_list[i]

            for i in range(0, len(word_list)):
                for j in range(0, 3):
                    if i+j >= len(word_list):
                        break
                    gram = ' '.join(word_list[i:i+j+1])
                    if gram not in self.frq_counter:
                        self.frq_counter[gram] = 0
                    self.frq_counter[gram] += 1

    def query_generator(self):
        line_counter = 0
        for line in self.f_source:
            if line_counter % 200 == 0:
                print "line: ", line_counter, "\tdict_size: ", len(self.frq_counter)
            if line_counter > 10000:
                break
            sent_list = self.sent_spliter(line)
            self.gram_generator(sent_list)
            line_counter += 1

        frq_list = sorted(self.frq_counter.iteritems(), key=lambda x: x[1], reverse=True)
        for i in range(0, len(frq_list))[::-1]:
            if frq_list[i][1] < 10:
                del frq_list[i]

        count = 0
        frq_begin = frq_list[0][1]
        frq_end = frq_begin
        f = open("../Static/query_set_" + str(frq_begin), 'w')
        for gram_pair in frq_list:
            gram = gram_pair[0]
            frq = gram_pair[1]

            if count >= 2000 and frq != frq_end:
                f.close()
                count = 0
                frq_begin = frq
                frq_end = frq
                f = open("../Static/query_set_" + str(frq_begin), 'w')
            elif count < 2000 and frq != frq_end:
                frq_end = frq

            f.write(gram + '\n')
            count += 1

if __name__ == "__main__":
    qg = QueryGenerator()
    qg.query_generator()