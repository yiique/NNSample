__author__ = 'liushuman'

import pickle
import time


RAW_CORPUS = '../Static/199801_renminribao_utf8.txt'
DICT = '../Static/199801_renminribao_utf8.dict'
SEGMENT = '../Static/199801_renminribao_utf8.segment'


class PreProcessor(object):

    def __init__(self):
        try:
            f_dict = open(DICT, 'r')
            f_segment = open(SEGMENT, 'r')
            self.dict = pickle.load(f_dict)
            self.segment = pickle.load(f_segment)
            f_dict.close()
            f_segment.close()
        except:
            self.preprocess()

    def close_event(self):
        f_dict = open(DICT, 'w')
        f_segment = open(SEGMENT, 'w')
        pickle.dump(self.dict, f_dict)
        pickle.dump(self.segment, f_segment)
        f_dict.close()
        f_segment.close()

    def preprocess(self):
        self.dict = {}
        self.segment = []

        f_corpus = open(RAW_CORPUS, 'r')

        print "preprocessing raw corpus..."
        count = 0
        for line in f_corpus:
            count += 1
            if count % 2000 == 0:
                print "preprocessing...", count, time.ctime()

            if line.startswith('\n'):
                continue

            word_sequence = self.get_data(line)

            for word in word_sequence:
                if word not in self.dict:
                    self.dict[word] = 0
                self.dict[word] += 1
            self.segment.append(word_sequence)

        self.dict = sorted(self.dict.iteritems(), key=lambda d:d[1], reverse=True)
        self.dict = [x[0] for x in self.dict][:20000]
        self.dict.append('NONE')
        #print ' '.join(self.dict)

        print "dict size: ", len(self.dict)
        print "set size: ", len(self.segment)
        self.close_event()

    def get_data(self, line):
        sequences = line.split('  ')

        return [(x.split('/')[0]).strip() for x in sequences[1:-1]]

    def generate_word_index(self, word):
        try:
            return self.dict.index(word)
        except:
            return self.dict.index("NONE")


if __name__ == "__main__":
    preprocessor = PreProcessor()
