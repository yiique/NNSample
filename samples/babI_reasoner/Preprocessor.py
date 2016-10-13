__author__ = 'liushuman'

import pickle


train_name = "qa19_path-finding_train.txt"
test_name = "qa19_path-finding_test.txt"
corpus_name = "qa19_path-finding_train.preprocessed"
dictionary_name = "qa19_path-finding_dict.txt"

DI_DICT = {'ee': 0, 'es': 1, 'ew': 2, 'en': 3, 'ss': 4, 'sw': 5, 'sn': 6, 'ww': 7, 'wn': 8, 'nn': 9,
           0: 'ee', 1: 'es', 2: 'ew', 3: 'en', 4: 'ss', 5: 'sw', 6: 'sn', 7: 'ww', 8: 'wn', 9: 'nn'}


class Preprocessor(object):

    def __init__(self):
        try:
            f = open(dictionary_name, 'r')
            self.dictionary = pickle.load(f)
            f.close()
            f = open(corpus_name, 'r')
            self.train_pairs = pickle.load(f)
            self.test_pairs = pickle.load(f)
            f.close()
        except:
            print "preprocessing raw corpus..."
            self.initialize()

    def initialize(self):
        f_train = open(train_name, 'r')
        f_test = open(test_name, 'r')
        f_corpus = open(corpus_name, 'w')
        f_dict = open(dictionary_name, 'w')

        self.train_pairs = [{'f': []}]
        self.test_pairs = [{'f': []}]
        self.dictionary = {}

        self.parse(f_train, self.train_pairs, self.dictionary)
        self.parse(f_test, self.test_pairs, self.dictionary)

        self.dictionary = [key for key in self.dictionary]

        pickle.dump(self.dictionary, f_dict)
        pickle.dump(self.train_pairs[:-1], f_corpus)
        pickle.dump(self.test_pairs[:-1], f_corpus)
        f_train.close()
        f_test.close()
        f_corpus.close()
        f_dict.close()
        print "training set size:", len(self.train_pairs[:-1])
        print "testing set size:", len(self.test_pairs[:-1])
        print "dict size:", len(self.dictionary), self.dictionary
        print "init done."

    def parse(self, f_raw, pairs_list, dictionary_dict):
        for line in f_raw:
            line_processed, line_type = self.line_process(line)

            self.update_pairs(line_processed, line_type, pairs_list)
            self.update_dictionary(line_processed, line_type, dictionary_dict)

    def line_process(self, line):
        if line[0] != '6':
            line_type = 'f'
            line_processed = line[2:-2]
        else:
            line_type = 'q&a'
            line_processed = line[2:].split("?")
            split_index = line_processed[1].index(',')
            ans_pair = [line_processed[1][split_index-1], line_processed[1][split_index+1]]
            line_processed[1] = self.di2index(ans_pair)

        return line_processed, line_type

    def update_pairs(self, line_processed, line_type, pairs_list):
        pair = pairs_list[-1]
        if line_type == 'f':
            pair['f'].append(line_processed)
        else:
            pair['q'] = line_processed[0]
            pair['a'] = line_processed[1]
            pairs_list.append({'f': []})

    def update_dictionary(self, line_processed, line_type, dictionary_dict):
        if line_type == 'f':
            sentence = line_processed
        else:
            sentence = line_processed[0]

        for word in sentence.split(' '):
            dictionary_dict[word] = 0

    def di2index(self, di_list):
        if di_list[0] + di_list[1] in DI_DICT:
            return DI_DICT[di_list[0] + di_list[1]]
        else:
            return DI_DICT[di_list[1] + di_list[0]]

    def index2di(self, index):
        return DI_DICT[index]

    def generate_training_data(self, pair):
        question_sequence = pair["q"].split(' ')
        factors_sequence = [x.split(' ') for x in pair["f"]]

        question_encode = [0 for i in range(0, len(self.dictionary))]
        factors_encode = [[0 for i in range(0, len(self.dictionary))] for x in factors_sequence]

        for i in range(0, len(question_sequence)):
            word = question_sequence[i]
            index = self.dictionary.index(word)

            question_encode[index] = float(i+1)/float(len(question_sequence))

        for j in range(0, 5):
            factor = factors_sequence[j]
            for i in range(0, len(factor)):
                word = factor[i]
                index = self.dictionary.index(word)

                factors_encode[j][index] = float(i+1)/float(len(factor))

        y = pair["a"]

        return question_encode, factors_encode, y


if __name__ == '__main__':
    preprocessor = Preprocessor()
