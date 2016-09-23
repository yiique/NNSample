# coding:utf-8
__author__ = 'liushuman'

import pickle


file_prefix = "./"
source_name = "qa19_path-finding_train.txt"
QApair_name = "qa19_path-finding_train.preprocessed"
dictionary_name = "qa19_path-finding_dict.txt"


class PreProcess(object):

    def __init__(self):
        try:
            print "try loading from file..."
            f = open(file_prefix + dictionary_name, 'r')
            self.dictionary = pickle.load(f)
            f.close()
            f = open(file_prefix + QApair_name, 'r')
            self.QApairs = pickle.load(f)
            f.close()
        except:
            self.initialize()

    def initialize(self):
        print "init dict & training_set..."
        f_source = open(file_prefix + source_name, 'r')
        f_dictionary = open(file_prefix + dictionary_name, 'w')
        f_QApair = open(file_prefix + QApair_name, 'w')

        self.QApairs = []
        self.dictionary = {}
        self.QApair = {}
        self.QApair['f'] = []

        for line in f_source:
            line_processed, line_type = self.line_process(line)

            self.update_QApairs(line_processed, line_type)
            self.update_dictionary(line_processed, line_type)

        self.dictionary = [key for key in self.dictionary]

        pickle.dump(self.dictionary, f_dictionary)
        pickle.dump(self.QApairs, f_QApair)
        f_dictionary.close()
        f_QApair.close()
        f_source.close()
        print "init done."

    def line_process(self, line):
        if line[0] != '6':
            line_type = 'f'
            line_processed = line[2:-2]
        else:
            line_type = 'q&a'
            line_processed = line[2:].split("?")

        return line_processed, line_type

    def update_QApairs(self, line_processed, line_type):
        if line_type == 'f':
            self.QApair['f'].append(line_processed)
        else:
            self.QApair['q'] = line_processed[0]
            a_pair = line_processed[1][1:-1].split("\t")
            direction_sequence = a_pair[0].split(",")
            # step_sequence = a_pair[1].split(" ")
            # self.QApair['a'] = {}
            # for i in range(0, len(direction_sequence)):
                # self.QApair['a'][direction_sequence[i]] = step_sequence[i]
            self.QApair['a'] = direction_sequence

            self.QApairs.append(self.QApair)
            self.QApair = {}
            self.QApair['f'] = []

    def update_dictionary(self, line_processed, line_type):
        if line_type == 'f':
            sentence = line_processed
        else:
            sentence = line_processed[0]

        for word in sentence.split(' '):
            self.dictionary[word] = 0

    def generate_word_vector(self, word):
        word_vector = [0 for i in range(0, len(self.dictionary)+1)]

        in_vector = False
        for i in range(0, len(self.dictionary)):
            if word == self.dictionary[i]:
                word_vector[i+1] = 1
                in_vector = True
                break

        if not in_vector:
            word_vector[0] = 1

        return word_vector

    def generate_word_index(self, word):
        try:
            return self.dictionary.index(word)
        except:
            return 0


if __name__ == "__main__":
    preprocess = PreProcess()
    print preprocess.generate_word_vector('the')