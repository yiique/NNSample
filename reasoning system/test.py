__author__ = 'liushuman'

from preprocess import *
from RNNencoder import *
from DNNreasoner import *

if __name__ == "__main__":
    preprocess = PreProcess()
    rnnencoder = RNNencoder()
    dnnreasoner = DNNreasoner()

    '''question = "How do you go from the garden to the bedroom"
    sentences = ["The bedroom is south of the hallway",
                 "The bathroom is east of the office",
                 "The kitchen is west of the garden",
                 "The garden is south of the office",
                 "The office is south of the bedroom"]

    question_idxs = rnnencoder.generate_idxs([preprocess.generate_word_index(x)+1 for x in question.split(' ')])
    sentences_idxs = []
    for sentence in sentences:
        sentences_idxs.append(rnnencoder.generate_idxs([preprocess.generate_word_index(x)+1 for x in sentence.split(' ')]))

    question_encode = rnnencoder.theano_encode(question_idxs)[-1]
    sentences_encode = []
    for sentence_idxs in sentences_idxs:
        sentences_encode.append(rnnencoder.theano_encode(sentence_idxs)[-1])

    print dnnreasoner.index2di(dnnreasoner.theano_classify(question_encode, sentences_encode))'''

    precision = 0
    count = 0

    for QApair in preprocess.QApairs:
        question_encode, factors_encode, y = dnnreasoner.generate_training_data(QApair, preprocess, rnnencoder)
        answer_list = dnnreasoner.theano_classify(question_encode, factors_encode)

        if y == answer_list[1]:
            precision += 1
        else:
            print answer_list[0]
        count += 1
        if count == 20:
            break

    print precision, count