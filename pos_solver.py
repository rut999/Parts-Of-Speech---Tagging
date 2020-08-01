###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids: nakopa-rparvat-pvajja
#
# (Based on skeleton code by D. Crandall)
#


import random
import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
from typing import List, Any, Union, Dict


class Solver:

    def __init__(self):
        self.nothing = 'something'

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this! # FIXED :)
    def posterior(self, model, sentence, label):
        if model == "Simple":
            P = 0

            """ Calculate the Posterior Probabilities of simple model as P = p(word/tag)*p(tag) """
            for i in range(len(sentence)):
                P += math.log(self.pos_prob.get(label[i], 0.00000000001))
                P += math.log(self.em_prob.get((sentence[i], label[i]), 0.00000000001))
            return P
        elif model == "Complex":

            """Calculate the Posterior Probabilities of complex model as P = p(word/tag)*p(tag/prev_tag)*p(next_tag/tag) """
            P = 0
            for i in range(len(sentence)):
                if len(sentence) == 1:  # For sentences with length 1.
                    P += math.log(self.em_prob.get((sentence[i], label[i]), 0.00000000000000001))\
                        + math.log(self.trans_prob.get(('start', label[i]), 0.00000000000000001))
                elif i == 0:  # for First word.
                    P += math.log(self.em_prob.get((sentence[i], label[i]), 0.00000000000000001)) \
                        + math.log(self.trans_prob.get(('start', label[i]), 0.00000000000000001)) \
                        + math.log(self.trans_prob.get((label[i], label[i + 1]), 0.00000000000000001))
                elif i == len(sentence) - 1:  # For last word.
                    P += math.log(self.em_prob.get((sentence[i], label[i]), 0.00000000000000001)) \
                        + math.log(self.last_trans_prob[label[i - 1], label[0]].get(label[i], 0.00000000000000001))
                else:  # For other words.
                    P += math.log(self.em_prob.get((sentence[i], label[i]), 0.00000000000000001)) \
                        + math.log(self.trans_prob.get((label[i - 1], label[i]), 0.00000000000000001)) \
                        + math.log(self.trans_prob.get((label[i], label[i + 1]), 0.00000000000000001))

            return P
        elif model == "HMM":
            P = 0

            """ For HMM model P = p(word/tag)*prob(tag/prev_tag) """
            for i in range(len(sentence)):
                if i == 0:
                    P += math.log(self.em_prob.get((sentence[i], label[i]), 0.0000000000001))
                    P += math.log(self.trans_prob.get(('start', label[i])))
                else:
                    P += math.log(self.em_prob.get((sentence[i], label[i]), 0.0000000000001))
                    P += math.log(self.trans_prob.get((label[i - 1], label[i])))
            return P
        else:
            print("Unknown algo!")

    """ Do the training! """  # DONE :)

    def train(self, data):

        """ Intialize Dict for Counting all pos tags for each word. """
        self.word_pos_counts = dict()

        # Dictionary that stores(keys: words, vales: all pos tags  and their count for this word
        self.pos = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x',
                    '.']  # Parts of Speeches.

        self.pos_t = ['start', 'adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x',
                      '.']  # Transition Parts of Speeches.('start' represents initial pos tag.)

        """  Variable for counting the POS tags   """
        self.pos_counts = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,
                           'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}

        """ Initialize Dictionary for Calculating probability of complex model """
        self.last_trans_count = dict()
        self.last_trans_prob = dict()
        for i in self.pos:
            for j in self.pos:
                self.last_trans_count[i, j] = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,
                                               'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
                self.last_trans_prob[i, j] = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,
                                              'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}


        """  Variable for counting the transition counts   """
        self.trans_count = dict()
        for i in self.pos_t:
            for j in self.pos:
                self.trans_count[i, j] = 0  # Transition Probabilities of (pos1,pos2)

        """       Calculate the Count of all possible parts of speech for a Given Word           """
        for sentence, tags in data:

            """        Calculating the  pos counts for each word     """
            for w, t in zip(sentence, tags):
                if w in self.word_pos_counts.keys():
                    self.word_pos_counts[w][t] += 1
                else:
                    self.word_pos_counts[w] = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,
                                               'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
                    self.word_pos_counts[w][t] += 1

                """   Counting the number of times each POS appeared in the Training data set."""
                self.pos_counts[t] += 1

            """      Transition Counts of (pos1,pos2)    """
            for i in range(len(tags)):
                if i == 0:
                    self.trans_count['start', tags[i]] += 1
                elif i == len(tags) - 1:
                    self.trans_count[tags[i - 1], tags[i]] += 1

                    """Transition counts for the complex model"""
                    self.last_trans_count[tags[i-1], tags[0]][tags[i]] += 1
                else:
                    self.trans_count[tags[i - 1], tags[i]] += 1


        """ Calculate the prob of occurrence of each pos tag """
        self.pos_prob = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,
                         'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
        total_pos_tags = sum([self.pos_counts[i] for i in self.pos])
        for i in self.pos:
            self.pos_prob[i] = self.pos_counts[i]/total_pos_tags

        """    Counting the appearance of each pos tag given the previous pos tag     """
        self.pos2_pos1 = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,
                          'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}  # count of pos2 given pos1
        for i in self.pos:
            for j in self.pos_t:
                self.pos2_pos1[i] += self.trans_count[j, i]

        """     EMISSION PROBABILITIES     """
        self.em_prob = dict()
        for i in self.word_pos_counts.keys():
            for j in self.pos:
                if self.word_pos_counts[i][j] == 0:
                    self.em_prob[i, j] = 0.00000000000000001
                else:
                    self.em_prob[i, j] = self.word_pos_counts[i][j] / self.pos_counts[j]

        """    TRANSITION PROBABILITIES    """
        self.trans_prob = dict()
        for (i, j) in self.trans_count.keys():
            if self.trans_count[i, j] == 0:
                self.trans_prob[i, j] = 0.00000000000000001
            else:
                self.trans_prob[i, j] = self.trans_count[i, j] / self.pos2_pos1[j]

        """ Transition Probabilities for the complex model """
        self.total_counts = dict()  # Dictionary for total counts
        for i in self.last_trans_count.keys():
            self.total_counts[i] = 0
        for i in self.last_trans_count:
            for j in self.last_trans_count[i]:
                self.total_counts[i] += self.last_trans_count[i][j]
        for i in self.last_trans_count.keys():
            for j in self.last_trans_count[i]:
                if self.total_counts[i] == 0 or self.last_trans_count[i][j] == 0:
                    self.last_trans_prob[i][j] = 0.00000000000000001  # Intialize to 0.00000000000000001 for unknown Values.
                else:
                    self.last_trans_prob[i][j] = self.last_trans_count[i][j]/self.total_counts[i]



    # Functions for each algorithm. Right now this just returns nouns  fix this!  # LETS FIX THIS.;-)

    def simplified(self, sentence):
        """     For each word in sentence returns the highest occurred pos from training data else returns 'noun'   """
        labels = []
        label = 0
        for word in sentence:
            if word in self.word_pos_counts:
                max_tag_value = 0
                for x in self.pos:
                    if self.word_pos_counts[word][x] > max_tag_value:
                        label = x
                        max_tag_value = self.word_pos_counts[word][x]
                labels.append(label)
            else:
                """ Mostly nouns because names and places are highly likely to appear out of random in the new text """
                """ Therefore labeling them as 'noun' """
                labels.append("noun")
        return labels

    def complex_mcmc(self, sentence):
        count_dict = {}
        count = 0
        """To Check the solution by considering the sequence from simple model by which the iterations
         can be reduced to calculate the optimal pos tag sequence using gibs sampling """
        # seq = self.simplified(sentence)

        """ Consider random pos tag sequence such as 'noun' for all """
        seq = ["noun"] * len(sentence)

        """ Initialize count dictionary for pos tags during sampling  """
        for i in range(len(sentence)):
            count_dict[i] = {}
            for j in self.pos_counts.keys():
                count_dict[i][j] = 0

        for i in range(len(seq)):
            count_dict[i][seq[i]] += 1
        """ Start iterating 1000 times with healing period to 500 """
        for i in range(1000):
            for j in range(len(seq)):
                if len(sentence) == 1:
                    P = [math.log(self.em_prob.get((sentence[j], k), 0.00000000000000001))
                         + math.log(self.trans_prob.get(('start', k), 0.00000000000000001)) for k in self.pos]
                elif j == 0:
                    P = [math.log(self.em_prob.get((sentence[j], k), 0.00000000000000001))
                         + math.log(self.trans_prob.get(('start', k), 0.00000000000000001))
                         + math.log(self.trans_prob.get((k, seq[j + 1]), 0.00000000000000001)) for k in self.pos]
                elif j == len(seq) - 1:
                    P = [math.log(self.em_prob.get((sentence[j], k), 0.00000000000000001))
                         + math.log(self.last_trans_prob[seq[j - 1], seq[0]].get(k, 0.00000000000000001)) for k in self.pos]
                else:
                    P = [math.log(self.em_prob.get((sentence[j], k), 0.00000000000000001))
                         + math.log(self.trans_prob.get((seq[j - 1], k), 0.00000000000000001))
                         + math.log(self.trans_prob.get((k, seq[j + 1]), 0.00000000000000001)) for k in self.pos]
                P = [math.exp(P[i]) for i in range(len(P))]   # Convert to Probabilities
                total = sum(P)
                P = [P[i]/total for i in range(len(P))]  # Normalize the Values of P
                rand = random.random()    # Taking a Random Value between (0,1)
                c = 0
                for k in range(len(P)):
                    c += P[k]
                    if rand < c:
                        seq[j] = self.pos[k]
                        break
                """  Updating the counts of pos occurrences after the healing Period.  """
                if i > 500:
                    for k in range(len(seq)):
                        count_dict[k][seq[k]] += 1
        """ Take the values of maximum occurring pos tag for each word  """
        out_list = []
        for i in range(len(seq)):
            pos = ""
            max_count = 0
            for j in self.pos_counts.keys():
                if count_dict[i][j] >= max_count:
                    max_count = count_dict[i][j]
                    pos = j
            out_list.append(pos)
        # print(count_dict)
        return out_list
        # return ["noun"] * len(sentence)

    def hmm_viterbi(self, sentence):

        viterbi_table = [[] for i in range(len(self.pos_counts.keys()))]  # Create a Viterbi table

        pos_tags = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']  # POS tags

        # Route to store index of max values for each word
        viterbi_route = [[] for i in range(len(self.pos_counts.keys()))]

        k = 0
        for word in sentence:
            if k == 0:
                """ Possible POS tag Occurrences of first word """
                for i in range(0, 12):
                    word0_occ = math.log(self.em_prob.get((word, pos_tags[i]), 0.00000000000000001)) + \
                                math.log(self.trans_prob.get(('start', pos_tags[i]), 0.00000000000000001))
                    viterbi_table[i].append(word0_occ)
            else:
                for i in range(0, 12):
                    """ Emission Probability of the word for each pos tag """
                    em = math.log(self.em_prob.get((word, pos_tags[i]), 0.00000000000000001))
                    maximum_prev_state = []

                    """ Transition Probability of each pos tag from all prev pos tags """
                    for j in range(0, 12):
                        trans = self.trans_prob.get((pos_tags[j], pos_tags[i]), 0.00000000000000001)
                        temp = viterbi_table[j][k - 1] + math.log(trans)
                        maximum_prev_state.append(temp)

                    """ For each Transition probability obtained store the index of maximum value of transition """
                    max_val = max(maximum_prev_state)  # Store the max value of the transitions.
                    max_index = maximum_prev_state.index(max_val)  # Store the index of the max value.
                    viterbi_route[i].append(max_index)  # Save the value in route.
                    viterbi_table[i].append(em + max_val)  # update the viterbi table for the second word.
            k += 1  # Update k

        """ Back Track from the max value of the last column of viterbi_table """
        listes = []
        for i in viterbi_table:
            listes.append(i[len(i) - 1])
        indexes = listes.index(max(listes))
        sequence = [indexes]
        count = len(viterbi_route[0]) - 1
        while count >= 0:
            a = viterbi_route[indexes][count]
            sequence.append(a)
            indexes = a
            count -= 1
        sequence.reverse() # reverse the sequence
        labels = [pos_tags[i] for i in sequence]
        return labels

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labeling of the sentence, one
    #  part of speech per word.
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
