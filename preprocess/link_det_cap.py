import sys
import os
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import copy
import time

def wordnet_preprocess(vocab_A, vocab_B, pos=wn.NOUN, use_lemma=False):
    """
    Determine whether 2 given words are matched / have same meaning,
    by checking whether synsets have overlap, or whether it's hypernyms synset over the other.
    If use_lemma, plus checking whether the lemmas of the most-frequent synset match.
    Ref: https://github.com/Kenneth-Wong/het-eccv20/blob/master/data/vg200/triplet_match.py
    """
    vocab_A = vocab_A.lower()
    vocab_B = vocab_B.lower()
    if pos == wn.NOUN:
        if len(vocab_A.split(" ")) != 1:
            vocab_A = vocab_A.split(" ")[-1]
        if len(vocab_B.split(" ")) != 1:
            vocab_B = vocab_B.split(" ")[-1]

    base_vocab_A = wn.morphy(vocab_A)
    base_vocab_B = wn.morphy(vocab_B)
    if base_vocab_A is None or base_vocab_B is None:
        return False

    if not use_lemma:
        synsets_A = wn.synsets(base_vocab_A, pos)
        synsets_B = wn.synsets(base_vocab_B, pos)

        # justify whether two synsets overlap with each other
        for s_a in synsets_A:
            for s_b in synsets_B:
                if s_a == s_b or len(list(set(s_a.lowest_common_hypernyms(s_b)).intersection(set([s_a, s_b])))) > 0 :
                    return True
        return False
    else:
        synsets_A = wn.synsets(base_vocab_A, pos)
        synsets_B = wn.synsets(base_vocab_B, pos)
        synsets_A = [synsets_A[0]] if len(synsets_A) > 0 else [] # most frequent synset for given word
        synsets_B = [synsets_B[0]] if len(synsets_B) > 0 else [] # most frequent synset for given word

        # justify whether two synsets overlap with each other
        for s_a in synsets_A:
            for s_b in synsets_B:
                opt1 = s_a == s_b
                opt2 = len(list(set(s_a.lowest_common_hypernyms(s_b)).intersection(set([s_a, s_b])))) > 0
                s_a_lemma = [str(lemma.name()) for lemma in s_a.lemmas()]
                s_b_lemma = [str(lemma.name()) for lemma in s_b.lemmas()]
                overlap = [item for item in s_a_lemma if item in s_b_lemma]
                opt3 = len(overlap) > 0
                if opt1 or opt2 or opt3:
                    return True
        return False        

file_1 = "objects_oidv4_vocab.txt"  #  "objects_updown_vocab.txt" #  
file_2 = "parsed_nouns.txt" # the nouns parsed from captions
method = "synset"
use_lemma = True # whether use lemma for synset matching
threshold = 2.6 # similarity score threshold
index_mapped = [] # the index of word_list2 which has been mapped to some category in word_list1

word_list1 = [str(line.strip().split(',')[0]) for line in open(file_1).readlines()]  # root words
word_list2 = ['__background__'] + [str(line.strip().split(',')[0]) for line in open(file_2).readlines()]  # all words to be merged
word_list3 = ['__background__,0'] + [str(line.strip()) for line in open(file_2).readlines()]  # same as list2 but include frequency
print(len(word_list1))
print(len(word_list2))

word_map = {} # the mapping of word (string), used for visualization
cls_ind_map = {} # the mapping class index in dataset, used for model training and testing
for i in range(len(word_list1)):
    word_map[word_list1[i]] = []

    cls_ind_map[i] = []
    for j in range(len(word_list2)):
        if word_list1[i] == word_list2[j]: # directly matched
            word_map[word_list1[i]].append(word_list2[j])
            cls_ind_map[i].append(j)
            index_mapped.append(j)
        else:
            if wordnet_preprocess(word_list1[i], word_list2[j], pos=wn.NOUN, use_lemma=use_lemma):
                word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)

# save the word (string) matching results
file_name = 'word_map_{}'.format(method) if use_lemma else 'word_map_{}_no_lemma'.format(method)
with open(file_name+'.txt', 'w') as text_file:
    for i in range(len(word_list1)):
        this_map = word_map[word_list1[i]]
        if len(this_map) == 0:
            continue
        text_file.write(word_list1[i])
        for j in range(len(this_map)):
            text_file.write("," + this_map[j])
        text_file.write("\n")

# save the class index matching results
np.save(file_name+'.npy', cls_ind_map)

# show the category in word_list2 which was matched successfully
print("\nMatch successfully:")
cnt = 0
with open("match_successfully.txt", 'w') as f:
    for i, item in enumerate(word_list2):
        if i in index_mapped:
            print(word_list3[i])
            f.write(word_list3[i]+"\n")
            cnt += 1
    print("\nIn total, {} categories were matched!\n".format(cnt))
    f.write("\nIn total, {} categories were matched!\n".format(cnt))

# show the category in word_list2 which wasn't matched successfully
print("\nMatch failed:")
cnt = 0
with open("match_failed.txt", 'w') as f:
    for i, item in enumerate(word_list2):
        if i not in index_mapped:
            print(word_list3[i])
            f.write(word_list3[i]+"\n")
            cnt += 1
    print("\nIn total, {} categories failed to be matched!\n".format(cnt))
    f.write("\nIn total, {} categories failed to be matched!\n".format(cnt))

print("Done!")