import sys
import os
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import copy
import ipdb
import time
from nltk.stem import WordNetLemmatizer

# Manually check & match the top-250 caption nouns
noun_manual_dict = {"group": "people", "couple": "people", "herd": "animal", "computer": "laptop", "ocean": "wave",\
                    "donut": "food", "doughnut": "food",\
                    "dish": "plate", "suit": "jacket", "meat": "food", "monitor": "laptop", }
# Manually check & match the unmatched VG predicates as well as the top-100 caption predicates
rel_manual_dict = {"ride": "riding", "ride on": "riding", "sit on": "sitting on", "in front": "in front of", "underneath": "under", "beneath": "under", "close to": "near", "play with": "playing",\
                "attach": "attached to", "cover": "covered in",  "grow on": "growing on",\
                "fly": "flying in", "fly at": "flying in", "fly around": "flying in", "fly into": "flying in", "fly by": "flying in",\
                "hang on": "hanging from", "hang over": "hanging from", "hang": "hanging from", "hang onto": "hanging from", "hang around": "hanging from",\
                "lay on": "laying on", "lay": "laying on",\
                "look into": "looking at", "look over": "looking at", "look in": "looking at", "look": "looking at", "look to": "looking at",\
                "mount": "mounted on", "back of": "on back of", "paint": "painted on", "park": "parked on",\
                "stand by": "standing on", "stand around": "standing on", "stand": "standing on",\
                "walk on": "walking on", "walk into": "walking in", "walk": "walking in", "cross": "across"}

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
                if s_a == s_b or len(list(set(s_a.lowest_common_hypernyms(s_b)).intersection(set([s_b])))) > 0 :
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
                opt2 = len(list(set(s_a.lowest_common_hypernyms(s_b)).intersection(set([s_b])))) > 0
                s_a_lemma = [str(lemma.name()) for lemma in s_a.lemmas()]
                s_b_lemma = [str(lemma.name()) for lemma in s_b.lemmas()]
                overlap = [item for item in s_a_lemma if item in s_b_lemma]
                opt3 = len(overlap) > 0
                if opt1 or opt2 or opt3:
                    return True
        return False        

def make_alias_dict(dict_file):
    """
    create an alias dictionary from a VG file: first word  in a line is the representative, the others belong to it.
    """
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab

def predicate_match(vocab_A, vocab_B, lem=None):
    """
    String matching of 2 predicte words.
    Use lemmatization on predicates since:
    (1) although the SG parser had returned the lemma of the predicates, there are still "holding" & "hold"
    (2) although the alias file will handle some of the predicate lemma, lemmatization can be additional chance
    (3) lemmatization has errors on predicate words, empirically
    """
    vocab_A = vocab_A.lower()
    vocab_B = vocab_B.lower()

    #The NLTK Lemmatization method is based on WordNet’s built-in morphy function.
    base_vocab_A = lem.lemmatize(vocab_A, 'v')
    base_vocab_B = lem.lemmatize(vocab_B, 'v')

    if base_vocab_A == base_vocab_B:
        return True
    else:
        return False   

#####################################################################################################################
#### This file generate the Dict between VG concepts and caption concepts. The goal is to convert the predicted caption
#### categories into VG standard categories, for evaluation purpose. To this end, each caption concept can be only 
#### matched to one of the VG categories, not vice versa. The rest caption concepts that didn't match to any of the 
#### VG category will be ignored during ranking the triplets.
#### For nouns, the matching priority is as follows:
#### 1. Direct string match: caption "man" is converted into VG "man" 
#### 2. Root string match: caption "baseball player" is converted into VG "player"
#### 3. Synset match: instead of lowest_common_hypernyms, the VG "room" must be hypernym of the caption "bathroom"
#### 0. Manual dict match: by looking at the results from the matching above, manually construct a dict for corner cases
#### For predicates, the matching priority is as follows:
#### 1. Direct string match: caption "on" is converted into VG "on" 
#### 2. VG predicate alias file match: use the alias file from VG to match the predicates
#### 3. Lemmatization match: caption "have" is converted into VG "has"
#### 0. Manual dict match: by looking at the results from the matching above, manually construct a dict for corner cases
#####################################################################################################################

if __name__ == '__main__':
    cap2vg_cls_map = {}
    dataset = 'vg'
    ############################################################################ 
    #### 1. map nouns
    ############################################################################
    concept_type = "noun"
    file_1 = "parsed_nouns.txt"  
    file_2 = "objects_vg_vocab.txt"
    method = "synset"
    use_lemma = True if method == "synset" else True # whether use lemma for synset matching
    index_mapped = [] # the index of word_list2 which has been mapped to some category in word_list1

    word_list1 = ['__background__'] + [str(line.strip().split(',')[0]) for line in open(file_1).readlines()]  # root words
    word_list2 = [str(line.strip().split(',')[0]) for line in open(file_2).readlines()]  # all words to be merged
    word_list3 = [str(line.strip()) for line in open(file_2).readlines()]  # same as list2 but include frequency
    print(len(word_list1))
    print(len(word_list2))

    word_map = {} # the mapping of word (string), used for visualization
    cls_ind_map = {} # the mapping class index in dataset, used for model training and testing
    for i in range(len(word_list1)):
        word_map[word_list1[i]] = []
        cls_ind_map[i] = []
        this_got_matched = False

        # Priority 0: manual dictionary match
        for j in range(len(word_list2)):
            if word_list1[i] in noun_manual_dict and noun_manual_dict[word_list1[i]] == word_list2[j]:
                word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                print('Manual Map: {} <-- {}'.format(word_list2[j], word_list1[i]))
                break

        # Priority 1: direct string match
        for j in range(len(word_list2)):
            if word_list1[i] == word_list2[j]:
                word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                break
        
        # Priority 2: root string match 
        if not this_got_matched:
            for j in range(len(word_list2)):
                if word_list1[i].split()[-1] == word_list2[j].split()[-1]:
                    word_map[word_list1[i]].append(word_list2[j])
                    cls_ind_map[i].append(j)
                    index_mapped.append(j)
                    this_got_matched = True
                    break

        # Priority 3: synset match 
        if not this_got_matched:
            for j in range(len(word_list2)):
                if method == "synset":
                    if wordnet_preprocess(word_list1[i], word_list2[j], pos=wn.NOUN, use_lemma=use_lemma):
                        word_map[word_list1[i]].append(word_list2[j])
                        cls_ind_map[i].append(j)
                        index_mapped.append(j)
                        #print('Synset Map: {} <-- {}'.format(word_list2[j], word_list1[i]))

    # save the word (string) matching results
    file_name = 'word_map_{}'.format(concept_type) if use_lemma else 'word_map_no_lemma'
    with open(file_name+'.txt', 'w') as text_file:
        for i in range(len(word_list1)):
            this_map = word_map[word_list1[i]]
            #if len(this_map) == 0:
                #continue
            text_file.write(word_list1[i])
            for j in range(len(this_map)):
                text_file.write("," + this_map[j])
            text_file.write("\n")

    # save the class index matching results
    #np.save(file_name+'.npy', cls_ind_map)
    cap2vg_cls_map['object_cls'] = cls_ind_map

    # show the category in word_list2 which was matched successfully
    print("\nMatch successfully:")
    cnt = 0
    with open("match_successfully_{}.txt".format(concept_type), 'w') as f:
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
    with open("match_failed_{}.txt".format(concept_type), 'w') as f:
        for i, item in enumerate(word_list2):
            if i not in index_mapped:
                print(word_list3[i])
                f.write(word_list3[i]+"\n")
                cnt += 1
        print("\nIn total, {} categories failed to be matched!\n".format(cnt))
        f.write("\nIn total, {} categories failed to be matched!\n".format(cnt))

    print("Done!\n\n")

    ############################################################################ 
    #### 2. map predicates
    ############################################################################
    concept_type = "predicate"
    file_1 = "parsed_predicates.txt"  
    file_2 = "predicates_vg_vocab.txt"
    alias_file = "predicate_alias.txt"
    lem = WordNetLemmatizer()
    index_mapped = [] # the index of word_list2 which has been mapped to some category in word_list1

    word_list1 = ['__background__'] + [str(line.strip().split(',')[0]) for line in open(file_1).readlines()]  # root words
    word_list2 = [str(line.strip().split(',')[0]) for line in open(file_2).readlines()]  # all words to be merged
    word_list3 = [str(line.strip()) for line in open(file_2).readlines()]  # same as list2 but include frequency
    print(len(word_list1))
    print(len(word_list2))
    
    # predicate alias dictionary
    alias_dict, vocab_list = make_alias_dict(alias_file)
    word_map = {} # the mapping of word (string), used for visualization
    cls_ind_map = {} # the mapping class index in dataset, used for model training and testing
    for i in range(len(word_list1)):
        word_map[word_list1[i]] = []
        cls_ind_map[i] = []
        this_got_matched = False

        # Priority 0: manual dictionary match
        for j in range(len(word_list2)):
            if word_list1[i] in rel_manual_dict and rel_manual_dict[word_list1[i]] == word_list2[j]:
                word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                print('Manual Map: {} <-- {}'.format(word_list2[j], word_list1[i]))
                break

        # Priority 1: direct string match
        for j in range(len(word_list2)):
            if word_list1[i] == word_list2[j]:
                word_map[word_list1[i]].append(word_list2[j])
                cls_ind_map[i].append(j)
                index_mapped.append(j)
                this_got_matched = True
                break
        
        # Priority 2: VG predicate alias file match
        if not this_got_matched:
            for j in range(len(word_list2)):
                if (word_list1[i] in alias_dict and alias_dict[word_list1[i]] == word_list2[j]) or\
                    (word_list2[j] in alias_dict and alias_dict[word_list2[j]] == word_list1[i]):
                    word_map[word_list1[i]].append(word_list2[j])
                    cls_ind_map[i].append(j)
                    index_mapped.append(j)
                    this_got_matched = True
                    break

        # Priority 3: Lemmatization match
        if not this_got_matched:
            for j in range(len(word_list2)):
                if predicate_match(word_list1[i], word_list2[j], lem=lem):
                    word_map[word_list1[i]].append(word_list2[j])
                    cls_ind_map[i].append(j)
                    index_mapped.append(j)
                    print('Lemmatization Map: {} <-- {}'.format(word_list2[j], word_list1[i]))
                    this_got_matched = True

    # save the word (string) matching results
    file_name = 'word_map_{}'.format(concept_type) if use_lemma else 'word_map_no_lemma'
    with open(file_name+'.txt', 'w') as text_file:
        for i in range(len(word_list1)):
            this_map = word_map[word_list1[i]]
            #if len(this_map) == 0:
                #continue
            text_file.write(word_list1[i])
            for j in range(len(this_map)):
                text_file.write("," + this_map[j])
            text_file.write("\n")

    # save the class index matching results
    #np.save(file_name+'.npy', cls_ind_map)
    cap2vg_cls_map['predicate_cls'] = cls_ind_map

    # show the category in word_list2 which was matched successfully
    print("\nMatch successfully:")
    cnt = 0
    with open("match_successfully_{}.txt".format(concept_type), 'w') as f:
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
    with open("match_failed_{}.txt".format(concept_type), 'w') as f:
        for i, item in enumerate(word_list2):
            if i not in index_mapped:
                print(word_list3[i])
                f.write(word_list3[i]+"\n")
                cnt += 1
        print("\nIn total, {} categories failed to be matched!\n".format(cnt))
        f.write("\nIn total, {} categories failed to be matched!\n".format(cnt))

    print("Done!")

    # save the results
    np.save("{}2VG_word_map.npy".format(dataset), cap2vg_cls_map)