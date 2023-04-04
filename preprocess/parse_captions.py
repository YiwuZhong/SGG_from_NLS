import sys
sys.path.append("../")
import sng_parser
from pprint import pprint
import pickle
import time
import ipdb

# official example
graph = sng_parser.parse('A woman is playing the piano in the room.')
pprint(graph)
sng_parser.tprint(graph)  # we provide a tabular visualization of the graph.


# parse the given captions
input_file = 'vg_caption_triplet_POS_dict.pkl' #'COCO_caption_triplet_POS_dict.pkl' #'flickr30k_caption_triplet_POS_dict.pkl' #
output_file = "vg_caption_triplet_dict-spacy.pkl" #"COCO_caption_triplet_dict-spacy.pkl"
with open(input_file, 'rb') as f:
    cap_trip_pos_dict = pickle.load(f)

# process each caption, merge captions of each image, save the parsed triplets into dict
# ['A man laying in bed next to his dog.', [('man','lay in','bed'), ('man','lay next to','dog')], ['man', 'bed', 'dog']]
# the existing triplets are parsed by Stanford SG Parser, POS nouns are parsed by nltk
start = time.time()
cap_trip_spacy_dict = {} # save captions and parsed results
stats_dict = {}
stats_dict['nouns'] = {}
stats_dict['relations'] = {}
for i, img_id in enumerate(cap_trip_pos_dict):
    if i >= 100 and i % 100 == 0:
        print("{}th image: Used time {} for 100 images".format(i, time.time() - start))
        start = time.time()
    value = cap_trip_pos_dict[img_id]
    cap_trip_spacy_dict[img_id] = {}
    cap_trip_spacy_dict[img_id]['caption'] = []
    cap_trip_spacy_dict[img_id]['triplet'] = []

    for caption in value:
        # Note: use lemmatized noun/relation instead of original words, to merge the surface forms with same semantic concept
        sentence = caption[0].lower()
        graph = sng_parser.parse(sentence)
        entities = [item['lemma_head'] for item in graph['entities']] 
        triplets = [(entities[item['subject']], item['lemma_relation'], entities[item['object']]) for item in graph['relations']]
        cap_trip_spacy_dict[img_id]['caption'].append(sentence)
        cap_trip_spacy_dict[img_id]['triplet'].extend(triplets)
        
        # stanford_trip = caption[1]
        # print(sentence)
        # print(stanford_trip)
        # print(triplets)

        # record the frequency of each unique noun/relation
        for trp in triplets:
            # relation
            if trp[1] in stats_dict['relations']:
                stats_dict['relations'][trp[1]] += 1
            else:
                stats_dict['relations'][trp[1]] = 1
            # noun
            if trp[0] in stats_dict['nouns']:
                stats_dict['nouns'][trp[0]] += 1
            else:
                stats_dict['nouns'][trp[0]] = 1            
            if trp[2] in stats_dict['nouns']:
                stats_dict['nouns'][trp[2]] += 1
            else:
                stats_dict['nouns'][trp[2]] = 1

with open(output_file, 'wb') as f:
    pickle.dump(cap_trip_spacy_dict, f)
with open("stats-spacy.pkl", 'wb') as f:
    pickle.dump(stats_dict, f)

# write stats into text files
f = open("sorted_nouns.txt", "w")
nouns = [(item, stats_dict['nouns'][item]) for item in stats_dict['nouns']]
sorted_nouns = sorted(nouns, key=lambda x: x[1], reverse=True)
for item in sorted_nouns: f.write(item[0]+','+str(item[1])+'\n')
f.close()

f = open("sorted_relations.txt", "w")
rels = [(item, stats_dict['relations'][item]) for item in stats_dict['relations']]
sorted_rels = sorted(rels, key=lambda x: x[1], reverse=True)
for item in sorted_rels: f.write(item[0]+','+str(item[1])+'\n')
f.close()

ipdb.set_trace()
print('done')