import os
import json
import pprint
import logging
import multiprocessing as mp
from gensim.models import Word2Vec
from numpy.linalg import norm                                                      
from scipy.spatial.distance import cosine  
import spacy.en
nlp = spacy.en.English(load_vectors=False, entity=False)

#### Corpus Generators ####
## allows for repeating corpus generators
def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen


def tokenize_doc_map(doc):
    doc = json.loads(str(doc, encoding='utf-8-sig'))
    doc['toks'] = [[w.text for w in sent]
            for sent in nlp(doc['review_text']).sents]
    return doc


def mdsd_corpus(mdsd_path):
    p = mp.Pool(2)
    mdsd_files = os.listdir(mdsd_path)
    for mdsd_file in mdsd_files:
        logging.info('processing {}'.format(mdsd_file))
        f = open(mdsd_path+mdsd_file, 'rb')
        for doc in p.imap(tokenize_doc_map, f.readlines()):
            for sent in doc['toks']:
                yield [w.lower() for w in sent]

if __name__ == '__main__':
    mdsd_path = 'data/Multi_Domain_Sentiment_Dataset/json_files/'
    w2v_model_path = 'models/mdsd_w2v_model.txt'
    logging.basicConfig(filename='word2vec_training.log',                       
        level=logging.INFO,                                                     
        format='%(asctime)s %(levelname)s: %(message)s',                        
        datefmt='%m/%d/%Y %H:%M:%S') 
    model_w2v = Word2Vec(min_count=5, workers=2)                  
    model_w2v.build_vocab(mdsd_corpus(mdsd_path))
    model_w2v.train(mdsd_corpus(mdsd_path))
    model_w2v.save(w2v_model_path)
