import os
import json
import pprint
import logging
import multiprocessing as mp
from gensim.models import Word2Vec
from numpy.linalg import norm                                                      
from scipy.spatial.distance import cosine  
import spacy.en
from sklearn.externals import joblib
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


def mdsd_corpus(mdsd_path, p=None):
    if p == None:
        p = mp.Pool(mp.cpu_count())
    mdsd_files = os.listdir(mdsd_path)
    for mdsd_file in mdsd_files:
        logging.info('processing {}'.format(mdsd_file))
        f = open(mdsd_path+mdsd_file, 'rb')
        for doc in p.imap_unordered(tokenize_doc_map, f.readlines()):
            for sent in doc['toks']:
                yield [w.lower() for w in sent]


def identity(x):
    return x


if __name__ == '__main__':
    w2v = False
    lsi = True
    tsne = False
    
    mdsd_path = 'data/Multi_Domain_Sentiment_Dataset/json_files/'
    logging.basicConfig(filename='word2vec_training.log',                       
        level=logging.INFO,                                                     
        format='%(asctime)s %(levelname)s: %(message)s',                        
        datefmt='%m/%d/%Y %H:%M:%S') 

    if w2v:
        w2v_model_path = 'models/mdsd_w2v_model.txt'
        model_w2v = Word2Vec(min_count=3, workers=mp.cpu_count())                  
        model_w2v.build_vocab(mdsd_corpus(mdsd_path))
        model_w2v.train(mdsd_corpus(mdsd_path))
        model_w2v.save(w2v_model_path)
        joblib.dump(model_w2v.syn0norm, 'w2v_syn0norm.np')

    if lsi:
        p = mp.Pool(mp.cpu_count())
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        lsi = TruncatedSVD(n_components=400)
        vec = TfidfVectorizer(tokenizer=identity,
                preprocessor=identity,
                ngram_range=(1,5), min_df=3)
        term_doc_m = vec.fit(mdsd_corpus(mdsd_path, p)).T
        p.terminate()
        term_doc_m_lsi = lsi.fit(term_doc_m)
        joblib.dump(term_doc_m_lsi, 'term_doc_m_lsi')
        joblib.dump(term_doc_m, 'term_doc_m')
        joblib.dump(vec, 'lsi_vec')
        joblib.dump(lsi, 'lsi_model')

    if tsne:
        from sklearn.manifold import TSNE
        from sklearn.externals import joblib
        tsne = TSNE(n_components=2)
        syn0norm = joblib.load('w2v_syn0norm.np')
