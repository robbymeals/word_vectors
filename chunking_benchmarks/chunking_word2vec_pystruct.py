import sys
import logging
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from itertools import chain
from nltk.corpus import conll2000
from subprocess import Popen, PIPE
from shlex import split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelBinarizer
from pystruct.learners import OneSlackSSVM, FrankWolfeSSVM, NSlackSSVM
from pystruct.learners import SubgradientSSVM, StructuredPerceptron
from pystruct.models import ChainCRF


### relative ngram feature definitions
### for using in all_rel_ngram_feats below
UGRAMPOSITS = [[-3,],[-2,],[-1,],[0,],[1],[2],[3,]]
BGRAMPOSITS = [[-3,-2],[-2,-1],[-1,0],[0,1],[1,2],[2,3],]
TGRAMPOSITS = [[-3,-2,-1],[-2,-1,0],[-1,0,1],[0,1,2],[1,2,3],]
ALLNGRAMPOSITS = UGRAMPOSITS + BGRAMPOSITS 
#+ TGRAMPOSITS


def rel_ngram_feat(seq, curr_idx, rel_pos_list, feat_lab, div=u'|'):                        
    """
    given sequence, current index position, a list of relative position
    indices as ngram group and a feature label string,
    return relative ngram feature as string:
    >>> rel_ngram_feat([u'word0',u'word1',u'word2'], 0, [0,1], u'w')
    u'w[0]|w[1]=word0|word1'
    """
    feat_idx_list = [curr_idx + el for el in rel_pos_list]                      
    if any([el < 0 for el in feat_idx_list]):                                   
        return None                                                             
    if any([el >= len(seq) for el in feat_idx_list]):                           
        return None                                                             
    else:                                                                       
        one = div.join([u'{0}[{1}]'.format(feat_lab, el)                       
            for el in rel_pos_list])                                            
        two = div.join([str(seq[idx]) for idx in feat_idx_list])                                          
        return u'{0}={1}'.format(one,two)                                       
                                                                                

def all_rel_ngram_feats(seq, idx, feat_lab, div=u'|'):                                                         
    """
    given sequence, current index position and feature label
    return string representations of all unigram, bigram 
    and trigram relative ngram features
    for that sequence for that index position
    """
    features = [rel_ngram_feat(seq, idx, ngrampos, feat_lab, div=div)            
            for ngrampos in ALLNGRAMPOSITS]                                     
    features = [el for el in features if el != None]                            
    return features        


def iob_2_ngram_feats(sentence, lcase=True, combine=True, div=u'|'):
    """
    Given a sentence from conll2000 corpus
    in list of tuples format, where each
    tuple consists of a word token, a POS tag,
    and a BIO chunking tag, output list of tuples
    of string representation of token and part of speech tag 
    relative ngram features.
    """
    toks = [t[0] for t in sentence]
    tags = [t[1] for t in sentence]
    tok_feats = [all_rel_ngram_feats(toks, i, u'w', div=div) 
            for i in range(len(toks))]
    tag_feats = [all_rel_ngram_feats(tags, i, u'p', div=div) 
            for i in range(len(tags))]
    features = list(zip(tok_feats, tag_feats))
    if combine:
        features = [a+b for a,b in features]
    return features


## identity function for vectorizer
def identity(x):
    return x


## run conlleval perl script in subprocess
## returning stdout string results
def conlleval_results(preds_file):
    """
    given path to preds file in conlleval format,
    return string of eval results
    """
    cmd1 = "cat {}".format(preds_file) 
    cmd2 = "perl conlleval.txt"
    p1 = Popen(split(cmd1), stdout=PIPE)
    p2 = Popen(split(cmd2), stdin=p1.stdout, stdout=PIPE)
    return str(p2.stdout.read(), 'UTF-8')


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


## flattened token string features
def flat_transf_str_feats_corpus(corpus):
    for sentence in corpus:
        for token in iob_2_ngram_feats(sentence):
            yield token


## structured token matrix features
def transf_mtrx_corpus(corpus, vec):
    for sentence in corpus:
        yield vec.transform(iob_2_ngram_feats(sentence))


## structured tsvd features
@multigen
def tsvd_corpus(corpus, vec, tsvd):
    for sentence in corpus:
        yield tsvd.transform(vec.transform(iob_2_ngram_feats(sentence)))


## crfsuite string features
def crfsuite_fmt(corpus):
    for sentence in corpus:
        features = iob_2_ngram_feats(sentence)
        features = u'\n'.join([
            sentence[i][2] + u'\t' + u'\t'.join(tok_feats) 
            for i, tok_feats in enumerate(features)])+u'\n\n'
        yield features


## vowpal wabbit searn/learning2search string features
def searn_fmt(corpus, label2id):
    for sentence in corpus:
        features = iob_2_ngram_feats(sentence, combine=False, div=u'+')
        tok_feats = [u'|w '+u' '.join(features[i][0]).replace(u':','-SC-')
                for i in range(len(features))]
        pos_feats = [u'|p '+u' '.join(features[i][1]).replace(u':','-SC-')
                for i in range(len(features))]
        features = u'\n'.join( [u'{} {} {}'.format(
            label2id[sentence[i][2]], tok_feats[i], pos_feats[i])
            for i in range(len(sentence))]) + u'\n\n'
        yield features


## svc predictions to structure
def svc_preds(corpus, vec, model):
    for sentence in corpus:
        yield model.predict(vec.transform(iob_2_ngram_feats(sentence)))


## preds and original label in format expected by conllevel perl script
def conlleval_fmt(corpus, preds):
    for i, sentence in enumerate(corpus):
        pred = preds[i]
        sentence_w_pred = [a+(b,) for a,b in zip(sentence, pred)]
        yield u'\n'.join([u'{} {} {} {}'.format(* tok) 
            for tok in sentence_w_pred]) + u'\n\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--crfsuite', action='store_true')
    parser.add_argument('--searn', action='store_true')
    parser.add_argument('--lsi', action='store_true')
    parser.add_argument('--oneslack', action='store_true')
    parser.add_argument('--subgrad', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--evals', action='store_true')
    args = parser.parse_args()

    from gensim.models import Word2Vec

    ## get train and test set indices
    print('loading conll2000 datasets')
    iob_train = conll2000.iob_sents('train.txt')
    iob_test = conll2000.iob_sents('test.txt')

    glove = Word2Vec.load('../crowdflower/vecs/glove_840B_w2v.pkl')
    glove_mean = np.mean(glove.syn0)
    parts_of_speech = set(flatten([[w[1] for w in sent] for sent in iob_train]))
    pos_vecs = {pos:np.vstack(flatten([[glove[w[0]] for w in sent
        if all([w[0] in glove.vocab, w[1] == pos])]
        for sent in iob_train])) for pos in parts_of_speech}
    pos_vecs_means = {pos:np.mean(pos_vecs[pos], 0) for pos in pos_vecs.keys()}

    def get_word_vec(word):
        try:
            return glove[word[0]]
        except KeyError:
            try:
                print(word)
                return pos_vecs_means[word[1]]
            except KeyError:
                print(word[1])
                return glove_mean
            
    iob_train_vecs = [np.vstack([get_word_vec(iob_train[j][i])
        for i in range(len(iob_train[j]))]) for j in range(len(iob_train))]
    iob_test_vecs = [np.vstack([get_word_vec(iob_test[j][i])
        for i in range(len(iob_test[j]))]) for j in range(len(iob_test))]

    from itertools import combinations_with_replacement
    def get_graph_edges(feature_matrix, max_window=3):
        if len(feature_matrix.shape) == 1:
            nodes = list(range(1))
        else:
            nodes = list(range(feature_matrix.shape[0]))
        edges = [edge for edge in combinations_with_replacement(nodes, 2)
                if abs(edge[0] - edge[1]) <= max_window]
        edges = np.array(edges)
        return edges

    y_train_flat = list(chain(* [[tok[2] for tok in sentence] 
        for sentence in iob_train]))
    y_test_flat = list(chain(* [[tok[2] for tok in sentence] 
        for sentence in iob_test]))
    label2id = {l:i for i,l in enumerate(set(y_train_flat))}
    id2label = {l:i for i,l in label2id.items()}
    y_train = [np.array([label2id[tok[2]] for tok in sentence]) 
            for sentence in iob_train]

    svc = SVC()
    gs_svc = GridSearchCV(svc,
            param_grid={'C':[0.01,0.1,0.5,1.0,2.0,10.0]},
            verbose=1)
    gs_svc.fit(X_train_flat, y_train_flat)

    test_conll_svc = conlleval_fmt(iob_test, svc_test_preds) 
    test_conll_svc_file = open('test_conll_svc.txt', 'wb')
    for sentence in test_conll_svc:
        test_conll_svc_file.write(bytes(sentence, 'UTF-8'))
    test_conll_svc_file.close()
    print(conlleval_results('test_conll_svc.txt'))

    if args.linear:
        #### LinearSVC Benchmark ####
        ## train vectorizer
        print('training count vectorizer')
        vec = CountVectorizer(tokenizer=identity, preprocessor=identity, min_df=5)
        X_train_flat = vec.fit_transform(flat_transf_str_feats_corpus(iob_train))
        y_train_flat = list(chain(* [[tok[2] for tok in sentence] 
                            for sentence in iob_train]))
        assert len(y_train_flat) == X_train_flat.shape[0], "y_train and X_train shapes not aligned"

        ### fit flat linear svc
        print('fitting flat linear svc')
        svc = LinearSVC(verbose=args.verbose)
        svc.fit(X_train_flat, y_train_flat)

        ### evaluate flat benchmark results
        test_conll_svc = conlleval_fmt(iob_test, 
                list(svc_preds(iob_test, vec, svc)))
        test_conll_svc_file = open('test_conll_svc.txt', 'wb')
        for sentence in test_conll_svc:
            test_conll_svc_file.write(bytes(sentence, 'UTF-8'))
        test_conll_svc_file.close()
        print(conlleval_results('test_conll_svc.txt'))

    if args.crfsuite:
        #### crfsuite Benchmark ####
        ### write crfsuite training set file
        print('generating crfsuite training and test sets')
        train_crfsuite = crfsuite_fmt(iob_train)
        train_crfsuite_file = open('train_crfsuite.txt', 'wb')
        for sentence in train_crfsuite:
            train_crfsuite_file.write(bytes(sentence, 'UTF-8'))
        train_crfsuite_file.close()

        ### write crfsuite test set file
        test_crfsuite = crfsuite_fmt(iob_test)
        test_crfsuite_file = open('test_crfsuite.txt', 'wb')
        for sentence in test_crfsuite:
            test_crfsuite_file.write(bytes(sentence, 'UTF-8'))
        test_crfsuite_file.close()

        ### fit crfsuite model, writing iteration results to stdout
        print("fitting crfsuite model")
        crfsuite_fit = "crfsuite learn -e 2 -m crfsuite_conll2000.m "+\
                "train_crfsuite.txt test_crfsuite.txt"
        crfsuite_fit_p = Popen(split(crfsuite_fit), stdout=PIPE)
        while True:
            o = str(crfsuite_fit_p.stdout.readline(), 'UTF-8')
            if o == '' and crfsuite_fit_p.poll() != None: 
                break
            if args.verbose:
                sys.stdout.write(o)
            else:
                pass
        
        ### get crfsuite preds and conll performance
        print("getting crfsuite preds")
        crfsuite_tag = "crfsuite tag -m crfsuite_conll2000.m test_crfsuite.txt"
        crfsuite_tag_p = Popen(split(crfsuite_tag), stdout=PIPE)
        print("reading in preds")
        test_crfsuite_preds = str(crfsuite_tag_p.stdout.read(), 'UTF-8')
        print("formatting preds")
        test_crfsuite_preds = [preds.split('\n') for preds 
                in test_crfsuite_preds.split('\n\n') if preds != '']
        print("outputting preds")
        test_conll_crfsuite = conlleval_fmt(iob_test, test_crfsuite_preds)
        test_conll_crfsuite_file = open('test_conll_crfsuite.txt', 'wb')
        for sentence in test_conll_crfsuite:
            test_conll_crfsuite_file.write(bytes(sentence, 'UTF-8'))
        test_conll_crfsuite_file.close()
        print(conlleval_results('test_conll_crfsuite.txt'))

    if args.searn:
        #### SEARN/Learning2Search Benchmark ####
        ## string labels to label ids                                                                                                                    
        y_train_flat = list(chain(* [[tok[2] for tok in sentence] 
                            for sentence in iob_train]))
        y_test_flat = list(chain(* [[tok[2] for tok in sentence] 
                            for sentence in iob_test]))
        label2id = {l:i+1 for i,l in enumerate(set(y_train_flat+y_test_flat))}                                                                                         
        id2label = {l:i for i,l in label2id.items()}
        print('generating searn training and test sets')
        train_searn = searn_fmt(iob_train, label2id)
        train_searn_file = open('train_searn.txt', 'wb')
        for sentence in train_searn:
            train_searn_file.write(bytes(sentence, 'UTF-8'))
        train_searn_file.close()

        ### write searn test set file
        test_searn = searn_fmt(iob_test, label2id)
        test_searn_file = open('test_searn.txt', 'wb')
        for sentence in test_searn:
            test_searn_file.write(bytes(sentence, 'UTF-8'))
        test_searn_file.close()

        searn_fit = "vw -b 28 -k -c -d train_searn.txt --passes 4 "+\
            "--search_task sequence "+\
            "--search 45 --search_neighbor_features -2:w,-1:w,1:w,2:w "+\
            "--affix -3w,-2w,-1w,+3w,+2w,+1w -f searn.model"
        searn_fit_p = Popen(split(searn_fit), stdout=PIPE)
        while True:
            o = str(searn_fit_p.stdout.readline(), 'UTF-8')
            if o == '' and searn_fit_p.poll() != None: 
                break
            if args.verbose:
                sys.stdout.write(o)
            else:
                pass
        
        searn_tag = "vw -t -i searn.model test_searn.txt -p test_preds_searn.txt"
        searn_tag_p = Popen(split(searn_tag), stdout=PIPE)
        while True:
            o = str(searn_tag_p.stdout.readline(), 'UTF-8')
            if o == '' and searn_tag_p.poll() != None: 
                break
            if args.verbose:
                sys.stdout.write(o)
            else:
                pass
        
        f = open('test_preds_searn.txt','rb')
        test_preds_searn = [[id2label[int(p)] for p in l.split()] 
            for l in str(f.read(), 'UTF-8').split('\n') if l.strip() != '']
        f.close()
        test_conll_searn = conlleval_fmt(iob_test, test_preds_searn)
        test_conll_searn_file = open('test_conll_searn.txt', 'wb')
        for sentence in test_conll_searn:
            test_conll_searn_file.write(bytes(sentence, 'UTF-8'))
        test_conll_searn_file.close()
        print(conlleval_results('test_conll_searn.txt'))

    if args.lsi:
        ### pystruct with tsvd/lsi Benchmark ###
        ### fit LSI on 3d ngram features
        vec = CountVectorizer(tokenizer=identity, 
                    preprocessor=identity, 
                    min_df=5)
        X_train_flat = vec.fit_transform(
                flat_transf_str_feats_corpus(iob_train))
        y_train_flat = list(chain(* [[tok[2] for tok in sentence] 
                            for sentence in iob_train]))
        assert len(y_train_flat) == X_train_flat.shape[0], \
                "y_train and X_train shapes not aligned"
        tsvd = TruncatedSVD(n_components=200)
        tsvd.fit(X_train_flat)
        X_train_tsvd = tsvd_corpus(iob_train, vec, tsvd)
        X_test_tsvd = tsvd_corpus(iob_test, vec, tsvd)
    
        ## string labels to label ids
        label2id = {l:i for i,l in enumerate(set(y_train_flat))}
        id2label = {l:i for i,l in label2id.items()}
        y_train = [np.array([label2id[tok[2]] for tok in sentence]) 
                for sentence in iob_train]

        if args.oneslack:
            ### fit oneslack ssvm
            crf = ChainCRF()
            os_ssvm = OneSlackSSVM(crf, verbose=args.verbose, n_jobs=-1,
                    use_memmapping_pool=0, 
                    show_loss_every=20,
                    tol=0.01, cache_tol=0.1)
            os_ssvm.fit(list(X_train_tsvd), y_train)
            test_os_ssvm_preds = [[id2label[i] for i in sent] 
                    for sent in os_ssvm.predict(X_test_tsvd)]
            test_conll_os_ssvm = conlleval_fmt(iob_test, test_os_ssvm_preds)
            test_conll_os_ssvm_file = open('test_conll_os_ssvm.txt', 'wb')
            for sentence in test_conll_os_ssvm:
                test_conll_os_ssvm_file.write(bytes(sentence, 'UTF-8'))
            test_conll_os_ssvm_file.close()
            print(conlleval_results('test_conll_os_ssvm.txt'))

        if args.subgrad:
            ### fit subgradient ssvm                                                       
            crf = ChainCRF()                                                            
            sg_ssvm = SubgradientSSVM(crf, max_iter=200, 
                    verbose=args.verbose, n_jobs=-1,                           
                    use_memmapping_pool=0, show_loss_every=20, shuffle=True)                                            
            sg_ssvm.fit(list(X_train_tsvd), y_train)                                    
            test_sg_ssvm_preds = [[id2label[i] for i in sent]                           
                    for sent in sg_ssvm.predict(X_test_tsvd)]                           
            test_conll_sg_ssvm = conlleval_fmt(iob_test, test_sg_ssvm_preds)            
            test_conll_sg_ssvm_file = open('test_conll_sg_ssvm.txt', 'wb')              
            for sentence in test_conll_sg_ssvm:                                         
                test_conll_sg_ssvm_file.write(bytes(sentence, 'UTF-8'))                                 
            test_conll_sg_ssvm_file.close()                                             
            print(conlleval_results('test_conll_sg_ssvm.txt'))       

    if args.evals:
        print(conlleval_results('test_conll_svc.txt'))
        print(conlleval_results('test_conll_crfsuite.txt'))
        print(conlleval_results('test_conll_searn.txt'))
        print(conlleval_results('test_conll_os_ssvm.txt'))
        print(conlleval_results('test_conll_sg_ssvm.txt'))       

