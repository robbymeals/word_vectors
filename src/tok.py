import requests 
import json
import time
import logging
import urllib.parse as urlparse
import pprint
import string
from requests import ConnectionError
from utils import flatten
import spacy.en
nlp = spacy.en.English()


def open_requests_conn_pool():
    """
    open a global requests connection pool
    """
    global sess
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100)
    sess.mount('http://', adapter)


def upstreams_is_up():
    """
    test whether upstreams application is running
    """
    resp = sess.post(url=u"http://{UPSTREAMSHOST}:{UPSTREAMSPORT}/upstreams".format(**Settings))
    return resp.status_code == 200


def tokenize_doc(doc, twitter=False, content_field='content'):
    """
    return doc dict updated with tokens and pos tags
    """
    url = twagUrl if twitter else stagUrl
    try:
        doc.setdefault('retries', 0)
        doc['route'] = urlparse.urlparse(url).path
        data = dict()
        data['content'] = doc[content_field]
        json_headers = {'Content-type': 'application/json', 
			'Accept': 'text/plain'}
        toks_request = sess.post(url=url, data=json.dumps( dict(data.items()) ),
				 headers=json_headers)
        if toks_request.status_code != 200:
            from flask import current_app, Response, abort
            data = json.loads(toks_request.content)
            data['error_location'] = "upstreams application"
            if dir(current_app):
                data = json.dumps(data)
                res = Response(headers=json_headers)
                res.set_data(data)
                res.status_code = toks_request.status_code
                raise abort(res)
        doc.update(toks_request.json())
        return doc
    except ConnectionError:
        doc['retries'] += 1
        if doc['retries'] > 20:
            raise Exception("Request still failing after 20 retries, " + \
                    "is 'upstreams' app up?")
        time.sleep(0.5)
        doc = tokenize_doc(doc, twitter=twitter)
        logging.info(u"tokenizing '{}', {} retries".format(
            doc[content_field],doc['retries']))
        return doc


def tokenize_insight(insight, twitter=False):
    """
    return subject, property and context tokens
    """
    url = twagUrl if twitter else stagUrl
    insight = dict(insight.items())
    context = tokenize_doc({'content':insight['content']}, twitter=twitter)
    subj = tokenize_doc({'content':insight['subject']}, twitter=twitter)
    prop = tokenize_doc({'content':insight['property']}, twitter=twitter)
    insight['context_toks'] = flatten(context['toks'])
    insight['subj_toks'] = flatten(subj['toks'])
    insight['prop_toks'] = flatten(prop['toks'])
    return insight 


def get_sentence_indices(doc):
    """
    get sentence indices for a document
    """
    sents = [[0,len(sent)] for sent in doc['toks']]
    for i in range(1,len(sents)):
        sents[i] = [x+sents[i-1][-1] for x in sents[i]]
    doc['sentence_indices'] = sents
    return doc


def preprocess_doc(doc, twitter=False):
    """
    run all preprocessing on a document
    """
    doc.update(tokenize_doc({k:v for k,v in doc.items() 
        if k in [u'content','correct','error']}, twitter=twitter))
    doc = get_sentence_indices(doc)
    return doc


def preprocess_insight(insight, twitter=False):
    """
    run all preprocessing on an insight 
    """
    # if insight dict has content field and is empty string, 
    # this will be False
    hasContent = insight.get('content', None)
    hasInsight = insight.get('insight', None)
    if not hasContent and hasInsight:
        insight['content'] = insight['insight']
    insight = tokenize_insight(insight, twitter=twitter)
    return insight
