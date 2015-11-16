### Convert Multi-Domain Sentiment Dataset
### in pseudo-XML format to well-structured json
import json
import argparse
import logging
import codecs
import multiprocessing as mp

def get_line_nos(file_path, pattern, encoding='iso-8859-1'):
    """get line numbers in file that match pattern"""
    f = codecs.open(file_path, 'r', encoding=encoding)
    indices = [i for i, line in enumerate(f) if line.strip() == pattern]
    f.close()
    return indices


def parse_xml_doc(doc):
    """ 
    pseudo-parsing function that uses the fact that every tag is on 
    its own newline to deal with the pseudo-xml 
    (WITH LOVE, WITH LOVE, THANKS FOR THE DATA!)
    """
    begin = '<{}>'
    end = '</{}>'
    all_tags = [ 'rating', 'product_type', 'helpful', 'asin', 'title', 
                 'review_text', 'reviewer_location', 'date', 
                 'reviewer', 'product_name', 'unique_id' ]
    doc_out = {t:[] for t in all_tags}
    curr_tag = False
    for t in all_tags:
        for line in doc[1:-1]:
            if line.strip() == begin.format(t):
                curr_tag = True
            elif line.strip() == end.format(t):
                curr_tag = False
            elif curr_tag:
                doc_out[t].append(line)
    doc_out = {k:'\n'.join(v).strip() for k,v in doc_out.items()}
    return doc_out


def get_xml_docs(file_path, ends, encoding='iso-8859-1'):
    """returns iterable of xml strings"""
    f = codecs.open(file_path, 'r', encoding=encoding)
    docs = []
    doc = []
    j = 0
    for i, l in enumerate(f):
        doc.append(l)
        if i == ends[j]:
            doc_out = doc[:]
            doc = []
            j+=1
            yield doc_out


def conv_doc_to_json(doc):
    parsed_doc = parse_xml_doc(doc)
    doc_json_dump = json.dumps(parsed_doc)+u'\n'
    return doc_json_dump


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description="script to convert Multi-Domain Sentiment Dataset \n"+\
           "from malformed pseudo-xml to wellformed json;")
    parser.add_argument('infile', 
            help="path to multi-domain sentiment dataset file")
    parser.add_argument('--outfile', '-o', help="save path")
    args = parser.parse_args()
    if args.outfile is None:
        outfile_path = './{}'.format(
                '_'.join(args.infile.split('/')[-2:])) + '.json'
    else:
        outfile_path = '{}/{}'.format(args.outfile, 
                '_'.join(args.infile.split('/')[-2:])) + '.json'
    print(outfile_path)

    print("getting doc line boundaries")
    ends = get_line_nos(args.infile, "</review>")
    print("getting doc xml strings")
    docs_xml = get_xml_docs(args.infile, ends)
    outfile = codecs.open(outfile_path, 'w', encoding='utf-8-sig')
    p = mp.Pool(2*mp.cpu_count())
    print("parsing xml, converting to json and saving to file")
    for doc in p.imap(conv_doc_to_json, docs_xml):
        outfile.write(doc)
    outfile.close()
    p.terminate()
