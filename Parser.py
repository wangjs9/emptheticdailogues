import pandas as pd
import numpy as np
import json, re, time
from datetime import timedelta

def get_dif_time(start_time):
    """get time difference"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def constituency(data=None):
    import requests
    import psutil
    port = None
    for port_candidate in range(9000, 65535):
        if port_candidate not in [conn.laddr[1] for conn in psutil.net_connections()]:
            port = port_candidate
            break
    url = 'http://localhost:' + str(port)

    properties = {'annotators': 'parse', 'outputFormat': 'json'}
    params = {'properties': str(properties), 'pipelineLanguage': 'en'}

    r = requests.post(url, params=params, data=data, headers={'Connection': 'close'})
    r_dict = json.loads(r.text)

    return r_dict

def helper(input_file_name, output_file_name, prompt=True):
    """
    input file is a csv file;
    output file is a json file.
    """
    data = pd.read_csv(input_file_name, sep='\t', header=0, index_col=False, encoding='UTF-8')
    data_len = data.shape[0]
    assumpte_size = 1
    conv_ids = np.empty([len(data), assumpte_size], dtype=np.uint16)
    lengths = np.empty([len(data), assumpte_size], dtype=np.uint16)
    sentences = np.empty([len(data), assumpte_size], dtype=np.object)
    lemmas = np.empty([len(data), assumpte_size], dtype=np.object)
    VPs = np.empty([len(data), assumpte_size], dtype=np.object)
    NPs = np.empty([len(data), assumpte_size], dtype=np.object)
    dependencies = np.empty([len(data), assumpte_size], dtype=np.object)

    start_time = time.time()
    for index, row in data.iterrows():
        if index % 1000 == 0 and index != 0:
            print(' {} row/{} rows, time cost: {}'.format(index, data_len, get_dif_time(start_time)))
        if not prompt:
            emotion, idx, paragraph = row
        else:
            emotion, paragraph = row
            idx = index
        parser = Parser(paragraph)
        sentence = parser.sentences
        length = parser.length
        lemma = parser.lemma
        constituency = parser.constituency # (sentence_num, 2)
        VP = [const['VP'] for const in constituency]
        NP = [const['NP'] for const in constituency]
        dependency = parser.dependency # (sentence_num, sentence_len)

        try:
            conv_ids[index, 0] = idx
            lengths[index, 0] = length
            sentences[index, 0] = sentence
            lemmas[index, 0] = lemma
            VPs[index, 0] = VP
            NPs[index, 0] = NP
            dependencies[index, 0] = dependency
        except:
            print(idx)
            print(index)

    sen_dict = {
        'conv_ids': conv_ids,
        'emotions': np.array(data.loc[:, ['emotion']]),
        'lengths': lengths,
        'sentences': sentences,
        'lemmas': lemmas,
        'VPs': VPs,
        'NPs': NPs,
        'dependencies': dependencies,
    }
    np.save(output_file_name, sen_dict)

class Parser(object):
    def __init__(self, paragraph):
        paragraph = str(paragraph)
        sentences = nlp(paragraph).sentences
        self._lemma = [' '.join([token.words[0].lemma for token in sentence.tokens if token.words[0].lemma != None]) for sentence in sentences]
        self._sentences = [sen for sen in self._lemma if sen != '']
        self._sentences = [' '.join([token.words[0].text for token in sentence.tokens if token.words[0] != None]) for sentence in sentences]
        self._sentences = [sen for sen in self._sentences if sen != '']
        self._length = len(sentences)
        self._dependency = [[(dep_edge[0].index, dep_edge[1]) for dep_edge in sentence.dependencies] for sentence in sentences]
        self._constituency = list()
        for sentence in self._sentences:
            parse = core_nlp.parse(sentence)
            self._constituency.append(self.get_content(parse[0]))

    @property
    def lemma(self):
        """Return lemmatization of sentences, and the value is a list, whose length is the number of sentences."""
        return self._lemma

    @property
    def sentences(self):
        return self._sentences

    @property
    def length(self):
        """Return the number of sentences in the paragraph, and the value is an Integer."""
        return self._length

    @property
    def constituency(self):
        """Return constituency relationship, and the value is a dictionary.
        [[...], [...]] the last one is VP"""
        return self._constituency

    @property
    def dependency(self):
        """Return dependencies relationship, and the value is a list:
        [[(last_node_index, pos), ...],...], the index is (list_index+1)."""
        return self._dependency

    def get_content(self, parse):
        def get_pos(parse):
            out = 0
            target = []
            while out >= 0:
                top = parse.pop(0)
                if top == '(':
                    out += 1
                elif top == ')':
                    out -= 1
                else:
                    try:
                        _, word = top.split()
                        target.append(word)
                    except:
                        True
            return ' '.join(target)

        parse = [ele.strip() for ele in re.split('([()])', parse) if ele not in ['', ' ', '\s+']]
        VP_index = [idx for idx, ele in enumerate(parse) if ele == 'VP']
        NP_index = [idx for idx, ele in enumerate(parse) if ele == 'NP']
        VP_list = []
        NP_list = []

        for idx in VP_index:
            VP_list.append(get_pos(parse[idx:]))
        for idx in NP_index:
            NP_list.append(get_pos(parse[idx:]))

        return {'VP': VP_list, 'NP': NP_list}

import stanfordnlp
nlp = stanfordnlp.Pipeline()
from stanfordcorenlp import StanfordCoreNLP
core_nlp = StanfordCoreNLP('C:/Users/csjwang/Documents/Python Scripts/stanford-corenlp-4.1.0')
# helper('./$empatheticdialogues/valid_prompt_emotion_version.csv', './$empatheticdialogues/valid_prompt_parser_version.npy')
# helper('./$empatheticdialogues/prompt_emotion_version.csv', './$empatheticdialogues/prompt_parser_version.npy')
# helper('./$empatheticdialogues/valid_emotion_version.csv', './$empatheticdialogues/valid_parser_version.npy', False)
# helper('./$empatheticdialogues/emotion_version.csv', './$empatheticdialogues/parser_version.npy', False)

core_nlp.close()

