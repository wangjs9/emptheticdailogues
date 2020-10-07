import stanfordnlp, re
nlp = stanfordnlp.Pipeline()
from stanfordcorenlp import StanfordCoreNLP
core_nlp = StanfordCoreNLP('C:/Users/csjwang/Documents/Python Scripts/stanford-corenlp-4.1.0')
text = "What a difference a year makes. Last year one evening my family was at home when a tree fell on the house and broke through the ceiling."

def get_content(parse):
    parse = [ele.strip() for ele in re.split('([()])', parse) if ele not in ['', ' ', '\s+']]
    VP_index = [idx for idx, ele in enumerate(parse) if ele=='VP']
    NP_index = [idx for idx, ele in enumerate(parse) if ele=='NP']
    VP_list = []
    NP_list = []

    for idx in VP_index:
        VP_list.append(get_pos(parse[idx:]))
    for idx in NP_index:
        NP_list.append(get_pos(parse[idx:]))

    return VP_list, NP_list

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

doc = nlp(text)
sentences = doc.sentences
# dependency = [[(dep_edge[0].index, dep_edge[1]) for dep_edge in sentence.dependencies] for sentence in sentences]
sentence = [' '.join([token.words[0].text for token in sentence.tokens]) for sentence in sentences]

parse = core_nlp.parse(sentence[1])
core_nlp.close()
print("__________")
sentences = parse[0]
VP, NP = get_content(sentences)
print(VP)
print(NP)

