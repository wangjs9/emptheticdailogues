import pandas as pd
import numpy as np
from collections import Counter
import re, ast
# from senticnet.senticnet import SenticNet
# SN = SenticNet('en')

def read_file(file_name):
    conv_list = pd.read_csv(file_name, sep=',', header=0, index_col=False, encoding='unicode_escape')
    conv_ids = conv_list[['conv_id']].drop_duplicates()
    start_ids = conv_ids.index
    context = conv_list.loc[start_ids, ['context']]
    prompt = conv_list.loc[start_ids, ['prompt']]
    his_len = (start_ids[1:] - start_ids[:-1]).values
    his_len = np.append(his_len, conv_list.index[-1] - start_ids[-1] + 1)

    turn_len = max(his_len)
    multi_turn = np.empty([len(his_len), turn_len], dtype=np.object)
    start_ids = start_ids[1:]
    start_id = 0
    for i, id in enumerate(start_ids):
        utterances = conv_list.loc[start_id:(id-1), ['utterance']].values
        multi_turn[i, 0:his_len[i]] = utterances.reshape(1, -1)
        start_id = id
    np.save('valid_multi_turn_version.npy', multi_turn)

    his_len = np.array(his_len)
    np.save('valid_hist_len.npy', his_len)
    context = np.array(context)
    np.save('valid_context.npy', context)
    prompt = np.array(prompt)
    np.save('valid_prompt.npy', prompt)

def emotion_class(file_name, context_file, output_file):
    conv = np.load(file_name, allow_pickle=True)
    context = np.load(context_file, allow_pickle=True) # emotions
    conv = pd.DataFrame(conv) # turns
    col_names = conv.columns
    sen_emo = np.empty([len(col_names)*len(conv), 3], dtype=np.object)

    line = 0
    for idx, emotion in enumerate(context):
        conversation = conv.loc[idx, :]
        for turn in conversation:
            if turn == None:
                continue
            sen_emo[line, 0] = emotion[0]
            sen_emo[line, 1] = idx
            sen_emo[line, 2] = turn.replace('_comma_', ',')
            line += 1

    sen_emo = sen_emo[:line]
    sen_emo = pd.DataFrame(sen_emo, columns = ['emotion', 'conversation_id', 'post'])
    sen_emo.to_csv(output_file, sep='\t', index=False, encoding='UTF-8')

def statics(hist_len_file, context_file, class_file, emotion_version_file):
    hist_len = np.load(hist_len_file)
    context = np.load(context_file).reshape(-1)
    class_list = np.load(class_file)
    emotion_version = pd.read_csv(emotion_version_file, sep='\t', encoding='UTF-8', header=None)

    f = open('statics.txt', 'w', encoding='UTF-8')

    # length of conversation
    max_len = np.max(hist_len)
    min_len = np.min(hist_len)
    mean_len = np.mean(hist_len)
    f.write('Length of conversation:\n')
    f.write('max_len\tmin_len\tmean_len\n')
    f.write('{}\t{}\t{}\n\n\n'.format(max_len, min_len, mean_len))

    # length of sentecne
    sentences = emotion_version[[1]].values.reshape(-1)
    min_len = 1000
    max_len = 0
    total_len = 0
    for sen in sentences:
        length = len(sen.split())
        if max_len < length:
            max_len = length
        if min_len > length:
            min_len = length
        total_len += length
    f.write('Length of sentence:\n')
    f.write('max_len\tmin_len\tmean_len\n')
    f.write('{}\t{}\t{}\n\n\n'.format(max_len, min_len, total_len/len(sentences)))

    # distribution of emotions
    f.write('Distribution of emotions ({} emotions):\n\n'.format(len(class_list)))
    f.write('Conversation:\n')
    class_num = Counter(context)
    for emo in class_list:
        f.write('{}: {}\n'.format(emo, class_num[emo]))
    f.write('\nSentence:\n')
    class_num = Counter(emotion_version[0].values.reshape(-1))
    for emo in class_list:
        f.write('{}: {}\n'.format(emo, class_num[emo]))
    f.close()

def emotion_prompt(prompt_file, context_file):
    prompt = np.load(prompt_file)
    context = np.load(context_file)
    prompt = pd.DataFrame(prompt)
    context = pd.DataFrame(context)

    output = pd.concat([context, prompt], axis=1)
    output.to_csv('valid_prompt_emotion_version.csv', sep='\t', index=False, encoding='UTF-8')

def sentic_analysis(input_file, sentic_class_file):
    data = pd.read_csv(input_file, sep='\t', header=None, index_col=False, encoding='UTF-8')
    sentic_class = np.load(sentic_class_file, allow_pickle=True).item()
    sentic_words = sentic_class['vocab']
    sentences = data[1].tolist()
    length = 0
    sentic = 0
    sentic_sen = set()

    for idx, row in enumerate(sentences):
        words = row.split()
        length += len(words)
        for word in words:
            if word in sentic_words:
                sentic += 1
                sentic_sen.update({idx})

    print('There are {} words out of {} in senticnet, {} sentences out of {} include sentic words'.format(
        sentic, length, len(sentic_sen), len(sentences)
    ))

def sentic_seg(senticnet_info_path, input_file, output_file):
    sentic_info = np.load(senticnet_info_path, allow_pickle=True).item()
    vocab = sentic_info['vocab']
    vocab_ = sentic_info['vocab_']
    replce_dict = {re.escape(word): vocab_[idx] for idx, word in enumerate(vocab)}
    pattern = re.compile('|'.join(replce_dict.keys()))
    original = pd.read_csv(input_file, sep='\t', encoding='UTF-8', header=0, index_col=False)
    for idx, row in original.iterrows():
        original.loc[idx, 'post'] = pattern.sub(lambda m: replce_dict[re.escape(m.group(0))], row['post'])
    original.to_csv(output_file, sep='\t', encoding='UTF-8', index=False)

def ECE_read_file(input_file_path):
    data = np.load(input_file_path, allow_pickle=True).item()

    conv_ids = data['conv_ids']
    emotion32s = data['emotions']
    lengths = data['lengths']
    sentences = data['sentences']
    VPs = data['VPs']
    NPs = data['NPs']
    dependencies = data['dependencies']
    return conv_ids, emotion32s, lengths, sentences, VPs, NPs, dependencies

def clause_keywords(score_file_path, parser_file_path, output_file_path, emotion_clause_file_path):
    punc = r'( \. | , | ; | ! | -- | \? )'
    clause_scores = pd.read_csv(score_file_path, header=0, sep='\t')
    # conv_ids	emotion32	emotion8	VPs	VP_score	NPs	NP_score
    conv_ids, _, _, sentences, VPs, _, _ = ECE_read_file(parser_file_path)
    keywords_version = np.empty([0, 6], dtype=object) # conv_ids, no, label, emotion32, emotion8, clause
    emotion_clasue = []
    for index, row in clause_scores.iterrows():
        conv_ids, emotion32, emotion8, VPs, VP_score, NPs, NP_score = row
        VPs = ast.literal_eval(VPs)
        if VPs == [[]]:
            VPs = ast.literal_eval(NPs)
            VP_score = NP_score

        lengths = [len(VP) for VP in VPs]
        VP_score = ast.literal_eval(VP_score)
        VP_score = [List[emotion8] for List in VP_score]
        emo_idx = [idx for idx, score in enumerate(VP_score) if score > 0.5]
        if emo_idx == []:
            try:
                emo_score = max(VP_score)
                emo_idx = [VP_score.index(emo_score)]
            except:
                emo_idx = []
        emoID = []
        emoClauses = []
        for idx in emo_idx:
            for i, length in enumerate(lengths):
                if idx - length < 0:
                    emoID.append(i)
                    break
                else:
                    idx -= length
            emoClause = VPs[i][idx]
            emoClauses.append(emoClause)
        emotion_clasue.append([conv_ids, emoClauses])
        sentence = sentences[index][0]
        senNo = 0
        flag = False ########
        for senID, sen in enumerate(sentence):
            sen = re.split(punc, sen)
            values = sen[::2]
            delimiters = sen[1::2] + ['']
            sen = [value + delimiters[i] for i, value in enumerate(values)]
            for s in sen:
                emo = False
                for cid, clause in enumerate(emoClauses):
                    if clause in s and senID == emoID[cid]:
                        emo = True
                        flag = True ########
                tmp = np.array([[conv_ids, senNo, emotion32, emotion8, emo, s]], dtype=object)
                senNo += 1
                keywords_version = np.concatenate((keywords_version, tmp), axis=0)
        if not flag:
            print(emoClauses) ########
            print(sentence)
            input()
    data = pd.DataFrame(keywords_version)
    data.to_csv(output_file_path, sep='\t', header=['conv_ids', 'no', 'label', 'emotion32', 'emotion8', 'clause'], index=False)
    data = pd.DataFrame(emotion_clasue)
    data.to_csv(emotion_clause_file_path, sep='\t', header=['conv_ids', 'clause'], index=False)

def clause_label(prompt_file_path, score_file_path, output_file_path):
    prompt = pd.read_csv(prompt_file_path, header=0, sep='\t')
    conv = pd.read_csv(score_file_path, header=0, sep='\t')
    current_id = -1
    keywords = list()
    # conv_id, clause_no, emotion, context, emotional, chatbot, clause_content

    for idx, row in conv.iterrows():
        emo32, emo8, id, posts, scores = row
        posts = ast.literal_eval(posts)
        scores = ast.literal_eval(scores)
        if id != current_id:
            num = 0
            chatbot = False
            current_id = id
            _, _, contexts, context_scores = prompt.loc[id]
            contexts = ast.literal_eval(contexts)
            context_scores = ast.literal_eval(context_scores)
            max_emo = 0
            for i, cxt in enumerate(contexts):
                max_emo = context_scores[i][emo8] if context_scores[i][emo8] else max_emo
                emo_cls = True if context_scores[i][emo8] > 0.8 else False
                tmp = [id, num, emo8, True, emo_cls, False, cxt]
                num += 1
                keywords.append(tmp)
        for i, post in enumerate(posts):
            emo_cls = True if scores[i][emo8] >= 0.8*max_emo else False
            tmp = [id, num, emo8, False, emo_cls, chatbot, post]
            num += 1
            keywords.append(tmp)
        chatbot = not chatbot

    keywords = pd.DataFrame(keywords)
    keywords.to_csv(output_file_path, sep='\t', header=['conv_id', 'clause_no', 'label', 'context?', 'emotion?', 'chatbot?', 'clause_content'], index=False)

# emotion_class('valid_multi_turn_version.npy', 'valid_context.npy', 'valid_emotion_version.csv')
# statics('valid_hist_len.npy', 'valid_context.npy', 'valid_prompt.npy', 'class_list.npy', 'valid_emotion_version.csv')
# emotion_prompt('valid_prompt.npy', 'valid_context.npy')
# sentic_analysis('prompt_emotion_version.csv', 'C:/Users/csjwang/Documents/.csjwang/sentic_info/sentic_info.npy')

# clause_keywords('clause_scores/emotion_clause_scores_valid.csv', 'npy/prompt_parser_version.npy', 'clause_scores/keywords/prompt_clause_keywords.csv', 'clause_scores/clauses/prompt_emotion_clause.csv')

clause_label('prompt_scores.csv', 'multi_turn_scores.csv', 'clause_keywords.csv')
clause_label('valid_prompt_scores.csv', 'valid_multi_turn_scores.csv', 'valid_clause_keywords.csv')