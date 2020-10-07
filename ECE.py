from senticnet.senticnet import SenticNet
SN = SenticNet('en')
from datetime import timedelta
import time, re, torch
import numpy as np
import pandas as pd
import classifier_model
from classifier import predict

class Config(object):
    def __init__(self):
        self.parser_info_path = './$empatheticdialogues/npy/parser_version.npy'
        self.senticnet_info_path = './sentic_info/sentic_info.npy'

        self.sentic_score_path = './sentic_info/sentic_score.npy'
        self.sentic_score_path_csv = './sentic_info/sentic_score.csv'

    @property
    def emotion8(self):
        return {'#anger': 0, '#admiration': 1, '#interest': 2, '#surprise': 3, '#fear': 4, '#sadness': 5, '#joy': 6, '#disgust': 7}

    @property
    def emotion32(self):
        return {
            'afraid': 'fear',
            'angry': 'anger',
            'annoyed': 'disgust',
            'anticipating': 'interest',
            'anxious': 'fear',
            'apprehensive': 'fear',
            'ashamed': 'disgust',
            'caring': 'joy',
            'confident': 'admiration',
            'content': 'joy',
            'devastated': 'sadness',
            'disappointed': 'disgust',
            'disgusted': 'disgust',
            'embarrassed': 'disgust',
            'excited': 'joy',
            'faithful': 'admiration',
            'furious': 'joy',
            'grateful': 'admiration',
            'guilty': 'sadness',
            'hopeful': 'interest',
            'impressed': 'admiration',
            'jealous': 'anger',
            'joyful': 'joy',
            'lonely': 'sadness',
            'nostalgic': 'sadness',
            'prepared': 'interest',
            'proud': 'admiration',
            'sad': 'sadness',
            'sentimental': 'sadness',
            'surprised': 'superise',
            'terrified': 'fear',
            'trusting': 'interest',
        }

def get_dif_time(start_time):
    """get time difference"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def sentic_helper(output_file_name, language='en'):
    import importlib
    sentic_info = dict()
    data_module = importlib.import_module("senticnet.babel.data_" + language)
    data = data_module.senticnet
    vocab = list(data.keys())
    sentic_info['vocab_'] = vocab
    sentic_info['class'] = {data[word][4] for word in vocab}
    vocab = [word.replace('_', ' ') for word in vocab]
    sentic_info['vocab'] = vocab # [word for word in vocab if len(word.split()) > 1]
    # print(sentic_info['class'])
    np.save(output_file_name, sentic_info)

def read_file(input_file_path):
    data = np.load(input_file_path, allow_pickle=True).item()

    conv_ids = data['conv_ids']
    emotion32s = data['emotions']
    lengths = data['lengths']
    sentences = data['sentences']
    VPs = data['VPs']
    NPs = data['NPs']
    dependencies = data['dependencies']
    return conv_ids, emotion32s, lengths, sentences, VPs, NPs, dependencies

def sentic_score_analysis(input_file_path, senticnet_info_path, sentic_score_path, sentic_score_csv=None):
    emotions, _, sentences, VPs, NPs, _ = read_file(input_file_path)
    emotion_scores = np.empty((len(emotions), 5), dtype=object)
    sentic_info = np.load(senticnet_info_path, allow_pickle=True).item()
    vocab = sentic_info['vocab']
    vocab_ = sentic_info['vocab_']
    emotion_class = {cls: idx for idx, cls in enumerate(sentic_info['class'])}
    inverse_emotion_class = {idx: cls for idx, cls in enumerate(sentic_info['class'])}
    replce_dict = {re.escape(word): vocab_[idx] for idx, word in enumerate(vocab)}
    pattern = re.compile('|'.join(replce_dict.keys()))
    start_time = time.time()
    for idx, sentence in enumerate(sentences):

        if idx % 1000 == 1:
            print('Progress: {} / {} instances, time cost: {}'.format(idx, len(emotions), get_dif_time(start_time)))
        emotion32 = emotions[idx]

        # calculate the sentic scores of sentences
        sentic_scores = list()
        for clause in sentence[0]:
            clause = pattern.sub(lambda m: replce_dict[re.escape(m.group(0))], clause)
            clause = clause.split()
            sentic_score = [0.0 for i in range(8)]
            for word in clause:
                try:
                    mood_1, mood_2 = SN.moodtags(word)
                    score = float(SN.polarity_intense(word))
                    sentic_score[emotion_class[mood_1]] += score
                    sentic_score[emotion_class[mood_2]] += score
                except:
                    continue
            sentic_scores.append(sentic_score)

        # get the most posible emotion from 8 classes
        total_sentic_score = [abs(sum([scores[i] for scores in sentic_scores])) for i in range(8)]
        max_index = total_sentic_score.index(max(total_sentic_score))
        emotion8_1 = inverse_emotion_class[max_index]
        total_sentic_score[max_index] = -1
        emotion8_2 =  inverse_emotion_class[total_sentic_score.index(max(total_sentic_score))]
        emotion8 = [emotion8_1, emotion8_2]
        # calculate the sentic scores of verb phrases
        sentic_scores_vps = list()
        for phrase in VPs[idx][0]:
            sentic_scores_vp = list()
            for clause in phrase:
                clause = pattern.sub(lambda m: replce_dict[re.escape(m.group(0))], clause)
                clause = clause.split()
                sentic_score = [0.0 for i in range(8)]
                for word in clause:
                    try:
                        mood_1, mood_2 = SN.moodtags(word)
                        score = float(SN.polarity_intense(word))
                        sentic_score[emotion_class[mood_1]] += score
                        sentic_score[emotion_class[mood_2]] += score
                    except:
                        continue
                sentic_scores_vp.append(sentic_score)
            sentic_scores_vps.append(sentic_scores_vp)
        # calcualte the sentic scores of noun phrases
        sentic_scores_nps = list()
        for phrase in NPs[idx][0]:
            sentic_scores_np = list()
            for clause in phrase:
                clause = pattern.sub(lambda m: replce_dict[re.escape(m.group(0))], clause)
                clause = clause.split()
                sentic_score = [0.0 for i in range(8)]
                for word in clause:
                    try:
                        mood_1, mood_2 = SN.moodtags(word)
                        score = float(SN.polarity_intense(word))
                        sentic_score[emotion_class[mood_1]] += score
                        sentic_score[emotion_class[mood_2]] += score
                    except:
                        continue
                sentic_scores_np.append(sentic_score)
            sentic_scores_nps.append(sentic_scores_np)

        # add [emotion, sentence_score, vp_scores, np_scores] into the numpy array
        emotion_scores[idx, :] = emotion32, emotion8, sentic_scores, sentic_scores_vps, sentic_scores_nps

    np.save(sentic_score_path, emotion_scores)
    if sentic_score_csv != None:
        emotion_scores = pd.DataFrame(emotion_scores)
        emotion_scores.to_csv(sentic_score_csv, sep='\t', header=['emotion32', 'emotion8', 'sentence_score', 'VP_score', 'NP_score'], index=False)

def emotion_claue(input_file_path, output_path):
    from utils import build_iterator, build_dataset_from_np
    conv_ids, emotions, _, _, VPs, NPs, _ = read_file(input_file_path)
    dataset = './$empatheticdialogues/'
    config = classifier_model.Config(dataset, dataset)
    model = classifier_model.Model(config).to(config.device)
    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    conv_ids = conv_ids.reshape(-1,).tolist()
    VP_score = []
    NP_score = []
    emotions8 = []

    print('Generate emotion scores for VPs:')
    for idx, VP in enumerate(VPs):
        if idx % 1000 == 0:
            print(' This is the {}/{} instances'.format(idx, len(emotions)))
        data, emotion8 = build_dataset_from_np(config, VP[0], emotions[idx][0])

        if len(data) == 0:
            VP_score.append(data)
            emotions8.append(emotion8)
            continue

        data_iter = build_iterator(data, config)
        logits = predict(model, data_iter).cpu().numpy().tolist()
        VP_score.append(logits)
        emotions8.append(emotion8)

    print('Generate emotion scores for NPs:')
    for idx, NP in enumerate(NPs):
        if idx % 1000 == 0:
            print(' This is the {}/{} instances'.format(idx, len(emotions)))
        data, emotion8 = build_dataset_from_np(config, NP[0], emotions[idx][0])
        if len(data) == 0:
            NP_score.append(data)
            continue

        data_iter = build_iterator(data, config)
        logits = predict(model, data_iter).cpu().numpy().tolist()
        NP_score.append(logits)

    emotions = emotions.reshape(-1,).tolist()
    VPs = VPs.reshape(-1,).tolist()
    NPs = NPs.reshape(-1,).tolist()

    emotion_clause = []
    for i in range(len(emotions)):
        tmp = [conv_ids[i], emotions[i], emotions8[i], VPs[i], VP_score[i], NPs[i], NP_score[i]]
        emotion_clause.append(tmp)

    emotion_clause = pd.DataFrame(emotion_clause)
    emotion_clause.to_csv(output_path, sep='\t', header=['conv_ids','emotion32', 'emotion8', 'VPs', 'VP_score', 'NPs', 'NP_score'], index=False)

def emotion_classification(input_file_path, output_path):
    from utils import build_iterator, build_dataset_from_list
    data = pd.read_csv(input_file_path, sep='\t', header=0)
    punc = r'(\S\. |\S\? |\S! |\S, )'
    shapes = list()
    cls_data = list()
    emotions = list()
    indexes = list()
    for index, row in data.iterrows():
        emo, id, post = row
        emotions.append(emo)
        indexes.append(id)
        clauses = re.split(punc, post)
        values = clauses[::2]
        delimiters = clauses[1::2] + ['']
        clauses = [value + delimiters[i] for i, value in enumerate(values)]
        clauses = [clause for clause in clauses if clause not in ['\s+', '']]
        cls_data.extend(clauses)
        shapes.append(len(clauses))

    dataset = './$empatheticdialogues/'
    config = classifier_model.Config(dataset, dataset)
    model = classifier_model.Model(config).to(config.device)
    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    data_iter, emotion8s = build_dataset_from_list(config, cls_data, emotions)
    data_iter = build_iterator(data_iter, config)
    logits = predict(model, data_iter).cpu().numpy().tolist()

    res = list()
    start = 0
    for idx, index in enumerate(indexes):
        length = shapes[idx]
        end = start + length
        tmp = [emotions[idx], emotion8s[idx], index, cls_data[start:end], logits[start:end]]
        res.append(tmp)
        start = end
    # emo, id, post list, score list
    res = pd.DataFrame(res)
    res.to_csv(output_path, sep='\t', header=['emotion32', 'emotion8', 'conversation_id', 'posts', 'scores'], index=False)

def prompt_emotion_classification(input_file_path, output_path):
    from utils import build_iterator, build_dataset_from_list
    data = pd.read_csv(input_file_path, sep='\t', header=0)
    punc = r'(\S\. |\S\? |\S! |\S, )'
    shapes = list()
    cls_data = list()
    emotions = list()
    for index, row in data.iterrows():
        emo, post = row
        emotions.append(emo)
        clauses = re.split(punc, post)
        values = clauses[::2]
        delimiters = clauses[1::2] + ['']
        clauses = [value + delimiters[i] for i, value in enumerate(values)]
        clauses = [clause for clause in clauses if clause not in ['\s+', '']]
        cls_data.extend(clauses)
        shapes.append(len(clauses))

    dataset = './$empatheticdialogues/'
    config = classifier_model.Config(dataset, dataset)
    model = classifier_model.Model(config).to(config.device)
    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    data_iter, emotion8s = build_dataset_from_list(config, cls_data, emotions)
    data_iter = build_iterator(data_iter, config)
    logits = predict(model, data_iter).cpu().numpy().tolist()

    res = list()
    start = 0
    for idx, emo in enumerate(emotions):
        length = shapes[idx]
        end = start + length

        tmp = [emo, emotion8s[idx], cls_data[start:end], logits[start:end]]
        res.append(tmp)
        start = end
    # emo, id, post list, score list
    res = pd.DataFrame(res)
    res.to_csv(output_path, sep='\t', header=['emotion32', 'emotion8', 'posts', 'scores'], index=False)


# config = Config()
# sentic_score_analysis(config.parser_info_path, config.senticnet_info_path, config.sentic_score_path, config.sentic_score_path_csv)

# emotion_claue('./$empatheticdialogues/npy/prompt_parser_version.npy', './$empatheticdialogues/prompt_emotion_clause_scores.csv')
# emotion_claue('./$empatheticdialogues/npy/valid_prompt_parser_version.npy', './$empatheticdialogues/valid_prompt_emotion_clause_scores.csv')
# emotion_claue('./$empatheticdialogues/npy/parser_version.npy', './$empatheticdialogues/emotion_clause_scores.csv')
# emotion_claue('./$empatheticdialogues/npy/valid_parser_version.npy', './$empatheticdialogues/valid_emotion_clause_scores.csv')
emotion_classification('./$empatheticdialogues/valid_multi_turn.csv', 'valid_multi_turn_scores.csv')
emotion_classification('./$empatheticdialogues/multi_turn.csv', 'multi_turn_scores.csv')
prompt_emotion_classification('./$empatheticdialogues/prompt.csv', 'prompt_scores.csv')
prompt_emotion_classification('./$empatheticdialogues/valid_prompt.csv', 'valid_prompt_scores.csv')




