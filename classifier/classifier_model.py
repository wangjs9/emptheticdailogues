import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Config(object):
    def __init__(self, dataset_path, save_path):
        self.train_path = dataset_path + 'prompt_emotion_version.csv'
        self.valid_path = dataset_path + 'valid_prompt_emotion_version.csv'
        # self.valid_path = 'C:/Users/csjwang/Documents/.csjwang/test.txt'
        self.save_path = save_path + 'classifier/model.path'
        self.load_model_path = save_path + 'classifier/model_prompt.path'
        self.vocab_path = save_path + 'vocab.txt'
        self.sentic_data_path = save_path + 'sentic_data.json'

        class_path = dataset_path + 'class_list.npy'
        self.class_list = np.load(class_path, allow_pickle=True)
        self.senticnet_info_path = './sentic_info/sentic_info.npy'
        # self.emotion8 = {'#surprise': 1, '#anger': 2, '#joy': 3, '#sadness': 4, '#fear': 5, '#disgust': 6, '#interest': 7, '#admiration': 8}
        self.emotion8 = ['#surprise', '#anger', '#joy', '#sadness', '#fear', '#disgust', '#interest', '#admiration']
        self.emotion32 = {
            'afraid': '#fear',
            'angry': '#anger',
            'annoyed': '#anger',
            'anticipating': '#interest',
            'anxious': '#fear',
            'apprehensive': '#fear',
            'ashamed': '#disgust',
            'caring': '#interest',
            'confident': '#admiration',
            'content': '#joy',
            'devastated': '#sadness',
            'disappointed': '#disgust',
            'disgusted': '#disgust',
            'embarrassed': '#disgust',
            'excited': '#joy',
            'faithful': '#admiration',
            'furious': '#joy',
            'grateful': '#admiration',
            'guilty': '#sadness',
            'hopeful': '#interest',
            'impressed': '#admiration',
            'jealous': '#anger',
            'joyful': '#joy',
            'lonely': '#sadness',
            'nostalgic': '#sadness',
            'prepared': '#interest',
            'proud': '#admiration',
            'sad': '#sadness',
            'sentimental': '#sadness',
            'surprised': '#surprise',
            'terrified': '#fear',
            'trusting': '#interest',
        }

        # hyperparameters:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_num = len(self.emotion8)
        self.epoch_num = 3
        self.batch_size = 32
        self.pad_size = 50
        self.learning_rate = 3e-5

        # LSTM parameters:
        self.require_improvement = 1000
        self.bert_hidden_size = 768
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.5

class Model(nn.Module):
    def __init__(self, config, class_num=None):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                    num_labels=config.class_num, output_attentions=False,
                    output_hidden_states=False)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.rnn = nn.GRU(config.bert_hidden_size,
                          config.hidden_size, num_layers=config.num_layers, batch_first=True,
                          bidirectional=True, dropout=config.dropout)
        if class_num == None:
            class_num = config.class_num
        self.fc = nn.Linear(config.hidden_size*2, class_num)

    def forward(self, x, y, pred=False):
        """
        x: [post, mask]
        """
        post = x[0]
        mask = x[2]
        hid, pooled = self.bert(post, token_type_ids=None, attention_mask=mask)
        output, _ = self.rnn(hid)
        logits = torch.tanh(self.fc(output[:, -1, :]))
        if pred:
            return logits
        loss = F.cross_entropy(logits, y)
        return loss, logits
