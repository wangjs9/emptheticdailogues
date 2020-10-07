import torch, time
from tqdm import tqdm
from datetime import timedelta
from transformers import BertTokenizer
PAD, UNK, CLS = '<PAD>', '<UNK>', '<CLS>'

def build_dataset(config, do_train, do_test):
    class_list = config.emotion32
    class_dic = {cls: idx for idx, cls in enumerate(config.emotion8)}
    def load_dataset(path):
        contents = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        with open(path, 'r', encoding='UTF-8') as f:
            col_name = True
            for line in tqdm(f):
                lin = line.strip()
                if not lin or col_name:
                    col_name = False
                    continue
                label, content = lin.split('\t')
                label = class_dic[class_list[label]]
                encoded_dict = tokenizer.encode_plus(content,
                                 add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                 max_length=config.pad_size,  # Pad & truncate all sentences.
                                 padding='max_length',
                                 return_attention_mask=True,  # Construct attn. masks.
                                 return_tensors='np',  # Return pytorch tensors.
                                 truncation=True)
                input_ids = encoded_dict['input_ids'][0]
                seq_len = sum(encoded_dict['attention_mask'][0])
                contents.append((input_ids, int(label), seq_len, encoded_dict['attention_mask'][0]))

        return contents  # [([...], 0), ([...], 1), ...]

    train = None
    valid = None
    test = None
    if do_train:
        train = load_dataset(config.train_path)
        valid = load_dataset(config.valid_path)
    if do_test:
        test = load_dataset(config.test_path)
    return train, valid, test

def build_dataset_from_np(config, np_arr, emotion):
    class_list = config.emotion32
    class_dic = {cls: idx for idx, cls in enumerate(config.emotion8)}

    contents = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    arr = list()
    for ele in np_arr:
        arr += ele

    for line in arr:
        content = line.strip()
        label = 0
        encoded_dict = tokenizer.encode_plus(content,
                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                         max_length=config.pad_size,  # Pad & truncate all sentences.
                         padding='max_length',
                         return_attention_mask=True,  # Construct attn. masks.
                         return_tensors='np',  # Return pytorch tensors.
                         truncation=True)
        input_ids = encoded_dict['input_ids'][0]
        seq_len = sum(encoded_dict['attention_mask'][0])
        contents.append((input_ids, int(label), seq_len, encoded_dict['attention_mask'][0]))
    return contents, class_dic[class_list[emotion]]  # [([...], 0), ([...], 1), ...]

def build_dataset_from_list(config, _list, emotions):
    class_list = config.emotion32
    class_dic = {cls: idx for idx, cls in enumerate(config.emotion8)}

    contents = []
    emotions = [class_dic[class_list[emotion]] for emotion in emotions]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for line in _list:
        label = 0
        encoded_dict = tokenizer.encode_plus(line,
                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                         max_length=config.pad_size,  # Pad & truncate all sentences.
                         padding='max_length',
                         return_attention_mask=True,  # Construct attn. masks.
                         return_tensors='np',  # Return pytorch tensors.
                         truncation=True)
        input_ids = encoded_dict['input_ids'][0]
        seq_len = sum(encoded_dict['attention_mask'][0])
        contents.append((input_ids, label, seq_len, encoded_dict['attention_mask'][0]))
    return contents, emotions  # [([...], 0), ([...], 1), ...]

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if self.n_batches == 0 or len(batches) % self.batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):

        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    """get time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
