from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from tqdm import tqdm
import os
import numpy as np
import random
import time
from utils import print_rank0
import math
def load_pickle(file_path):

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data
    
class DataSequential(Dataset):
    def __init__(self, args, tokenizer, tokenized_text, tokenized_domain, token_id_dict, mode='train'):
        super().__init__()
        print_rank0(f"Loading {mode} data...", args.rank)
        self.args = args
        self.tokenizer = tokenizer
        self.tokenized_text = tokenized_text
        self.tokenized_domain = tokenized_domain
        self.token_id_dict = token_id_dict
        self.mode = mode
        self.length = 0
        self.data = None
        self.max_seq_length = 10
        self.max_token_length = args.max_token_length
        self.item_title_list = None
        self.item_count = max(list(pickle.load(open(f"{args.data_path}/{args.dataset}/iid2asin.pkl", 'rb')).keys())) + 1
        self.single_domain_iid = pickle.load(open(f'{args.data_path}/{args.dataset}/single_domain_iid.pkl', 'rb'))
        self.domain_iid = {key: idx for idx, key in enumerate(self.single_domain_iid.keys())}
        self.invert_iid = self.invert_dict(self.single_domain_iid)
        train_path = f'{args.data_path}/{args.dataset}/single_domain_iid_train.pkl'


        self.load_data()
        self.load_seq()
        self.item_title_tokens = None

        self.tokenize_item_titles()
        self.load_negative()
        self.sample_valid(self.data)


        if os.path.exists(train_path):
            self.single_domain_iid_train = pickle.load(open(train_path, 'rb'))
        else:
            fallback_path = f'{args.data_path}/{args.dataset}/single_domain_iid.pkl'
            print(f"'{train_path}' not found. Loading from '{fallback_path}' instead.")
            self.single_domain_iid_train = pickle.load(open(fallback_path, 'rb'))

        valid_path = f'{args.data_path}/{args.dataset}/single_domain_iid_valid.pkl'
        if os.path.exists(valid_path):
            self.single_domain_iid_valid = pickle.load(open(valid_path, 'rb'))
        else:
            fallback_path = f'{args.data_path}/{args.dataset}/single_domain_iid.pkl'
            print(f"'{valid_path}' not found. Loading from '{fallback_path}' instead.")
            self.single_domain_iid_valid = pickle.load(open(fallback_path, 'rb'))

        self.candi_item_attention_mask = None
        self.candi_item_input_ids = None
        self.generate_cate_items()
        print_rank0(f"Load {mode} data successfully", args.rank)

    def invert_dict(self, input_dict):
        output_dict = {}
        for key, values in input_dict.items():
            for value in values:
                output_dict[value] = key
        with open(f"{self.args.data_path}/{self.args.dataset}/inverted_iid.pkl", "wb") as f:
            pickle.dump(output_dict, f)
        return output_dict



    def training_item(self, train):
        training_item = [sample[0] + [sample[1]] for sample in train]

        iid = self.single_domain_iid

        training_item = {item for sublist in training_item for item in sublist}

        filtered_dict = {
            key: [value for value in values if value in training_item]
            for key, values in iid.items()
        }

        output_file = f"{self.args.data_path}/{self.args.dataset}/single_domain_iid_train.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(filtered_dict, f)
        print("Training Itemset Processed!")

    def val_item(self, train, valid):

        training_item = [sample[0] + [sample[1]] for sample in train]
        valid_item = [sample[0] + [sample[1]] for sample in valid]
        training_item = training_item + valid_item

        iid = self.single_domain_iid

        training_item = {item for sublist in training_item for item in sublist}

        filtered_dict = {
            key: [value for value in values if value in training_item]
            for key, values in iid.items()
        }

        output_file = f"{self.args.data_path}/{self.args.dataset}/single_domain_iid_valid.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(filtered_dict, f)
        print("Validation Itemset Processed!")

    def sample_valid(self, datas):
        if self.args.valid_ratio == 1 or self.mode != 'valid':
            return
        import random
        random.seed(42)
        sample_idx = random.sample(list(range(len(datas))), int(len(datas) * self.args.valid_ratio))
        sample_idx.sort()
        new_datas = []
        for idx in sample_idx:
            new_datas.append(datas[idx])
        self.nega_items = []
        self.length = len(new_datas)
        self.data = new_datas

    def load_negative(self):
        if self.mode == 'train':
            print_rank0("Don't load negatives!", self.args.rank)
            return
        self.nega_items = pickle.load(open(f'{self.args.data_path}/{self.args.dataset}/negatives_{self.mode}-{self.args.nega_count}.pkl', 'rb'))
        print_rank0("Load negatives successfully", self.args.rank)


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        example_input = self.generate_example_input(self.data[item], item)
        example_input.append(item)
        return example_input

    def load_seq(self):
        import json
        if os.path.exists(f'local_dataset/{self.args.dataset}/seq.json'):
            with open(f'local_dataset/{self.args.dataset}/seq.json', 'r') as f:
                seq = json.load(f)
            self.seq = seq
            return
        review_datas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/review_datas.pkl", 'rb'))
        seq = {}
        for user in tqdm(review_datas.keys(), desc='Splitting Train/Valid/Test'):
            seq_iid_list = [i[0] for i in review_datas[user]]
            seq[user] = seq_iid_list
        

        if self.args.rank == 0:
            os.makedirs(f'local_dataset/{self.args.dataset}/', exist_ok=True)
            with open(f'local_dataset/{self.args.dataset}/seq.json', 'w') as f:
                json.dump(seq, f, indent=4)
        
        else:
            time.sleep(20)

        self.seq = seq

    def load_data(self):
        if os.path.exists(f'local_dataset/{self.args.dataset}/train_data.pkl'):
            if self.mode == 'train':
                self.data = pickle.load(open(f'local_dataset/{self.args.dataset}/train_data.pkl', 'rb'))
            elif self.mode == 'valid':
                self.data = pickle.load(open(f'local_dataset/{self.args.dataset}/valid_data.pkl', 'rb'))
            elif self.mode == 'test':
                self.data = pickle.load(open(f'local_dataset/{self.args.dataset}/test_data.pkl', 'rb'))
            self.length = len(self.data)
            return

        review_datas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/review_datas.pkl", 'rb'))
        train_data = []
        valid_data = []
        test_data = []

        for user in tqdm(review_datas.keys(), desc='Splitting Train/Valid/Test'):
            seq_iid_list = [review_datas[user][0][0]]
            seq_iid_cate_list = [review_datas[user][0][2]]

            # To build full-seq data
            # temp_train_data = []
            # temp_valid_data = []
            # temp_test_data = []
            for i in range(1, len(review_datas[user])):
                target_iid = review_datas[user][i][0]
                target_iid_cate = review_datas[user][i][2]
                target_time = review_datas[user][i][3]
                if target_time < 1628643414042:
                    train_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                elif target_time >= 1658002729837:
                    test_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                else:
                    valid_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])

                seq_iid_list = seq_iid_list + [review_datas[user][i][0]]
                seq_iid_cate_list = seq_iid_cate_list + [review_datas[user][i][2]]

                seq_iid_list = seq_iid_list[-self.max_seq_length:]
                seq_iid_cate_list = seq_iid_cate_list[-self.max_seq_length:]
            # if temp_train_data != []:
            #     train_data.append(temp_train_data[-1])
            # if temp_valid_data != []:
            #     valid_data.append(temp_valid_data[-1])
            # if temp_test_data != []:
            #     test_data.append(temp_test_data[-1])
        if self.args.rank == 0:
            os.makedirs(f'local_dataset/{self.args.dataset}/', exist_ok=True)
            pickle.dump(train_data, open(f'local_dataset/{self.args.dataset}/train_data.pkl', 'wb'))
            pickle.dump(valid_data, open(f'local_dataset/{self.args.dataset}/valid_data.pkl', 'wb'))
            pickle.dump(test_data, open(f'local_dataset/{self.args.dataset}/test_data.pkl', 'wb'))
        else:
            time.sleep(20)

        self.training_item(train_data)
        self.val_item(train_data, valid_data)

        if self.mode == 'train':
            self.data = train_data
        elif self.mode == 'valid':
            self.data = valid_data
        elif self.mode == 'test':
            self.data = test_data
        else:
            raise NotImplementedError
        self.length = len(self.data)

    def generate_cate_items(self):
        candi_item_input_ids = []
        candi_item_attention_mask = []
        fp_tokens = 42 + len(self.tokenized_text)
        # fp_tokens = 21
        for idx in range(self.item_count):
            if 'Qwen' in self.args.backbone:
                candi_tokens = self.tokenized_text + self.item_title_tokens[idx] + [151665] + [self.tokenizer.eos_token_id]
            pad_len = fp_tokens - len(candi_tokens)
            if pad_len >= 0:
                candi_item_input_ids.append(candi_tokens + [self.tokenizer.eos_token_id] * pad_len)
                candi_item_attention_mask.append((len(candi_tokens) * [1] + [0] * pad_len))
            else:
                candi_item_input_ids.append(candi_tokens[:fp_tokens-2]+ [151665]+ [self.tokenizer.eos_token_id])
                candi_item_attention_mask.append( fp_tokens * [1])
        self.candi_item_input_ids = candi_item_input_ids
        self.candi_item_attention_mask = candi_item_attention_mask

    def tokenize_item_titles(self):
        if os.path.exists(f'local_dataset/{self.args.dataset}/tokenized_{self.args.backbone}.pkl'):
            tokenized = pickle.load(open(f'local_dataset/{self.args.dataset}/tokenized_{self.args.backbone}.pkl', 'rb'))
            self.item_title_tokens = tokenized['item_title_tokens']
            return

        item_metas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/meta_datas.pkl", 'rb'))
        iid2asin = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/iid2asin.pkl", 'rb'))
        item_title_list = ['None'] * self.item_count
        for iid, asin in iid2asin.items():
            item_title = item_metas[asin]['title'] if (
                    'title' in item_metas[asin].keys() and item_metas[asin]['title']) else 'None'
            item_title = item_title + '; '
            item_title_list[iid] = item_title

        def calculate_token_lengths(str_list):
            results = []
            length = [len(string.split()) for string in str_list]
            max_length = max(length)
            min_length = min(length)
            avg_length = sum(length) / len(length)
            condition_length = sum(1 for l in length if l < 21) / len(length)
            return
    
        # To save textual information of items.
        with open(f"./local_dataset/{self.args.dataset}/item_title_list.pkl", "wb") as file:
            pickle.dump(item_title_list, file)
            
        calculate_token_lengths(item_title_list)
        item_max_tokens = 42
        item_title_tokens = []
        for start in tqdm(range(0, len(item_title_list), 32), desc='Tokenizing'):
            tokenized_text = self.tokenizer(item_title_list[start: start + 32],
                                            truncation=True,
                                            max_length=item_max_tokens,
                                            padding=False,
                                            add_special_tokens=False,
                                            return_tensors=None,
                                            )

            item_title_tokens.extend(tokenized_text['input_ids'])

        self.item_title_tokens = item_title_tokens


        tokenized = {
            'item_title_tokens': self.item_title_tokens,
        }
        if self.args.rank == 0:
            pickle.dump(tokenized, open(f'local_dataset/{self.args.dataset}/tokenized_{self.args.backbone}.pkl', 'wb'))
        else:
            time.sleep(10)


    def mask_sequence(self,example, ratio ):
        seq_iid_list, target_iid,cate_list, target_cate = example[0], example[1], example[2], example[3]
        if len(list(set(cate_list))) > 1 and self.mode == 'train':
            mask = [
                0
                if cate == target_cate and random.random() < ratio 
                else 1 for cate in cate_list
            ]
            revised_seq_iid_list = []
            revised_cate_iid_list = []
            for idx, flag in enumerate(mask):
                if flag == 1:
                    revised_seq_iid_list.append(seq_iid_list[idx])
                    revised_cate_iid_list.append(cate_list[idx])

            revised_example = [revised_seq_iid_list, example[1], revised_cate_iid_list, target_cate]
            return revised_example
        return example



        
    def generate_example_input(self, example, example_idx):

        example = self.mask_sequence(example, self.args.mask_ratio)
        seq_iid_list, target_iid, target_domain = example[0], example[1], example[3]
        target_domain_id = self.domain_iid[target_domain]

        sequence_input_ids = []
        sequence_attention_mask = [] 

        if self.args.prompt_selection == 0:
            temp_tokenized_text = self.tokenized_text
        elif self.args.prompt_selection == 1:
            temp_tokenized_text = self.tokenized_text[:14] + self.tokenized_domain[example[3]] \
                + self.tokenized_text[17:]
        else:
            temp_tokenized_text = self.tokenized_text

        sequence_input_ids.extend(temp_tokenized_text)
        sequence_attention_mask.extend([1] * len(temp_tokenized_text))

        for seq_iid in seq_iid_list:
            sequence_attention_mask.extend([1] * len(self.item_title_tokens[seq_iid]))
            sequence_input_ids.extend(self.item_title_tokens[seq_iid])

        if 'Qwen' == self.args.backbone[: 4]:
            sequence_input_ids = sequence_input_ids + [151665]+ [self.tokenizer.eos_token_id]
            sequence_attention_mask.append(1)
            sequence_attention_mask.append(1)

        if self.mode == 'train':
            negative_items = random.sample(self.single_domain_iid_train[example[3]], self.args.train_nega_count)
            target_position = random.randint(0, self.args.train_nega_count)
            negative_items = negative_items[0:target_position] + [target_iid] + negative_items[target_position:]
        else:
            if self.mode == 'valid':
                negative_items = self.single_domain_iid_valid[example[3]]
            else:
                negative_items = self.single_domain_iid[example[3]]
            target_position = negative_items.index(target_iid)

        if self.mode == 'train':
            candi_item_input_ids = [self.candi_item_input_ids[x] for x in negative_items]
            candi_item_attention_mask = [self.candi_item_attention_mask[x] for x in negative_items]
        else:
            # In validation/test, item embs are pre-generated
            candi_item_input_ids = [0] * len(negative_items)
            candi_item_attention_mask = [0] * len(negative_items)


        return {
            'item_input_ids': candi_item_input_ids,
            'item_attention_mask': candi_item_attention_mask,
            'sequence_input_ids': sequence_input_ids,
            'sequence_attention_mask': sequence_attention_mask,
            'target_position': target_position,
            'target_iid': target_iid,
            'target_domain': target_domain_id,
            'negative_items': negative_items,
            'example_index': example_idx
        }

    def collate_fn(self, batch_data):
        item_input_ids = []
        item_attention_mask = []
        sequence_input_ids = []
        sequence_attention_mask = []
        target_position = []
        target_iid = []
        target_domain = []
        negative_items = []
        example_index = []

        max_seq_len = max(len(x['sequence_input_ids']) for x in batch_data)

        for example in batch_data:
            if self.mode == 'train':
                item_input_ids.extend(example['item_input_ids'])
                item_attention_mask.extend(example['item_attention_mask'])
            
            seq_pad_len = max_seq_len - len(example['sequence_input_ids'])
            sequence_input_ids.append(example['sequence_input_ids'] + [self.tokenizer.eos_token_id] * seq_pad_len)
            sequence_attention_mask.append(example['sequence_attention_mask'] + [0] * seq_pad_len)
            
            target_position.append(example['target_position'])
            target_iid.append(example['target_iid'])
            target_domain.append(example['target_domain'])
            negative_items.append(example['negative_items'])
            example_index.append(example['example_index'])

        if self.mode != 'train':
            # Padding negative items for batching in validation/test if they have different lengths
            max_neg_len = max(len(x) for x in negative_items)
            negative_items_padded = [list(np.pad(x, (0, max_neg_len - len(x)), 'constant', constant_values=-1)) for x in negative_items]
            negative_items = negative_items_padded

        return {
            'item_input_ids': torch.LongTensor(item_input_ids) if item_input_ids else None,
            'item_attention_mask': torch.LongTensor(item_attention_mask) if item_attention_mask else None,
            'sequence_input_ids': torch.LongTensor(sequence_input_ids),
            'sequence_attention_mask': torch.LongTensor(sequence_attention_mask),
            'target_position': torch.LongTensor(target_position),
            'target_iid': torch.LongTensor(target_iid),
            'target_domain': torch.LongTensor(target_domain),
            'negative_items': torch.LongTensor(negative_items),
            'example_index': torch.LongTensor(example_index)
        }

    def get_items_tokens(self):
        item_ids = []
        item_attn = []
        fp_tokens = 42+ len(self.tokenized_text)
        for iid in range(len(self.item_title_tokens)):
            if 'Qwen' == self.args.backbone[: 4]:
                item_tokens = self.tokenized_text +self.item_title_tokens[iid] + [151665] + [self.tokenizer.eos_token_id]
            pad_len = fp_tokens - len(item_tokens)
            if pad_len >= 0:
                item_ids.append(item_tokens + [self.tokenizer.eos_token_id] * pad_len)
                item_attn.append(len(item_tokens) * [1] + pad_len * [0])
            else:
                item_ids.append(item_tokens[:fp_tokens-2]+ [151665] + [self.tokenizer.eos_token_id]
                )
                item_attn.append(fp_tokens * [1])

        return {'item_ids': torch.LongTensor(item_ids),
                'item_attn': torch.LongTensor(item_attn)}

