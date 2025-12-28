import gzip
import json
import os
from tqdm import tqdm
import ssl
from collections import defaultdict
from param import parse_args
import torch
import html
import re
import datasets
import pandas as pd 
datasets.logging.set_verbosity_error()


amazon_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games',
}

def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def parse_csv(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield pd.read_csv(l)



def analysis_diff_user(review_datas):
    cnt = 0
    for user, inters in review_datas.items():
        domain = inters[0][2]
        for inter in inters[1: ]:
            if inter[2] != domain:
                cnt += 1
                break
    print(f"{cnt}/{len(review_datas)} has interaction in different domains!")

def analysis_diff_dataset(review_datas):
    domain_users = {}
    domain_other_users = {}
    domain_cnts = {}

    for user, inters in review_datas.items():
        user_domain = set()
        for inter in inters:
            user_domain.add(inter[2])

        for domain in user_domain:
            if domain in domain_users.keys():
                domain_users[domain] += 1
            else:
                domain_users[domain] = 1

            if len(user_domain) > 1:
                if domain in domain_other_users.keys():
                    domain_other_users[domain] += 1
                else:
                    domain_other_users[domain] = 1
        if len(user_domain) in domain_cnts.keys():
            domain_cnts[len(user_domain)] += 1
        else:
            domain_cnts[len(user_domain)] = 1
    print(domain_users)
    print(domain_other_users)
    print(domain_cnts)


def read_meta(dataset, data_path='../dataset'):
    meta_datas = {}
    meta_file = f'{data_path}/Metadata/meta_{dataset}.json.gz'
    for info in tqdm(parse(meta_file), desc='Loading meta'):
        info['dataset'] = dataset
        meta_datas[info['asin']] = info

    return meta_datas


def read_meta_2023(dataset, data_path='../dataset'):
    meta_datas = {}
    ## meta_file = 
    
    for info in tqdm(meta_file, desc='Loading meta'):
        info['dataset'] = dataset
        meta_datas[info['parent_asin']] = info

    return meta_datas

def read_review(dataset, meta_data, data_path='../dataset', ratio=1.0):
    review_datas = {}
    # older Amazon
    # data_file = f'{data_path}/Ratings/{dataset}.csv'
    data_file = f'./dataset/Ratings/Source/{dataset}.csv.gz'
    total_items = meta_data.keys()
    df = pd.read_csv(data_file)
    
    for _, row in tqdm(df.iterrows(), desc='Loading ratings'):
        user = row['user_id']
        parent_asin = row['parent_asin']
        time = row['timestamp']
        rating = row['rating']
        
        if user not in review_datas.keys():
            review_datas[user] = []
        if [parent_asin, rating, dataset, time] in review_datas[user]:
            continue
        review_datas[user].append([parent_asin, rating, dataset, time])

    new_review_datas = {}
    raw_inters = 0
    for user in tqdm(review_datas.keys(), desc='Delete no meta'):
        raw_inters += len(review_datas[user])
        new_review_datas[user] = []
        for inter in review_datas[user]:
            if inter[0] in total_items:
                new_review_datas[user].append(inter)
    print(f"Raw inters:{raw_inters}")

    review_datas = new_review_datas
    meta_filter_inters = 0
    for user in review_datas.keys():
        meta_filter_inters += len(review_datas[user])
    print(f"Meta filter inters:{meta_filter_inters}")

    import random
    random.seed(2023)

    sampled_user_file = f"./dataset/Sampled_10k/{dataset}_10k.json"

    with open(sampled_user_file, 'r') as f:
        selected_user = json.load(f)
    selected_user = list(selected_user.keys())

    selected_review_data = {}
    for user in selected_user:
        selected_review_data[user] = review_datas[user]
    return selected_review_data


def cal_user_item_count(review_datas):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, user_items in tqdm(review_datas.items(), desc='cal user item count'):
        user_count[user] += len(user_items)
        for item in user_items:
            item_count[item[0]] += 1
    assert sum(user_count.values()) == sum(item_count.values())
    return user_count, item_count


def k_core(review_datas, user_k=5, item_k=5):
    new_review_datas = defaultdict(list)
    epoch = 0
    while True:
        start_user_count, start_item_count = cal_user_item_count(review_datas)
        print(
            f"Epoch:{epoch} five core START | User Count:{len(start_user_count)} | Item Count:{len(start_item_count)} | Reviews:{sum(start_user_count.values())}")

        # user five-core
        users = list(review_datas.keys())
        for user in users:
            if len(review_datas[user]) < user_k:
                del review_datas[user]

        # item five-core
        users = list(review_datas.keys())
        item_count = defaultdict(int)
        for user in users:
            for item in review_datas[user]:
                item_count[item[0]] += 1

        for user in users:
            for item in review_datas[user]:
                if item_count[item[0]] >= item_k:
                    new_review_datas[user].append(item)

        review_datas = new_review_datas
        new_review_datas = defaultdict(list)

        end_user_count, end_item_count = cal_user_item_count(review_datas)
        print(
            f"Epoch:{epoch} END | User Count:{len(end_user_count)} | Item Count:{len(end_item_count)}  | Reviews:{sum(end_user_count.values())}")
        if len(start_user_count) == len(end_user_count) and len(start_item_count) == len(end_item_count):
            break
        epoch += 1

    print(
        f"Finish 5-core | Users:{len(end_user_count)} | Items:{len(end_item_count)} | Reviews:{sum(end_user_count.values())}")
    return review_datas


def sort_review_by_time(review_datas):
    for user in tqdm(review_datas.keys(), desc='Sorting by time'):
        review_datas[user].sort(key=lambda x: (x[-1], x[0], x[1]))

    dup_inter = 0
    new_review_datas = {}
    for user in tqdm(review_datas.keys(), desc='Deleting duplicate interaction'):
        user_inters = [review_datas[user][0]]
        for inter_idx in range(1, len(review_datas[user])):
            if review_datas[user][inter_idx] != user_inters[-1]:
                user_inters.append(review_datas[user][inter_idx])
            else:
                dup_inter += 1
        new_review_datas[user] = user_inters
    print(f"Delete duplicate {dup_inter} interactions!")
    return new_review_datas


def clean_meta(review_datas, meta_datas):
    items = set()
    for user, user_items in review_datas.items():
        for item in user_items:
            items.add(item[0])

    for meta_item in list(meta_datas.keys()):
        if meta_item not in items:
            del meta_datas[meta_item]
    return meta_datas


def re_item_id(meta_datas):
    items_asin = meta_datas.keys()
    items_id = list(range(1, len(items_asin) + 1))
    asin2iid = {}
    iid2asin = {}
    for asin, iid in zip(items_asin, items_id):
        asin2iid[asin] = iid
        iid2asin[iid] = asin
    return asin2iid, iid2asin

def re_user_id(review_datas):
    new_review_datas = {}
    uid2rid = {}
    rid2uid = {}
    for user, user_inters in review_datas.items():
        new_review_datas[user] = user_inters
        uid2rid[len(uid2rid)+1] = user
        rid2uid[user] = len(rid2uid)+1
    return new_review_datas, uid2rid, rid2uid

def transfer_review_iid(review_datas, asin2iid):
    for user in review_datas.keys():
        for inter in review_datas[user]:
            inter[0] = asin2iid[inter[0]]
    return review_datas


def get_item_text_info(meta_datas, iid2asin):
    text_list = ["None"]
    for i in range(1, len(iid2asin) + 1):
        item_meta = meta_datas[iid2asin[i]]
        title = item_meta['title'] if ('title' in item_meta.keys()) and item_meta['title'] else "None"
        price = item_meta['price'] if ('price' in item_meta.keys()) and item_meta['price'] else "None"
        category = item_meta['category'] if ('category' in item_meta.keys()) and item_meta['category'] else "None"
        brand = item_meta['brand'] if ('brand' in item_meta.keys()) and item_meta['brand'] else "None"
        item_text = f"{title}; {category}; {brand};"
        text_list.append(item_text)
    return text_list


def merge_review_datas(total_review_datas, review_datas):
    # Mixed
    for user, reviews in review_datas.items():
        if user in total_review_datas.keys():
            total_review_datas[user].extend(reviews)
        else:
            total_review_datas[user] = reviews

    # Separate

    # for user, reviews in review_datas.items():
    #     if user in total_review_datas.keys():
    #         total_review_datas[user].extend(reviews)
    #     else:
    #         total_review_datas[user+'_'+reviews[0][2]] = reviews


def save_pickle(datas, dataset, name):
    import pickle
    os.makedirs(f'{args.data_path}/{dataset}', exist_ok=True)
    pickle.dump(datas, open(f"{args.data_path}/{dataset}/{name}.pkl", 'wb'))

def load_pickle(dataset, name):
    import pickle
    return pickle.load(open(f"{args.data_path}/{dataset}/{name}.pkl", 'rb'))


def process(datasets):
    datasets = datasets.split(',')
    total_review_datas = {}
    total_meta_datas = {}

    print(len(datasets))
    for dataset in datasets:
        print(f"Processing - {dataset}")
        # dataset = amazon_dataset2fullname[dataset]
        # download_file(dataset, args.data_path)
        meta_datas = read_meta_2023(dataset, args.data_path)  # read meta data
        review_datas = read_review(dataset, meta_datas, args.data_path, args.ratio)  # read Review data
        merge_review_datas(total_review_datas, review_datas)
        total_meta_datas.update(meta_datas)

    # total_review_datas = k_core(total_review_datas, args.user_k, args.item_k)  # Review five-core
    total_meta_datas = clean_meta(total_review_datas, total_meta_datas)  # clean meta data

    total_review_datas = sort_review_by_time(total_review_datas)            # sorting
    asin2iid, iid2asin = re_item_id(total_meta_datas)  # re-index item
    total_review_datas, uid2rid, rid2uid = re_user_id(total_review_datas)   # re-index user
    total_review_datas = transfer_review_iid(total_review_datas, asin2iid)  # update review
    analysis_diff_user(total_review_datas)
    analysis_diff_dataset(total_review_datas)

    if len(datasets) == 1:
        dataset = 'm' + datasets[0]
    else:
        dataset = 'm_' + ''.join(x[0] for x in datasets)                  
    single_domain_iid = {}
    for domain in datasets:
        single_domain_iid[domain] = []
    for asin in total_meta_datas.keys():
        single_domain_iid[total_meta_datas[asin]['dataset']].append(asin2iid[asin])
    save_pickle(single_domain_iid, f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", "single_domain_iid")
    save_pickle(total_meta_datas, f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", "meta_datas")           
    save_pickle(total_review_datas, f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", "review_datas")   
    save_pickle(asin2iid, f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", "asin2iid")                 
    save_pickle(iid2asin, f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", "iid2asin")               
    save_pickle(uid2rid, f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", "uid2rid")                 
    save_pickle(rid2uid, f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", "rid2uid")                 

from tqdm import tqdm
def generate_negatives(dataset):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2023)
        torch.cuda.manual_seed_all(2023)
    torch.manual_seed(2023)

    single_domain_iid = load_pickle(f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", "single_domain_iid")
    review_datas = load_pickle(f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", 'review_datas')

    valid_negatives = []
    test_negatives = []
    import random
    random.seed(42)
    for user in tqdm(review_datas.keys()):

        valid_negatives.append(random.sample(single_domain_iid[review_datas[user][-2][-2]], args.nega_count))
        test_negatives.append(random.sample(single_domain_iid[review_datas[user][-1][-2]], args.nega_count))
        # valid_negatives.append(single_domain_iid[review_datas[user][-2][-2]])
        # test_negatives.append(single_domain_iid[review_datas[user][-1][-2]])
    args.nega_count = '1000'
    save_pickle(torch.LongTensor(valid_negatives), f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", f'negatives_valid-{args.nega_count}')
    save_pickle(torch.LongTensor(test_negatives), f"{dataset}-{args.ratio}-{args.user_k}-{args.item_k}", f'negatives_test-{args.nega_count}')





if __name__ == '__main__':
    args = parse_args()

    args.dataset = "Industrial_and_Scientific,Office_Products,Automotive,Tools_and_Home_Improvement,Books,CDs_and_Vinyl"
    args.nega_count = 100
    process(args.dataset)
    if len(args.dataset.split(',')) == 1:
        dataset = 'm' + args.dataset
    else:
        dataset = 'm_' + ''.join(x[0] for x in args.dataset.split(','))                                       
    generate_negatives(dataset)
