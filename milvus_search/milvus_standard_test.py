#%%
from milvus import Milvus, IndexType, MetricType, Status
import sys
from glob import glob
import torch
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image
# import insightface
import matplotlib.pyplot as plt
import csv
import time

client = Milvus(uri='tcp://172.21.100.15:19531')
client.list_collections()

client.drop_collection('data')

# Create collection demo_collection if it dosen't exist.
collection_name = 'data'

status, ok = client.has_collection(collection_name)
if not ok:
    param = {
        'collection_name': collection_name,
        'dimension': 512,
        'metric_type': MetricType.IP  # optional
    }
    client.create_collection(param)

_, collection = client.get_collection_info(collection_name)
print("Collection: ", collection)
status, result = client.count_entities(collection_name)
print("Result: ", result)
#%% 
pkl_path = './pkls/mix_logits_r100_BGD_501_ref_cv.pkl'

embs = []

with open(pkl_path, 'rb') as f:
    samples = pickle.load(f)

embs = np.array([np.array(i['emb']) for i in samples])
print(len(embs))
print(len(samples))
#%% convert list to numpy array
knownEmbedding = embs
print(knownEmbedding.shape)

knownNamesId = [i['name'] for i in samples]
print('len Ids: ',len(knownNamesId))
print('len Embs: ',knownEmbedding.shape[0])
# insert true data into true_collection_
status, ids = client.insert(collection_name=collection_name, records=knownEmbedding, ids=list(range(len(knownNamesId))))
if not status.OK():
    print("Insert failed: {}".format(status))
print(len(ids))
print('Status: ',status)
#%%
client.flush([collection_name])
# Get demo_collection row count
status, result = client.count_entities(collection_name)
print(result)
print(status)

ivf_param = {'nlist': 1024}
#%%
# status = client.create_index(collection_name, IndexType.IVF_FLAT, ivf_param)
status = client.create_index(collection_name, IndexType.FLAT, ivf_param)

 # describe index, get information of index
status, index = client.get_index_info(collection_name)
print(index)
#%%
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/glint_r50_QDTaging_mix_3_test_cv.pkl'
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/glint_r50_QDT_elder_23_test_cv.pkl'
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/glint_r50_QDTaging_23_test_cv.pkl'
pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/glint360k_r50_BGD_455_test_cv.pkl'
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/glint360k_r100_BGD_455_test_cv.pkl'
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/mix_logits_r100_BGD_455_test_cv.pkl'
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/mix_logits_r50_BGD_455_test_cv.pkl'
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/sst_r50_455_test_cv.pkl'
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/trip_frz_r50_BGD_455_test_cv.pkl'
# pkl_testPath = '/home/quantum/Desktop/Workspace/chientv/pkls/trip_r50_BGD_455_test_cv.pkl'

with open(pkl_testPath, 'rb') as f:
    test_samples = pickle.load(f)

query_vectors = np.array([i['emb'] for i in test_samples])

print(query_vectors.shape)
#%% 
import random

"""def test_performance():
    n = 500
    period = 0

    for i in tqdm(range(n)):
        random_select = random.choices(range(0, 248258), k = 32)
        test_vectors = query_vectors[random_select, :]

        start = time.time()
        top_k = 7
        
        param = {
            'collection_name': collection_name,
            'query_records': test_vectors,
            'top_k': top_k,
            'params' : {'nprobe': 16 }
        }

        status, results = client.search(**param)
        end = time.time()

        time.sleep(1e-2)
        period += (end-start)
    
    print('\nTop_k:', top_k)
    print('Test shape:',test_vectors.shape)
    print('Avg time:', period/n)"""

# test_performance()
#%%
import math

top_k = 1
hyper_p = math.floor(top_k/2)

param = {
    'collection_name': collection_name,
    'query_records': query_vectors,
    'top_k': top_k,
    'params' : {'nprobe': 16 }
}

status, results = client.search(**param)
print(results, '\n',len(results))
#%%
def top_k_pred(j):
    names = []
    for i in range(top_k):
        names.append(samples[results[j][i].id]['name'])       
    check = {}
    for name in names:
        check[name] = names.count(name)
    _max = 0.
    for key in check.keys():
        if check[key] > _max:
            _max = check[key]
            out_name = key
    
    score = []
    cnt = 0
    for i in range(top_k):
        if samples[results[j][i].id]['name'] == out_name:
            # distance = min(distance, float(results[j][i].distance)) #float(results[j][i].distance)
            score.append(results[j][i].distance)
            cnt += 1

    if top_k > 1:
        if len(score) >= hyper_p:
            return out_name, sum(sorted(score)[-hyper_p:])/hyper_p
        else:
            return None
            # return out_name, sum(score)/len(score)
    elif top_k == 1:
        return out_name, score[0], samples[results[j][0].id]['key']

def top_k_pred_1(j):
    """
    This function takes j as the index of image in the test set
    """
    # take the whole names after searching with milvus
    names = [samples[results[j][i].id]['name'] for i in range(top_k)]

    # count the frequency of each distinct name
    check = {}
    for name in names:
        check[name] = names.count(name)

    # check the name with the largest frequency
    max_key = max(check, key=check.get)
    max_value = max(check.values())

    # start computing the score of the name with maximum appearing time
    score = [results[j][i].distance for i in range(top_k) if samples[results[j][i].id]['name'] == max_key]

    if top_k > 1:
        if len(score) >= hyper_p:            
            return max_key, sum(sorted(score)[-hyper_p:])/hyper_p
        else:
            return None
            # return max_key, max_value, sum(score)/len(score)
    elif top_k == 1:
        return max_key, score[0], samples[results[j][0].id]['key']
print(top_k_pred(0))

def top_k_pred_2(j):
    """
    This function takes j as the index of image in the test set
    """
    # take the whole names after searching with milvus
    names = [samples[results[j][i].id]['name'] for i in range(top_k)]
    
    dict_all = {}
    for name in names:
        scores = [results[j][i].distance for i in range(top_k) if samples[results[j][i].id]['name'] == name]
        if len(scores) > hyper_p:
            dict_all[name] = scores
    # print(dict_all)

    score_dict = {}
    for name, scores in dict_all.items():
        # start computing the score of the name with maximum appearing time
        if hyper_p >= 1 :
            score_dict[name] = sum(sorted(scores)[-hyper_p:])/hyper_p
        else:
            score_dict[name] = scores[0]
    # print(score_dict)
    
    max_key = max(score_dict, key=score_dict.get)
    max_score = score_dict[max_key]
    
    if top_k > 1:
        return max_key, max_score
    elif top_k == 1:
        return max_key, max_score, samples[results[j][0].id]['key']
#%%
scores = []

def analysis(th, f, show=True):
    global c_0, c_1, c_2, c_3, c_4, scores
    print('\nThreshold:', th)
    error = 0
    good_pred = 0
    tot = 0
    pos = 0
    mask = {}
    # for staff in os.listdir('/home/quantum/Desktop/Workspace/BenchMark21/Testing_data'):
    for staff in os.listdir('/home/quantum/Downloads/QDT_aging/All'):
        mask[os.path.basename(staff)]=0
    if show:
        print('\nLen test samples',len(test_samples))
        print('Query vectors',query_vectors.shape[0])
    assert len(test_samples) == query_vectors.shape[0]
    # count = 0
    # hist = {}
    # for name in os.listdir('/home/quantum/Pictures/FaceAging_qdt/aging'):
    #     hist[name] = []

    for j, test_sample in enumerate(test_samples):
        # gt = test_sample['key'].split('/')[-2]
        gt = test_sample['name']
        if top_k > 1:
            try:
                pred_name, score = top_k_pred(j)
                if score >= th:
                    tot+=1
                    if pred_name == gt: #or pred_name == 'Unknown':
                        good_pred +=1
                        mask[gt] = 1                        
                    else:
                        error +=1
                        if show:
                            print('\nWrong - GT: %s, Img: %s, Pred: %s, Score: %f'%(gt, test_sample['key'], pred_name, score))
            except:
                pass                
        elif top_k == 1:
            pred_name, score, ref = top_k_pred(j)            
            if score >= th:
                tot+=1
                if pred_name == gt: #or pred_name == 'Unknown':
                    good_pred +=1
                    mask[gt] = 1
                else:
                    error +=1
                    if show:
                        print('\nWrong - GT: %s, Img: %s, Pred: %s, Score: %f, Ref: %s'%(gt, test_sample['key'], pred_name, score, ref))
    

        print(test_sample['key'], '\t', f'{score:.4f}', 'Pred:', pred_name, '\n', os.path.basename(ref))        
        scores.append(score)
    
    pos_rate = 100*good_pred/len(test_samples)
    print('==> Positive rate {}/{}={:.2f}%'.format(good_pred,len(test_samples),pos_rate))
    
    if show:
        print('\n***Missing:')
        for _, key in enumerate(mask):
            if mask[key] == 1:
                pos+=1
            if mask[key] == 0:
                print(key.replace('/n',''))
        print('\n==> Total miss: %d people'%(len(mask)-pos))

    print('==> Wrong recognizing:',error,'images')
    # print('Truely recognizing: %d/%d'%(good_pred,tot))
    assert tot == error + good_pred
    prec = good_pred/tot*100
    print('==> *Precision: %d/%d = %.3f%%'%(good_pred,tot, prec))
    if show:
        print('==> *Recall: %d/%d = %.3f%%'%(pos, len(mask), pos/len(mask)*100))

    beta = 1
    f1score = (1+beta**2)*(good_pred/tot)*(good_pred/len(test_samples))/(beta**2*good_pred/tot+good_pred/len(test_samples))
    print('==> *F1-score: %.2f'%(f1score))
    if show:
        print('==> mAP: %.2f'%(good_pred/tot*good_pred/len(test_samples)))
    
    out = f'{pos_rate:.2f}%\n{error} imgs\n{prec:.3f}%\n{f1score:.2f}'
    # create the csv write and write a row to the csv file
    # writer = csv.writer(f)
    # writer.writerow([str(th), out])

    return 
#%%
# open the file in the write mode
f = open('/home/quantum/Desktop/Workspace/chientv/excel/milvus_test.csv', 'w', encoding='UTF8')

analysis(0.63, None, show=False)
#%%

from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize =(8, 4))
ax.hist(scores, bins = 10)
print(len(scores), (np.array(scores)<0.63).sum())
plt.savefig('test.png')
# for th in np.arange(0.63, 0.64, 0.001):
# # for th in np.arange(0.66, 0.67, 0.001):
#     print('top_k:',top_k)
#     out = analysis(th, f, False)

# f.close()
# %%
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize =(8, 4))
# for name in os.listdir('/home/quantum/Pictures/FaceAging_qdt/aging'):
arr = np.array(hist['214433_Mac Luu Phong'])
print(f'{(arr>0.63).sum()}/{arr.shape[0]}')
ax.hist(arr, bins = 10)
plt.savefig('test.png')