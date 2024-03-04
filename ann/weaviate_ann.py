import weaviate
import os
import utils
import pandas as pd
import numpy as np
import uuid
import time
import csv
from dotenv import load_dotenv
import h5py
import datetime
import pickle

base_vectors = []
query_vectors = []
truth = []
batch_size_value = 25000

distance_metrics = ['l2-squared', 'cosine', 'dot']
ef = [64, 128, 256, 512]
maxConnections =  [8, 16, 32, 64]
efConstruction = [64, 128, 256, 512]
k_values = [1,10,100]

load_dotenv()
url = f"http://{os.getenv('WEAVIATE_URL')}:{os.getenv('WEAVIATE_PORT')}"
print(f"Connecting to {url}...")
client = weaviate.Client(url)

def main(dataset_name, distance_metric_name):
    if distance_metric_name == 'l2-squared':
        distance_metric_to_name = 'euclidean'
    else:
        distance_metric_to_name = distance_metric_name

    for ef_val in ef:
        for m_val in maxConnections:
            for efc_val in efConstruction:
                to_name = dataset_name + '_' + 'HSNW' + '_' + distance_metric_to_name + '_' +  str(ef_val) + '_' + str(m_val) + '_' + str(efc_val)
                create_HSNW_collections(ef_val, m_val, efc_val, to_name, distance_metric_name)
                print('appended', to_name)
    create_flat_index(dataset_name, distance_metric_name)

def load_SIFT():
    base_vectors = pd.DataFrame({'vector': utils.read_fvecs(os.getenv('BASE_VECTORS_PATH')).tolist()})
    query_vectors = pd.DataFrame({'vector': utils.read_fvecs(os.getenv('QUERY_VECTORS_PATH')).tolist()})
    
    # with open(os.getenv('ANN_SIFT_COSINE_GT_PATH'), 'rb') as f:
    #     cosine_truth = pickle.load(f)
    
    # with open(os.getenv('ANN_SIFT_DOT_GT_PATH'), 'rb') as f:
    #     dot_truth = pickle.load(f)

    euc_truth = utils.read_ivecs(os.getenv('GROUND_TRUTH_PATH'))

    print('SIFT loaded')
    base_vectors['uuid'] = base_vectors.apply(lambda row: generate_uuid(), axis=1)
    print('SIFT uuid generated')
    return base_vectors, query_vectors, euc_truth

def load_GLOVE():
    query = []
    base = []
    with h5py.File(os.getenv('H5PY_GLOVE_PATH'), 'r') as hdf_file:
        for a in hdf_file['train']:
            base.append(list(a))
        for a in hdf_file['test']:
            query.append(list(a))

    base = pd.DataFrame({'vector': base}).head(200)
    query = pd.DataFrame({'vector': query}).head(200)

    with open(os.getenv("ANN_gloVe_EUCLIDEAN_GT_PATH"), 'rb') as f:
        euc_truth = pickle.load(f)

    with open(os.getenv("ANN_gloVe_COSINE_GT_PATH"), 'rb') as f:
        cosine_truth = pickle.load(f)
    
    with open(os.getenv('ANN_SIFT_DOT_GT_PATH'), 'rb') as f:
        dot_truth = pickle.load(f)

    print('GLOVE loaded')
    base['uuid'] = base.apply(lambda row: generate_uuid(), axis=1)
    print('GLOVE uuid generated')
    return base, query, [euc_truth, cosine_truth, dot_truth]

def generate_uuid():
    return str(uuid.uuid4())

def upload_data(class_name):
    uuids_in_dataframe = base_vectors["uuid"].tolist()
    
    data_objs = [
        {"title": f"Object {i+1}"} for i in range(len(base_vectors))
    ]
    client.batch.configure(batch_size=batch_size_value)
    with client.batch as batch:
        for i, data_obj in enumerate(data_objs):
            batch.add_data_object(
                data_obj,
                class_name,
                vector= base_vectors["vector"][i],
                uuid= uuids_in_dataframe[i]
            )

def create_flat_index(dataset_name, distance_metric_name):
    #with Flat index
    class_name = dataset_name + 'flat' 
    distance_metric = distance_metric_name
    client.schema.delete_class(class_name)
    # Class definition object. Weaviate's autoschema feature will infer properties when importing.
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "vectorIndexType": "flat",
        "vectorIndexConfig": {
        "distance": distance_metric
        }
    }
    # Add the class to the schema
    client.schema.create_class(class_obj)
    upload_data(class_name)
    print(class_name, ' uploaded')
    for k in k_values:
        run_query(class_name, k, query_vectors, base_vectors)
    client.schema.delete_class(class_name)

def create_HSNW_collections(ef, maxConnections, efConstruction, to_name, distance_metric_name):
    class_name = to_name
    distance_metric = distance_metric_name
    client.schema.delete_class(class_name)

    # Class definition object. Weaviate's autoschema feature will infer properties when importing.
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig":{
            "skip": False,
            "pq": {"enabled": False,},
            "maxConnections": maxConnections,
            "efConstruction": efConstruction,
            "ef": ef,
            "distance": distance_metric}
    }
    # Add the class to the schema
    client.schema.create_class(class_obj)
    upload_data(class_name)

    for k in k_values:
        run_query(class_name, k, query_vectors, base_vectors)
    client.schema.delete_class(class_name)

def run_query(name, k, query_vec, base_vec):

    result = []
    start_time = time.time()
    for _,elem in query_vec.iterrows():
        
        vec = elem["vector"]

        response = (
            client.query
            .get(name)
            .with_near_vector({
            "vector": vec
            })
            .with_limit(k)
            .with_additional(["id"])  
            .do()
        )
        result.append(response)
    end_time = time.time()
   
    result_ids = []
    for i in range(len(query_vec["vector"])):
        result_ids.append([])
        for j in range(k):
            result_ids[i].append(result[i]["data"]["Get"][name][j]["_additional"]["id"])

    result_indexes = []
    for i in result_ids:
        result_indexes.append(base_vec[base_vec['uuid'].isin(i)].index)

    metrics(end_time, start_time, name, result_indexes, result, k)

def metrics(end_time, start_time, name, result_indexes, result, k):
    time_taken = end_time - start_time
    queries_count = len(result)
    throughput = queries_count / time_taken
    average_latency = time_taken / queries_count

    true_positives = 0
    n_classified = 0
    for i,elem in enumerate(result_indexes):
        true_positives_iter = len(np.intersect1d(truth[i][:k], elem))
        true_positives += true_positives_iter
        n_classified += len(elem)

    print(f'{name}: k:{k} T:{throughput:.2f} L:{average_latency:.4f} R:{true_positives/n_classified}')
    
    with open("results_ann_weviate.csv", mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now(), name, k, (true_positives/n_classified), throughput, average_latency])

base_vectors, query_vectors, truth= load_SIFT()
main('SIFT', distance_metrics[0])

""" base_vectors, query_vectors, truth_vals = load_GLOVE()
base_vectors['vector'] = [[float(x) for x in vector ]for vector in base_vectors['vector']]
query_vectors['vector'] = [[float(x) for x in vector ]for vector in query_vectors['vector']]
for count,distance_metric_name in enumerate(distance_metrics):
   truth = truth_vals[count]
   main('GLOVE', distance_metric_name) """