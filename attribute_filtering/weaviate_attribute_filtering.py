import weaviate
import utils
import pandas as pd
import numpy as np
import uuid
import time
import csv
import random
import pickle
import os
from dotenv import load_dotenv
from os import getenv

load_dotenv()
url = f"http://{os.getenv('WEAVIATE_URL')}:{os.getenv('WEAVIATE_PORT')}"
print(f"Connecting to {url}...")
client = weaviate.Client(url)  # Replace the URL with that of your Weaviate instance

ef = [64, 128, 256, 512]
maxConnections =  [8, 16, 32, 64]
efConstruction = [64, 128, 256, 512]
k_values = [1,10,100]

def main():
    for ef_val in ef:
        for m_val in maxConnections:
            for efc_val in efConstruction:
                to_name = 'Siftsmall_HSNW_index' + '_' + str(ef_val) + '_' + str(m_val) + '_' + str(efc_val)
                create_HSNW_collections(ef_val, m_val, efc_val, to_name)
                print('appended', to_name)
    create_flat_index()

def data_pickle():
    with open(os.getenv('PICKLE_QUERY_VECTORS_PATH'), 'rb') as f:
        base_vectors_with_attributes = pickle.load(f)
    with open(os.getenv('PICKLE_BASE_VECTORS_PATH'), 'rb') as f:
        query_vectors_with_attributes = pickle.load(f)
    with open(os.getenv('PICKLE_GROUND_TRUTH_PATH'), 'rb') as f:
        truth = pickle.load(f)
    return base_vectors_with_attributes, query_vectors_with_attributes, truth

base_vectors_with_attributes, query_vectors_with_attributes, truth = data_pickle()

def upload_data(class_name):
    data_objs = [
        {"attr1": attr1, "attr2": attr2, "attr3": attr3}
        for attr1, attr2, attr3 in zip(base_vectors_with_attributes["attr1"].tolist(), base_vectors_with_attributes["attr2"].tolist(), base_vectors_with_attributes["attr3"].tolist())
    ]
    vectors = base_vectors_with_attributes["vector"].tolist()
    uuids_in_dataframe = base_vectors_with_attributes["uuid"].tolist()
    client.batch.configure(batch_size=100)  # Configure batch
    with client.batch as batch:
        for i, data_obj in enumerate(data_objs):
                batch.add_data_object(
                data_obj,
                class_name,
                vector=vectors[i],
                uuid= uuids_in_dataframe[i]
            )

def create_flat_index():
    #with Flat index
    class_name = "Siftsmall_flat_index" 
    distance_metric = "l2-squared"
    client.schema.delete_class(class_name)
    # Class definition object. Weaviate's autoschema feature will infer properties when importing.
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "vectorIndexType": "flat",
        "vectorIndexConfig": {
        "distance": distance_metric
        },
        "properties": [{
            "name": "attr1",
            "dataType": ["boolean"]},{
            "name": "attr2",
            "dataType": ["boolean"]},{
            "name": "attr3",
            "dataType": ["boolean"]}, 
            ]
    }
    # Add the class to the schema
    client.schema.create_class(class_obj)
    upload_data(class_name)
    print(class_name, ' uploaded')
    for k in k_values:
        run_query(class_name, k, query_vectors_with_attributes, base_vectors_with_attributes)
    client.schema.delete_class(class_name)

def create_HSNW_collections(ef, maxConnections, efConstruction, to_name):
    class_name = to_name
    distance_metric = "l2-squared"
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
            "distance": distance_metric},
        "properties": [{
                "name": "attr1",
                "dataType": ["boolean"]},{
                "name": "attr2",
                "dataType": ["boolean"]},{
                "name": "attr3",
                "dataType": ["boolean"]}, 
                ]
    }
    # Add the class to the schema
    client.schema.create_class(class_obj)
    upload_data(class_name, )

    for k in k_values:
         run_query(class_name, k, query_vectors_with_attributes, base_vectors_with_attributes)
    client.schema.delete_class(class_name)

def run_query(name, k, query_vec, base_vec):
    result = []
    start_time = time.time()
    for _, elem in query_vec.iterrows():
        
        vec = elem["vector"]
        attr1 = elem["attr1"]
        attr2 = elem["attr2"]
        attr3 = elem["attr3"]

        response = (
            client.query
            .get(name)
            .with_near_vector({
            "vector": vec
            })
            .with_limit(k)
            .with_additional(["id"])
            .with_where({
            "operator": "And",
            "operands": [
            {
                "path": ["attr1"],
                "operator": "Equal",
                "valueBoolean": attr1
            },
            {
                "path": ["attr2"],
                "operator": "Equal",
                "valueBoolean": attr2
            },
            {
                "path": ["attr3"],
                "operator": "Equal",
                "valueBoolean": attr3
            }
            ]
            })  
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
        print(base_vec[base_vec['uuid'].isin(i)].index)

    metrics(end_time, start_time, name, result_indexes, result, k)

def metrics(end_time, start_time, name, result_indexes, result, k):
    time_taken = end_time - start_time
    queries_count = len(result)
    throughput = queries_count / time_taken
    average_latency = time_taken / queries_count
    print(f"Throughput: of {name} {throughput:.2f} queries per second")
    print(f"Average Latency: of {name} {average_latency:.4f} seconds")
    print(f"k is {k}")

    true_positives = 0
    n_classified = 0
    for i,elem in enumerate(result_indexes):
        print(truth[i])

        true_positives_iter = len(np.intersect1d(truth[i][:k], elem))
        true_positives += true_positives_iter
        n_classified += len(elem)
    print(f'Average recall: of {name} {true_positives/n_classified}')
    
    with open("results_attribute_weviate.csv", mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([name, k, (true_positives/n_classified), throughput, average_latency])