from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import random, time, utils, datetime, csv
import pandas as pd
from dotenv import load_dotenv
import os
import pickle

def setup_client():
    qdrantClient = QdrantClient(host=os.getenv('QDRANT_URL'), port=os.getenv('QDRANT_PORT'), timeout=10000000)
    return qdrantClient

def read_dataset():
    base_vectors = utils.read_fvecs(os.getenv('BASE_VECTORS_PATH_TEST'))
    query_vectors = utils.read_fvecs(os.getenv('QUERY_VECTORS_PATH_TEST'))
    with open(f'{os.getenv("ANN_GROUND_TRUTH_DIR")}ANN_SIFT_DOT_GT.pkl', 'rb') as f:
        groundTruth = pickle.load(f)
    return base_vectors, query_vectors, groundTruth

def create_collection(qdrantClient, collection_name, ef_construct, m):
    vector_size = 128

    qdrantClient.delete_collection(collection_name=collection_name)

    qdrantClient.recreate_collection(
        collection_name=collection_name,
        hnsw_config=models.HnswConfigDiff(
            m=m,
            ef_construct=ef_construct
        ),
        vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
    )

def insert_values(base_vectors, qdrantClient, collection_name):
    batch_size = 25000
    num_batches = len(base_vectors) // batch_size + int(len(base_vectors) % batch_size > 0)
    print(f'Number of batches: {num_batches}')

    for batch_idx in range(num_batches):
        print(f'Current progress: {(batch_idx+1)*batch_size}/{len(base_vectors)}', end='\r')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(base_vectors))

        batch_vectors = base_vectors[start_idx:end_idx]

        qdrantClient.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=list(range(start_idx, end_idx)),
                vectors=batch_vectors.tolist()
            )
        )

def search_queries(query_vectors, qdrantClient, collection_name, ef, k):
    print(f'Search function starting')
    result_ids = []
    for i,elem in enumerate(query_vectors):
        print(f'Progress: {i}/{len(query_vectors)}', end='\r')
        search_result = qdrantClient.search(
            collection_name=collection_name, 
            search_params=models.SearchParams(
                hnsw_ef=ef,
                exact=False
            ), 
            query_vector=elem, 
            limit=k
        )
        result_ids.append([elem.id for elem in search_result])

    return result_ids

def print_metrics(ef_construct, m, ef, k, qps, recall, file_name, time_span_insert, time_span_search, timestamp):
    with open(file_name,'a') as fd:
        fd.write(f'{ef_construct}, {m}, {ef}, {k}, {qps}, {recall}, {time_span_insert}, {time_span_search}, {timestamp}\n')

def main():
    load_dotenv()
    ef_construct_values = [64, 128, 256, 512]
    m_values = [8, 16, 32, 64]
    ef_values = [64, 128, 256, 512]

    # Create csv file
    current_date = datetime.datetime.now().strftime("%d_%m_%y_%H:%M")
    headers = ["ef_construct", "m", "ef", "k", "qps", "recall", "time_span_insert", "time_span_search", "time_span_points", "timestamp"]
    file_name = f"qdrant_SIFT_dot_ann{current_date}.csv"
    # print(datetime)

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    # return

    qdrantClient = setup_client()
    baseV, queryV, groundTruthV = read_dataset()
    for ef_construct in ef_construct_values:
        for m in m_values:
            for ef in ef_values:
                print("---------------------------------- CURRENT CONFIGURATION ----------------------------------")
                print(f'ef_construct: {ef_construct} \nm: {m}\nef: {ef}')
                print("-------------------------------------------------------------------------------------------")

                collection_name = "ANN_SIFT_DOT"
                create_collection(qdrantClient, collection_name, ef_construct, m)

                print("---------------------------------- INSERTING VALUES ----------------------------------")
                start_time_insert = time.time()
                insert_values(baseV, qdrantClient, collection_name)
                end_time_insert = time.time()
                time_span_insert = end_time_insert - start_time_insert
                print("-------------------------------------------------------------------------------------")

                # truth = utils.top_k_neighbors(queryV, baseV, k=100, function='euclidean', filtering=False) 
                k_values = [1, 10, 100]
                for k in k_values:
                    start_time_search = time.time()
                    result_ids = search_queries(queryV, qdrantClient, collection_name, ef, k)
                    end_time_search = time.time()
                    time_span_search = end_time_search - start_time_search

                    true_positives = 0
                    n_classified = 0
                    for i,elem in enumerate(result_ids):
                        true_positives_iter = len(np.intersect1d(groundTruthV[i][:k], result_ids[i]))
                        true_positives += true_positives_iter
                        n_classified += len(elem)

                    qps = len(queryV) / time_span_search
                    recall = true_positives/n_classified

                    print_metrics(ef_construct, m, ef, k, qps, recall, file_name, time_span_insert, time_span_search, datetime.datetime.now().strftime("%d_%m_%y_%H:%M"))




if __name__ == "__main__":
    main()