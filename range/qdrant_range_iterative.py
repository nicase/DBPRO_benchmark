from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import random, time, utils, datetime, csv, pickle
import pandas as pd
from dotenv import load_dotenv
import os

def setup_client():
    qdrantClient = QdrantClient(host=os.getenv('QDRANT_URL'), port=os.getenv('QDRANT_PORT'), timeout=10000000)
    return qdrantClient

def read_dataset():
    # base_vectors = utils.read_fvecs("../../dataset/siftsmall/siftsmall_base.fvecs")
    # query_vectors = utils.read_fvecs("../../dataset/siftsmall/siftsmall_query.fvecs")
    # knn_groundtruth = utils.read_ivecs("../../dataset/siftsmall/siftsmall_groundtruth.ivecs")
    with open(os.getenv('PICKLE_QUERY_VECTORS_PATH_RANGE'), 'rb') as f:
        query_vectors_with_attributes = pickle.load(f)

    with open(os.getenv('PICKLE_BASE_VECTORS_PATH_RANGE'), 'rb') as f:
        base_vectors_with_attributes = pickle.load(f)


    return base_vectors_with_attributes, query_vectors_with_attributes

def create_collection(qdrantClient, collection_name, ef_construct, m):
    vector_size = 128

    qdrantClient.delete_collection(collection_name=collection_name)

    qdrantClient.recreate_collection(
        collection_name=collection_name,
        hnsw_config=models.HnswConfigDiff(
            m=m,
            ef_construct=ef_construct
        ),
        vectors_config=VectorParams(size=vector_size, distance=Distance.EUCLID),
    )

def insert_values(n, batch_points, qdrantClient, collection_name):
    batch_size = 50000
    num_batches = n // batch_size + int(n % batch_size > 0)
    print(f'Number of batches: {num_batches}')

    for batch_idx in range(num_batches):
        print(f'Current progress: {(batch_idx+1)*batch_size}/{n}', end='\r')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n)

        batch_points_i = batch_points[start_idx:end_idx]

        operation_info = qdrantClient.upsert(
            collection_name=collection_name,
            wait=True,
            points=batch_points_i
        )

def search_queries(query_vectors, qdrantClient, collection_name, threshold, ef):
    print(f'Search function starting')
    result_ids = []
    for i,elem in query_vectors.iterrows():
        print(f'Progress: {i}/{len(query_vectors)}', end='\r')
        # print(elem)
        vec = elem["vector"]
        # print(attr1, attr2, attr3)
        search_result = qdrantClient.search(
            collection_name=collection_name,
            search_params=models.SearchParams(
                hnsw_ef=ef,
                exact=False
            ),
            query_vector=vec, 
            score_threshold = threshold,
            limit=1000000
        )
        # print(f'Found {len(search_result)} ids under threshold')
        # print(np.array([elem.id for elem in search_result]))
        result_ids.append([elem.id for elem in search_result])

    return result_ids

def print_metrics(ef_construct, m, ef, qps, recall, file_name, time_span_insert, time_span_search, time_span_points, timestamp):
    with open(file_name,'a') as fd:
        fd.write(f'{ef_construct}, {m}, {ef}, {qps}, {recall}, {time_span_insert}, {time_span_search}, {time_span_points}, {timestamp}\n')


def main():
    load_dotenv()
    ef_construct_values = [64, 128, 256, 512]
    m_values = [8, 16, 32, 64]
    ef_values = [64, 128, 256, 512]


    # Create csv file
    current_date = datetime.datetime.now().strftime("%d_%m_%y_%H:%M")
    headers = ["ef_construct", "m", "ef", "qps", "recall", "time_span_insert", "time_span_search", "time_span_points", "timestamp"]
    file_name = f"qdrant_range_{current_date}.csv"
    # print(datetime)

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    # return

    qdrantClient = setup_client()
    baseV, queryV = read_dataset()
    # return
    for ef_construct in ef_construct_values:
        for m in m_values:
            for ef in ef_values:
                print("---------------------------------- CURRENT CONFIGURATION ----------------------------------")
                print(f'ef_construct: {ef_construct} \nm: {m}\nef: {ef}')
                print("-------------------------------------------------------------------------------------------")

                collection_name = "range_10K"
                create_collection(qdrantClient, collection_name, ef_construct, m)

                print("---------------------------------- INSERTING VALUES ----------------------------------")
                
                print("Creating Points")
                start_time = time.time()
                batch_points = [PointStruct(id=i, vector=elem["vector"]) for i, elem in baseV.iterrows()]
                end_time = time.time()
                time_span_points = end_time - start_time

                start_time_insert = time.time()
                insert_values(len(baseV), batch_points, qdrantClient, collection_name)
                end_time_insert = time.time()
                time_span_insert = end_time_insert - start_time_insert
                print("-------------------------------------------------------------------------------------")

                # truth = utils.top_k_neighbors(queryV, baseV, k=100, function='euclidean', filtering=False) 

                start_time_search = time.time()
                result_ids = search_queries(queryV, qdrantClient, collection_name, 600, ef)
                end_time_search = time.time()
                time_span_search = end_time_search - start_time_search

                true_positives = 0
                n_classified = 0
                for i,elem in enumerate(result_ids):
                    groundTruth_i = utils.range_truth_iterative(queryV.iloc[i].values[0], baseV, 600)
                    true_positives_iter = len(np.intersect1d(groundTruth_i, result_ids[i]))
                    true_positives += true_positives_iter
                    n_classified += len(elem)

                qps = len(queryV) / time_span_search
                recall = true_positives/n_classified

                print_metrics(ef_construct, m, ef, qps, recall, file_name, time_span_insert, time_span_search, time_span_points, datetime.datetime.now().strftime("%d_%m_%y_%H:%M"))


if __name__ == "__main__":
    main()