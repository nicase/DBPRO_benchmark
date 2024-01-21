from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import random, time, utils
import pandas as pd

def setup_client():
    qdrantClient = QdrantClient(host='localhost', port=6333, timeout=10000000)
    return qdrantClient

def read_dataset():
    # base_vectors = utils.read_fvecs("../../dataset/siftsmall/siftsmall_base.fvecs")
    # query_vectors = utils.read_fvecs("../../dataset/siftsmall/siftsmall_query.fvecs")
    # knn_groundtruth = utils.read_ivecs("../../dataset/siftsmall/siftsmall_groundtruth.ivecs")
    base_vectors = utils.read_fvecs("../../dataset/sift/sift_base.fvecs")
    query_vectors = utils.read_fvecs("../../dataset/sift/sift_query.fvecs")
    knn_groundtruth = utils.read_ivecs("../../dataset/sift/sift_groundtruth.ivecs")
    return base_vectors, query_vectors, knn_groundtruth

def create_collection(qdrantClient, collection_name):
    vector_size = 128

    qdrantClient.delete_collection(collection_name=collection_name)

    qdrantClient.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.EUCLID),
    )

def insert_values(base_vectors, qdrantClient, collection_name):
    batch_size = 50000
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

def search_queries(query_vectors, qdrantClient, collection_name):
    print(f'Search function starting')
    result_ids = []
    for i,elem in enumerate(query_vectors):
        print(f'Progress: {i}/{len(query_vectors)}', end='\r')
        search_result = qdrantClient.search(
            collection_name=collection_name, 
            query_vector=elem, 
            limit=100
        )
        result_ids.append([elem.id for elem in search_result])

    return result_ids

def print_metrics(result_ids, nqueries, time_span, truth):
    true_positives = 0
    n_classified = 0
    for i,elem in enumerate(result_ids):
        true_positives_iter = len(np.intersect1d(truth[i], result_ids[i]))
        true_positives += true_positives_iter
        n_classified += len(elem)
    print(true_positives)
    print(n_classified)
    print(f'QPS = {(nqueries / time_span):.4f}')
    print(f'Average recall: {true_positives/n_classified}')

def main():
    qdrantClient = setup_client()
    baseV, queryV, groundTruthV = read_dataset()

    collection_name = "ann_1M"
    create_collection(qdrantClient, collection_name)

    print("---------------------------------- INSERTING VALUES ----------------------------------")
    insert_values(baseV, qdrantClient, collection_name)
    print("-------------------------------------------------------------------------------------")

    # truth = utils.top_k_neighbors(queryV, baseV, k=100, function='euclidean', filtering=False) 

    start_time = time.time()
    result_ids = search_queries(queryV, qdrantClient, collection_name)
    end_time = time.time()
    time_span = end_time - start_time

    print_metrics(result_ids, len(queryV), time_span, groundTruthV)

if __name__ == "__main__":
    main()