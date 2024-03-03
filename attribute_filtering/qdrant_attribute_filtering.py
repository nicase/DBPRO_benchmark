from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import random, time, utils
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

def insert_values(base_vectors, batch_points, qdrantClient, collection_name):
    batch_size = 50000
    num_batches = len(base_vectors) // batch_size + int(len(base_vectors) % batch_size > 0)
    print(f'Number of batches: {num_batches}')

    for batch_idx in range(num_batches):
        print(f'Current progress: {(batch_idx+1)*batch_size}/{len(base_vectors)}', end='\r')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(base_vectors))

        batch_points_i = batch_points[start_idx:end_idx]

        operation_info = qdrantClient.upsert(
            collection_name=collection_name,
            wait=True,
            points=batch_points_i
        )

def search_queries(query_vectors_with_attributes, qdrantClient):
    print(f'Search function starting')
    result_ids = []
    for _,elem in query_vectors_with_attributes.iterrows():
        # print(elem)
        vec = elem["vector"]
        attr1 = elem["attr1"]
        attr2 = elem["attr2"]
        attr3 = elem["attr3"]
        # print(attr1, attr2, attr3)
        search_result = qdrantClient.search(
            collection_name="attribute_filtering", 
            query_vector=vec, 
            query_filter=models.Filter(
                # must = AND
                must=[
                    models.FieldCondition(
                        key="attr1",
                        match=models.MatchValue(
                            value=attr1,
                        ),
                    ),
                    models.FieldCondition(
                        key="attr2",
                        match=models.MatchValue(
                            value=attr2,
                        ),
                    ),
                    models.FieldCondition(
                        key="attr3",
                        match=models.MatchValue(
                            value=attr3,
                        ),
                    )
                ]
            ),
            limit=100
        )
        result_ids.append([elem.id for elem in search_result])

    return result_ids

def print_metrics(result_ids, nqueries, time_span, truth):
    true_positives = 0
    n_classified = 0
    for i,elem in enumerate(result_ids):
        true_positives_iter = len(np.intersect1d(truth[i], elem))
        true_positives += true_positives_iter
        n_classified += len(elem)

    print(f'QPS = {(nqueries / time_span):.4f}')
    print(f'Average recall: {true_positives/n_classified}')

def main():
    load_dotenv()
    qdrantClient = setup_client()
    baseV, queryV, groundTruthV = read_dataset()

    collection_name = "attribute_filtering_1M"
    create_collection(qdrantClient, collection_name)
    # print(baseV, queryV, groundTruthV)

    baseV_with_attributes = pd.DataFrame({'vector': baseV.tolist()})
    num_rows = len(baseV_with_attributes)
    baseV_with_attributes['attr1'] = [random.choice([True, False]) for _ in range(num_rows)]
    baseV_with_attributes['attr2'] = [random.choice([True, False]) for _ in range(num_rows)]
    baseV_with_attributes['attr3'] = [random.choice([True, False]) for _ in range(num_rows)]

    start_time = time.time()
    batch_points = [PointStruct(id=i, vector=elem["vector"], payload= {"attr1": elem["attr1"], "attr2": elem["attr2"], "attr3": elem["attr3"]}) for i, elem in baseV_with_attributes.iterrows()]
    end_time = time.time()
    time_span = end_time - start_time
    print(f'Points created in {time_span}')


    insert_values(baseV, batch_points, qdrantClient, collection_name)

    # Need to wait for status green here?

    queryV_with_attributes = pd.DataFrame({'vector': queryV.tolist()})
    num_rows = len(queryV_with_attributes)
    queryV_with_attributes['attr1'] = [random.choice([True, False]) for _ in range(num_rows)]
    queryV_with_attributes['attr2'] = [random.choice([True, False]) for _ in range(num_rows)]
    queryV_with_attributes['attr3'] = [random.choice([True, False]) for _ in range(num_rows)]

    truth = utils.top_k_neighbors(queryV_with_attributes, baseV_with_attributes, k=100, function='euclidean', filtering=True) 


    start_time = time.time()
    result_ids = search_queries(queryV_with_attributes, qdrantClient)
    end_time = time.time()
    time_span = end_time - start_time

    print_metrics(result_ids, len(queryV), time_span, truth)


if __name__ == "__main__":
    main()