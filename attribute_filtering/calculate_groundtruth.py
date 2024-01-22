from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import random, time, utils, datetime, csv
import pandas as pd
import pickle


def read_dataset():
    # base_vectors = utils.read_fvecs("../../dataset/siftsmall/siftsmall_base.fvecs")
    # query_vectors = utils.read_fvecs("../../dataset/siftsmall/siftsmall_query.fvecs")
    # knn_groundtruth = utils.read_ivecs("../../dataset/siftsmall/siftsmall_groundtruth.ivecs")
    base_vectors = utils.read_fvecs("../../dataset/sift/sift_base.fvecs")
    query_vectors = utils.read_fvecs("../../dataset/sift/sift_query.fvecs")
    knn_groundtruth = utils.read_ivecs("../../dataset/sift/sift_groundtruth.ivecs")
    return base_vectors, query_vectors, knn_groundtruth


baseV, queryV, _ = read_dataset()
base_vectors_with_attributes = pd.DataFrame({'vector': baseV.tolist()})
num_rows = len(base_vectors_with_attributes)
base_vectors_with_attributes['attr1'] = [random.choice([True, False]) for _ in range(num_rows)]
base_vectors_with_attributes['attr2'] = [random.choice([True, False]) for _ in range(num_rows)]
base_vectors_with_attributes['attr3'] = [random.choice([True, False]) for _ in range(num_rows)]

query_vectors_with_attributes = pd.DataFrame({'vector': queryV.tolist()})
num_rows = len(query_vectors_with_attributes)
query_vectors_with_attributes['attr1'] = [random.choice([True, False]) for _ in range(num_rows)]
query_vectors_with_attributes['attr2'] = [random.choice([True, False]) for _ in range(num_rows)]
query_vectors_with_attributes['attr3'] = [random.choice([True, False]) for _ in range(num_rows)]

truth = utils.top_k_neighbors(query_vectors_with_attributes, base_vectors_with_attributes)

with open('calculated_truth.pkl', 'wb') as file:
    pickle.dump(truth, file)