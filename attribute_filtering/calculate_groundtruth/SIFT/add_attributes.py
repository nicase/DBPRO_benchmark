from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import random, time, utils, datetime, csv
import pandas as pd
import pickle
import os
from dotenv import load_dotenv


def read_dataset():
    load_dotenv()
    base_vectors = utils.read_fvecs(os.getenv('BASE_VECTORS_PATH'))
    query_vectors = utils.read_fvecs(os.getenv('QUERY_VECTORS_PATH'))
    knn_groundtruth = utils.read_ivecs(os.getenv('GROUND_TRUTH_PATH'))
    return base_vectors, query_vectors, knn_groundtruth

baseV, queryV, _ = read_dataset()

# UNCOMMENT TO TEST (10K BASE VECS, 100 QUERY VECS)
# baseV = baseV[:10000]
# queryV = queryV[:100]

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

base_vectors_with_attributes.to_pickle('SIFT_BASEV_WITH_ATTRIBUTES.pkl')
query_vectors_with_attributes.to_pickle('SIFT_QUERYV_WITH_ATTRIBUTES.pkl')
