from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import random, time, utils, datetime, csv
import pandas as pd
import pickle, os
from dotenv import load_dotenv


def read_dataset():
    base_vectors = utils.read_fvecs(os.getenv('BASE_VECTORS_PATH'))
    query_vectors = utils.read_fvecs(os.getenv('QUERY_VECTORS_PATH'))
    knn_groundtruth = utils.read_ivecs(os.getenv('GROUND_TRUTH_PATH'))
    return base_vectors, query_vectors, knn_groundtruth

load_dotenv()
baseV, queryV, _ = read_dataset()
base_vectors = pd.DataFrame({'vector': baseV.tolist()})
num_rows = len(base_vectors)

query_vectors = pd.DataFrame({'vector': queryV.tolist()})
num_rows = len(query_vectors)

base_vectors.to_pickle('base_vectors_range.pkl')
query_vectors.to_pickle('query_vectors_range.pkl')

truth = utils.range_truth(query_vectors, base_vectors, threshold=600)

with open('calculated_truth_range.pkl', 'wb') as file:
    pickle.dump(truth, file)