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
baseV = pd.DataFrame({'vector': baseV.tolist()})
queryV = pd.DataFrame({'vector': queryV.tolist()})

# UNCOMMENT TO TEST (10K BASE VECS, 100 QUERY VECS)
# baseV = pd.DataFrame({'vector': baseV[:10000]})
# queryV = pd.DataFrame({'vector': queryV[:100]})


truth = utils.top_k_neighbors(queryV, baseV, function="dot")

with open('ANN_SIFT_DOT_GT.pkl', 'wb') as file:
    pickle.dump(truth, file)