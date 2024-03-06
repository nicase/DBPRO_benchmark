from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import numpy as np
import random, time, utils, datetime, csv, pickle, os, sys
import pandas as pd
from dotenv import load_dotenv


def read_dataset(test):
    load_dotenv()
    # print(os.getenv('BASE_VECTORS_PATH_TEST'))
    if test:
        base_vectors = utils.read_fvecs(os.getenv('BASE_VECTORS_PATH_TEST'))
        query_vectors = utils.read_fvecs(os.getenv('QUERY_VECTORS_PATH_TEST'))
        knn_groundtruth = utils.read_ivecs(os.getenv('GROUND_TRUTH_PATH_TEST'))
    else:
        base_vectors = utils.read_fvecs(os.getenv('BASE_VECTORS_PATH'))
        query_vectors = utils.read_fvecs(os.getenv('QUERY_VECTORS_PATH'))
        knn_groundtruth = utils.read_ivecs(os.getenv('GROUND_TRUTH_PATH'))

    return base_vectors, query_vectors, knn_groundtruth

if len(sys.argv) > 1:
    environment = sys.argv[1]
    print("Adding attributes to SIFT...")

    if environment == "test":
        print("Running in test env (10K base vectors, 100 query vectors, 100 GT)")
        baseV, queryV, _ = read_dataset(True)

    elif environment == "prod":
        # Run in production environment
        print("Running in production env (1M base vectors, 10k query vectors, 10k GT)")
        baseV, queryV, _ = read_dataset(False)
        baseV = baseV.tolist()[:500000]
        queryV = queryV.tolist()[:5000]

    else:
        # Handle invalid environment
        print("Invalid environment:", environment, "\nExiting...")
        sys.exit(1)
else:
    # Handle case when no environment parameter is passed
    print("No environment parameter passed. \nExiting")
    sys.exit(1)

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

ground_truth_folder = os.getenv('AF_GROUND_TRUTH_DIR')
base_vectors_with_attributes.to_pickle(f'{ground_truth_folder}SIFT_BASEV_WITH_ATTRIBUTES.pkl')
query_vectors_with_attributes.to_pickle(f'{ground_truth_folder}SIFT_QUERYV_WITH_ATTRIBUTES.pkl')
