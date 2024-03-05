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
    print("SIFT dot")

    if environment == "test":
        print("Running in test env (10K base vectors, 100 query vectors, 100 GT)")
        baseV, queryV, _ = read_dataset(True)

    elif environment == "prod":
        # Run in production environment
        print("Running in production env (1M base vectors, 10k query vectors, 10k GT)")
        baseV, queryV, _ = read_dataset(False)

    else:
        # Handle invalid environment
        print("Invalid environment:", environment, "\nExiting...")
        sys.exit(1)
else:
    # Handle case when no environment parameter is passed
    print("No environment parameter passed. \nExiting")
    sys.exit(1)

baseV = pd.DataFrame({'vector': baseV.tolist()})
queryV = pd.DataFrame({'vector': queryV.tolist()})

# UNCOMMENT TO TEST (10K BASE VECS, 100 QUERY VECS)
# baseV = pd.DataFrame({'vector': baseV[:10000]})
# queryV = pd.DataFrame({'vector': queryV[:100]})


truth = utils.top_k_neighbors(queryV, baseV, function="dot")
ground_truth_folder = os.getenv('ANN_GROUND_TRUTH_DIR')
with open(f'{ground_truth_folder}ANN_SIFT_DOT_GT.pkl', 'wb') as file:
    pickle.dump(truth, file)