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
    base_vectors, query_vectors, _ = utils.read_h5vs(os.getenv('H5PY_GLOVE_PATH'))
    return base_vectors, query_vectors, _

baseV, queryV, _ = read_dataset()

# UNCOMMENT TO TEST (10K BASE VECS, 100 QUERY VECS)
baseV = pd.DataFrame({'vector': baseV[:10000]})
queryV = pd.DataFrame({'vector': queryV[:100]})
# baseV = pd.DataFrame({'vector': baseV})
# queryV = pd.DataFrame({'vector': queryV})


truth = utils.top_k_neighbors(queryV, baseV, function="cosine")

with open('ANN_gloVe_COSINE_GT.pkl', 'wb') as file:
    pickle.dump(truth, file)
