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

load_dotenv()
ground_truth_folder = os.getenv('AF_GROUND_TRUTH_DIR')
with open(f'{ground_truth_folder}GLOVE_BASEV_WITH_ATTRIBUTES.pkl', 'rb') as f:
        baseV = pickle.load(f)
with open(f'{ground_truth_folder}GLOVE_QUERYV_WITH_ATTRIBUTES.pkl', 'rb') as f:
        queryV = pickle.load(f)

truth = utils.top_k_neighbors(queryV, baseV, function="cosine")

with open(f'{ground_truth_folder}AF_gloVe_DOT_GT.pkl', 'wb') as file:
    pickle.dump(truth, file)