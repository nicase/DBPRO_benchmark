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

with open('SIFT_BASEV_WITH_ATTRIBUTES.pkl', 'rb') as f:
        baseV = pickle.load(f)
with open('SIFT_QUERYV_WITH_ATTRIBUTES.pkl', 'rb') as f:
        queryV = pickle.load(f)

truth = utils.top_k_neighbors(queryV, baseV, function="cosine")

with open('AF_SIFT_COSINE_GT.pkl', 'wb') as file:
    pickle.dump(truth, file)