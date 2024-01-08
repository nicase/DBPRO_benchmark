from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
import numpy as np
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)


def bvecs_read(fname):
    a = np.fromfile(fname, dtype=np.int32, count=1)
    b = np.fromfile(fname, dtype=np.uint8)
    d = a[0]
    return b.reshape(-1, d + 4)[:, 4:].copy()


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


# Connect to Milvus server
client = connections.connect(host='localhost', port='19530')

has = utility.has_collection("SIFT10K_Collection")
print(f"Does collection SIFT10K_Collection exist in Milvus: {has}")

# Define the collection schema
dim = 128  # Dimension of SIFT vectors
collection_name = "SIFT10K_Collection"
field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
field2 = FieldSchema(name="feature", dtype=DataType.FLOAT_VECTOR, dim=dim)
schema = CollectionSchema(fields=[field1, field2], description="SIFT10K dataset collection")

print("Create collection SIFT10K")
collection = Collection("SIFT10K_Collection", schema)

# Read the SIFT datasets
base_vectors = list(fvecs_read('sift/sift_base.fvecs'))
query_vectors = list(fvecs_read('sift/sift_query.fvecs'))
training_vectors = list(fvecs_read('sift/sift_learn.fvecs'))
ground_truth = list(ivecs_read('sift/sift_groundtruth.ivecs'))

vector_ids = list(range(len(base_vectors)))

# Insert the base vectors into the collection

def insert_in_batches(collection, ids, vectors, batch_size=10000):
    total = len(vectors)
    for i in range(0, total, batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        mr = collection.insert([batch_ids, batch_vectors])
        print(f"Inserted batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

# Insert data
insert_in_batches(collection, vector_ids, base_vectors)

# Define an index type and its parameters
ivf_flat_params = {
    "metric_type": "L2",   # or any other metric type suitable for your data
    "index_type": "IVF_FLAT",  # choose an index type
    "params": {"nlist": 16384}  # adjust parameters based on your dataset and needs
}

hnsw_params = {
    "metric_type": "L2",  # or any other metric type suitable for your data
    "index_type": "HNSW",  # using HNSW index
    "params": {
        "M": 16,  # number of bi-directional links for each element, adjust as needed
        "efConstruction": 500  # size of the dynamic list for the nearest neighbors, adjust as needed
    }
}

# Create an index
collection.create_index(field_name="feature", index_params=ivf_flat_params)

# Load the collection
collection.load()

# Perform search and calculate throughput
start_time = time.time()
search_params = {"offset":0, "metric_type": "L2", "params": {"nprobe": 100}}
query_results = collection.search(query_vectors, "feature", search_params, limit=100)
print(query_results[0].distances)
end_time = time.time()

queries_count = len(query_vectors)
print(queries_count)
time_taken = end_time - start_time
throughput = queries_count / time_taken  # Queries per second


R = 100  # Set this to your desired rank

# Calculate recall@R
total_relevant = 0
total_retrieved_relevant = 0

for i, query_result in enumerate(query_results):
    ground_truth_ids = ground_truth[i]
    retrieved_ids = [hit.id for hit in query_result]  # Consider only the top R results

    relevant_retrieved = len(set(ground_truth_ids).intersection(retrieved_ids))
    print(f"Query {i}: Retrieved = {len(retrieved_ids)}, Ground Truth = {len(ground_truth_ids)}, Recall@{R} = {relevant_retrieved / len(ground_truth_ids) if len(ground_truth_ids) > 0 else 0}")
    
    print("GT : ", ground_truth_ids)
    print("Retrieved :", retrieved_ids)
    total_retrieved_relevant += relevant_retrieved
    total_relevant += len(ground_truth_ids)

recall_at_R = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
print(f"Total Recall@{R}: {recall_at_R}")

recall = total_retrieved_relevant / total_relevant

# Print the performance metrics
print(f"Throughput: {throughput:.2f} queries per second")
print(f"Recall: {recall:.4f}")
utility.drop_collection("SIFT10K_Collection")