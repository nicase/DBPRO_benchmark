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


def get_ground_truths(query_vectors, base_vectors, top_n=100):
    base_vectors = np.array(base_vectors)
    indices = []

    for query in query_vectors:
        query = np.array(query)
        # Compute the Euclidean distances
        distances = np.linalg.norm(base_vectors - query, axis=1)**2
        # Find the indices of the closest vectors
        closest_indices = np.argsort(distances)[:top_n]
        indices.append(closest_indices.tolist())

    return indices
    

def get_ground_truths_for_range_queries(query_vectors, base_vectors, radius, top_n):
    base_vectors = np.array(base_vectors)
    top_indices_within_radius = []

    for query in query_vectors:
        query = np.array(query)
        # Compute the Euclidean distances
        distances = np.linalg.norm(base_vectors - query, axis=1)**2
        # Get indices of the top N closest vectors
        top_n_indices = np.argsort(distances)[:top_n]
        # Filter out the indices that are outside the radius
        
        filtered_indices = [idx for idx in top_n_indices if distances[idx] <= radius]
        top_indices_within_radius.append(filtered_indices)
        

    return top_indices_within_radius



# Connect to Milvus server
client = connections.connect(host='localhost', port='19530')

COLLECTION_NAME = "SIFT_EXPERIMENT_RANGE"

has = utility.has_collection(COLLECTION_NAME)
print(f"Does collection SIFT10K_Collection exist in Milvus: {has}")


# Define the collection schema
dim = 128  # Dimension of SIFT vectors
collection_name = COLLECTION_NAME
field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
field2 = FieldSchema(name="feature", dtype=DataType.FLOAT_VECTOR, dim=dim)
schema = CollectionSchema(fields=[field1, field2], description="SIFT10K dataset collection")

print("Create collection SIFT10K")
collection = Collection(COLLECTION_NAME, schema)

# Read the SIFT datasets
base_vectors = list(fvecs_read('sift/sift_base.fvecs'))
query_vectors = list(fvecs_read('sift/sift_query.fvecs'))
training_vectors = list(fvecs_read('sift/sift_learn.fvecs'))
# ground_truth = list(ivecs_read('siftsmall/siftsmall_groundtruth.ivecs'))
RADIUS = 90000
# ground_truth = get_ground_truths(query_vectors, base_vectors, 100)
ground_truth = get_ground_truths_for_range_queries(query_vectors, base_vectors, RADIUS, 100)
for i in range(len(ground_truth)):
    print(len(ground_truth), len(ground_truth[i]))


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
    "params": {"nlist": 4096}  # adjust parameters based on your dataset and needs
}

hnsw_params = {
    "metric_type": "L2",  # or any other metric type suitable for your data
    "index_type": "HNSW",  # using HNSW index
    "params": {
        "M": 16,  # number of bi-directional links for each element, adjust as needed
        "efConstruction": 500  # size of the dynamic list for the nearest neighbors, adjust as needed
    }
}

index_start_time = time.time()
# Create an index
collection.create_index(field_name="feature", index_params=ivf_flat_params)
index_end_time = time.time()
indexing_time = index_end_time - index_start_time
print(f"Indexing Time: {indexing_time:.2f} seconds")

# Load the collection
collection.load()

# index_size = utility.get_index_size(collection_name, "feature")
#print(f"Index Size: {index_size} bytes")


# Perform search and calculate throughput
start_time = time.time()
search_params = {
    "offset":0,
    "metric_type": "L2",
    "params": {
        "nprobe": 128,
        # search for vectors with a distance smaller than 1.0
        "radius": RADIUS,
        # filter out vectors with a distance smaller than or equal to 0.8
        #"range_filter" : 0.8
        },
    }

# output_fields = ["dummy_attr_filtering"] 


query_start_time = time.time()
# query_results = collection.search([query_vector], "feature", search_params, limit=100, output_fields=output_fields, expr="dummy_attr_filtering < 3000")
query_results = collection.search(query_vectors, "feature", search_params, limit=100)
end_time = time.time()

queries_count = len(query_vectors)
time_taken = end_time - start_time
throughput = queries_count / time_taken
average_latency = time_taken / queries_count
print(f"Throughput: {throughput:.2f} queries per second")
print(f"Average Latency: {average_latency:.4f} seconds")

# Calculate recall@R
total_relevant = 0
total_retrieved_relevant = 0

for i in range(len(query_results)):
    query_result = query_results[i]
    # print(query_result)
    ground_truth_ids = ground_truth[i]
    retrieved_ids = [hit.id for hit in query_result]  # Consider only the top R results

    relevant_retrieved = len(set(ground_truth_ids).intersection(retrieved_ids))
    # print(f"Query {i}: Retrieved = {len(retrieved_ids)}, Ground Truth = {len(ground_truth_ids)}, Recall@{R} = {relevant_retrieved / len(ground_truth_ids) if len(ground_truth_ids) > 0 else 0}")
    
    # print("GT : ", ground_truth_ids)
    # print("Retrieved :", retrieved_ids)
    total_retrieved_relevant += relevant_retrieved
    total_relevant += len(ground_truth_ids)

recall = total_retrieved_relevant / total_relevant

print(f"Recall: {recall:.4f}")

utility.drop_collection(COLLECTION_NAME)
