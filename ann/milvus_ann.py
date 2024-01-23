from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
import numpy as np
import csv
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

# Read the SIFT datasets
base_vectors = list(fvecs_read('sift/sift_base.fvecs'))
query_vectors = list(fvecs_read('sift/sift_query.fvecs'))
training_vectors = list(fvecs_read('sift/sift_learn.fvecs'))
ground_truth = list(ivecs_read('sift/sift_groundtruth.ivecs'))

vector_ids = list(range(len(base_vectors)))

COLLECTION_NAME = "SIFT_EXPERIMENT_ANN"


m_values = [8, 16, 32, 64]
ef_construction_values = [64, 128, 256, 512]
limit_values = [1, 10, 100]

run_ivf = True

experiment_results = []

def run_experiment(run_ivf, m, ef, lim):
    experiment_data = {
        "experiment_type": "IVF" if run_ivf else "HNSW",
        "m": m,
        "ef": ef,
        "lim": lim
    }
    if not run_ivf:
        print("HNSW")
        print("m : ", m)
        print("ef : ", ef)
        print("lim : ", lim)
    else:
        print("IVF")
        print("nprobe : ", 128)
        print("nlist : ", 4096)
    # Define an index type and its parameters
    has = utility.has_collection(COLLECTION_NAME)
    if has:
        utility.drop_collection(COLLECTION_NAME)
    # print(f"Does collection SIFT10K_Collection exist in Milvus: {has}")


    # Define the collection schema
    dim = 128  # Dimension of SIFT vectors
    collection_name = COLLECTION_NAME
    field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
    field2 = FieldSchema(name="feature", dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[field1, field2], description="SIFT10K dataset collection")

    # print("Create collection SIFT10K")
    collection = Collection(COLLECTION_NAME, schema)

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
    ivf_flat_params = {
        "metric_type": "L2",   # or any other metric type suitable for your data
        "index_type": "IVF_FLAT",  # choose an index type
        "params": {"nlist": 4096}  # adjust parameters based on your dataset and needs
    }

    hnsw_params = {
        "metric_type": "L2",  # or any other metric type suitable for your data
        "index_type": "HNSW",  # using HNSW index
        "params": {
            "M": m,  # number of bi-directional links for each element, adjust as needed
            "efConstruction": ef  # size of the dynamic list for the nearest neighbors, adjust as needed
        }
    }

    index_start_time = time.time()
    # Create an index
    if run_ivf:
        collection.create_index(field_name="feature", index_params=ivf_flat_params)
    else:
        collection.create_index(field_name="feature", index_params=hnsw_params)
    index_end_time = time.time()
    indexing_time = index_end_time - index_start_time
    print(f"Indexing Time: {indexing_time:.2f} seconds")

    collection.load()

    # Perform search and calculate throughput
    
    if run_ivf:
        search_params = {
        "offset":0,
        "metric_type": "L2",
        "params": {
            "nprobe": 128
            },
        }
        run_ivf = False
    else:
        search_params = {
        "offset":0,
        "metric_type": "L2",
        }
    query_start_time = time.time()
    query_results = collection.search(query_vectors, "feature", search_params, limit=lim)
    end_time = time.time()

    queries_count = len(query_vectors)
    time_taken = end_time - query_start_time
    throughput = queries_count / time_taken
    average_latency = time_taken / queries_count
    print(f"Throughput: {throughput:.2f} queries per second")
    print(f"Average Latency: {average_latency:.4f} seconds")

    total_relevant = 0
    total_retrieved_relevant = 0

    for i in range(len(query_results)):
        query_result = query_results[i]
        ground_truth_ids = ground_truth[i][:lim]
        retrieved_ids = [hit.id for hit in query_result]

        relevant_retrieved = len(set(ground_truth_ids).intersection(retrieved_ids))
        # print(f"Query {i}: Retrieved = {len(retrieved_ids)}, Ground Truth = {len(ground_truth_ids)}, Recall@{R} = {relevant_retrieved / len(ground_truth_ids) if len(ground_truth_ids) > 0 else 0}")
        
        # print("GT : ", ground_truth_ids)
        # print("Retrieved :", retrieved_ids)
        total_retrieved_relevant += relevant_retrieved
        total_relevant += len(ground_truth_ids)

    recall = total_retrieved_relevant / total_relevant

    print(f"Average Recall: {recall:.4f}")

    experiment_data["indexing_time"] = indexing_time
    experiment_data["throughput"] = throughput
    experiment_data["average_latency"] = average_latency
    experiment_data["recall"] = recall

    # Add the experiment data to the list
    experiment_results.append(experiment_data)

    utility.drop_collection(COLLECTION_NAME)


run_experiment(run_ivf=True, m=None, ef=None, lim=1)
run_experiment(run_ivf=True, m=None, ef=None, lim=10)
run_experiment(run_ivf=True, m=None, ef=None, lim=100)
for m in m_values:
    for ef in ef_construction_values:
        for lim in limit_values:
            run_experiment(run_ivf=False, m=m, ef=ef, lim=lim)


def write_to_csv(file_name, data):
    keys = data[0].keys()
    with open(file_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

write_to_csv('experiment_results_ann_milvus.csv', experiment_results)
