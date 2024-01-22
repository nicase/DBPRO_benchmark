import weaviate
client = weaviate.Client("172.21.0.2:8080")  # Replace the URL with that of your Weaviate instance
import utils
import pandas as pd
import numpy as np
import uuid
import time
import csv

base_vectors = pd.DataFrame({'vector': utils.read_fvecs("/data-ssd/dimitrios/DBPRO_benchmark/data/sift/sift_base.fvecs").tolist()})
query_vectors = pd.DataFrame({'vector': utils.read_fvecs("/data-ssd/dimitrios/DBPRO_benchmark/data/sift/sift_query.fvecs").tolist()})


def generate_uuid():
    return str(uuid.uuid4())

base_vectors['uuid'] = base_vectors.apply(lambda row: generate_uuid(), axis=1)

print('uuid generated')

def upload_data(class_name):
    vectors = base_vectors["vector"].tolist()
    uuids_in_dataframe = base_vectors["uuid"].tolist()
    data_objs = [
        {"title": f"Object {i+1}"} for i in range(len(base_vectors))
    ]
    uuids_in_dataframe = base_vectors["uuid"].tolist()
    vectors = base_vectors["vector"].tolist()
    client.batch.configure(batch_size=100)  # Configure batch
    with client.batch as batch:
        for i, data_obj in enumerate(data_objs):
            batch.add_data_object(
                data_obj,
                class_name,
                vector=vectors[i],
                uuid= uuids_in_dataframe[i]
            )
#-------------------------------------------------------------------#
#with Flat index
class_name = "Siftsmall_flat_index" 
distance_metric = "l2-squared"
client.schema.delete_class(class_name)
# Class definition object. Weaviate's autoschema feature will infer properties when importing.
class_obj = {
    "class": class_name,
    "vectorizer": "none",
    "vectorIndexType": "flat",
    "vectorIndexConfig": {
    "distance": distance_metric
    }
}
# Add the class to the schema
client.schema.create_class(class_obj)
upload_data(class_name)
#-------------------------------------------------------------------#
def create_HSNW_collections(ef, maxConnections, efConstruction, to_name):
    class_name = to_name
    #with HSNW
    distance_metric = "l2-squared"
    client.schema.delete_class(class_name)
    # Class definition object. Weaviate's autoschema feature will infer properties when importing.
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig":{
            "skip": False,
            "cleanupIntervalSeconds": 300,
            "pq": {"enabled": False,},
            "maxConnections": maxConnections,
            "efConstruction": efConstruction,
            "ef": ef,
            "dynamicEfMin": 100,
            "dynamicEfMax": 500,
            "dynamicEfFactor": 8,
            "vectorCacheMaxObjects": 2000000,
            "flatSearchCutoff": 40000,
            "distance": distance_metric},
    }

    # Add the class to the schema
    client.schema.create_class(class_obj)
    upload_data(class_name)

ef = [64, 128, 256, 512]
maxConnections = [8, 16, 32, 64]
efConstruction = [64, 128, 256, 512]
names = []

for i in ef:
    for j in maxConnections:
        for k in efConstruction:
            to_name = 'Siftsmall_HSNW_index' + '_' + str(i) + '_' + str(j) + '_' + str(k)
            names.append(to_name)
            print('appending', to_name)
            create_HSNW_collections(i,j,k, to_name)
            print('appended', to_name)
#-------------------------------------------------------------------#

base = base_vectors.drop("uuid", axis=1)
truth = utils.top_k_neighbors(query_vectors, base, k=100, function='euclidean', filtering=False)

# truth contains the IDs of the k nearest neighbors that also satisfy the attribute filtering clause
def run_query(name, k, query_vec, base_vec):

    result = []
    start_time = time.time()
    for _,elem in query_vec.iterrows():
        
        vec = elem["vector"]

        response = (
            client.query
            .get(name)
            .with_near_vector({
            "vector": vec
            })
            .with_limit(k)
            .with_additional(["id"])  
            .do()
        )
        result.append(response)
    end_time = time.time()
   
    result_ids = []
    for i in range(len(query_vec["vector"])):
        result_ids.append([])
        for j in range(k):
            result_ids[i].append(result[i]["data"]["Get"][name][j]["_additional"]["id"])

    result_indexes = []
    for i in result_ids:
        result_indexes.append(base_vec[base_vec['uuid'].isin(i)].index)

    time_taken = end_time - start_time
    queries_count = len(result)
    throughput = queries_count / time_taken
    average_latency = time_taken / queries_count
    print(f"Throughput: {throughput:.2f} queries per second")
    print(f"Average Latency: {average_latency:.4f} seconds")

    true_positives = 0
    n_classified = 0
    for i,elem in enumerate(result_indexes):
        true_positives_iter = len(np.intersect1d(truth[i], elem))
        true_positives += true_positives_iter
        n_classified += len(elem)
    print(f'Average recall: {true_positives/n_classified}')

    with open("results_ann_weviate.csv", mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([name, k, (true_positives/n_classified), throughput, average_latency])

# Specify the file name
csv_file_name = "results_ann_weviate.csv"
# Define the column names
columns = ["name", "k",  "average_recall" ,"throughput", "latency"]
# Create an empty CSV file with header
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header to the CSV file
    writer.writerow(columns)
print(f"Empty CSV file '{csv_file_name}' with columns {columns} has been created.")

k_number = [1, 10, 100]
for i in names:
    for j in range(3):
        run_query(i, k_number[j], query_vectors, base_vectors)

run_query("Siftsmall_flat_index", 100, query_vectors, base_vectors)

