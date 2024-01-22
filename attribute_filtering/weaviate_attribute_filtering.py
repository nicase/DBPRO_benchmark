import weaviate
client = weaviate.Client("http://localhost:8080")  # Replace the URL with that of your Weaviate instance
import utils
import pandas as pd
import numpy as np
import random
import uuid
import time
import csv

base_vectors = utils.read_fvecs("datasets\small\siftsmall_base.fvecs")
query_vectors = utils.read_fvecs("datasets\small\siftsmall_query.fvecs")
knn_groundtruth = utils.read_ivecs("datasets\small\siftsmall_groundtruth.ivecs")

base_vectors_with_attributes = pd.DataFrame({'vector': base_vectors.tolist()})
query_vectors_with_attributes = pd.DataFrame({'vector': query_vectors.tolist()})

num_rows = len(query_vectors_with_attributes)
query_vectors_with_attributes['attr1'] = [random.choice([True, False]) for _ in range(num_rows)]
query_vectors_with_attributes['attr2'] = [random.choice([True, False]) for _ in range(num_rows)]
query_vectors_with_attributes['attr3'] = [random.choice([True, False]) for _ in range(num_rows)]

def generate_uuid():
    return str(uuid.uuid4())

num_rows = len(base_vectors_with_attributes)
base_vectors_with_attributes['attr1'] = [random.choice([True, False]) for _ in range(num_rows)]
base_vectors_with_attributes['attr2'] = [random.choice([True, False]) for _ in range(num_rows)]
base_vectors_with_attributes['attr3'] = [random.choice([True, False]) for _ in range(num_rows)]
base_vectors_with_attributes['uuid'] = base_vectors_with_attributes.apply(lambda row: generate_uuid(), axis=1)


def upload_data(class_name):
    data_objs = [
        {"attr1": attr1, "attr2": attr2, "attr3": attr3}
        for attr1, attr2, attr3 in zip(base_vectors_with_attributes["attr1"].tolist(), base_vectors_with_attributes["attr2"].tolist(), base_vectors_with_attributes["attr3"].tolist())
    ]
    vectors = base_vectors_with_attributes["vector"].tolist()
    uuids_in_dataframe = base_vectors_with_attributes["uuid"].tolist()

    data_objs = [
        {"attr1": attr1, "attr2": attr2, "attr3": attr3}
        for attr1, attr2, attr3 in zip(base_vectors_with_attributes["attr1"].tolist(), base_vectors_with_attributes["attr2"].tolist(), base_vectors_with_attributes["attr3"].tolist())
    ]
    uuids_in_dataframe = base_vectors_with_attributes["uuid"].tolist()

    vectors = base_vectors_with_attributes["vector"].tolist()
    client.batch.configure(batch_size=100)  # Configure batch
    with client.batch as batch:
        for i, data_obj in enumerate(data_objs):
                batch.add_data_object(
                data_obj,
                class_name,
                vector=vectors[i],
                uuid= uuids_in_dataframe[i]
            )

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
    },
     "properties": [{
        "name": "attr1",
        "dataType": ["boolean"]},{
        "name": "attr2",
        "dataType": ["boolean"]},{
        "name": "attr3",
        "dataType": ["boolean"]}, 
        ]
}

# Add the class to the schema
client.schema.create_class(class_obj)
upload_data(class_name)

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
        "properties": [{
                "name": "attr1",
                "dataType": ["boolean"]},{
                "name": "attr2",
                "dataType": ["boolean"]},{
                "name": "attr3",
                "dataType": ["boolean"]}, 
                ]
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

# truth contains the IDs of the k nearest neighbors that also satisfy the attribute filtering clause
base = base_vectors_with_attributes.drop("uuid", axis=1)
truth = utils.top_k_neighbors(query_vectors_with_attributes, base, k=100, function='euclidean', filtering=True)

def run_query(name, k, query_vec, base_vec):

    result = []
    start_time = time.time()
    for _,elem in query_vec.iterrows():
        
        vec = elem["vector"]
        attr1 = elem["attr1"]
        attr2 = elem["attr2"]
        attr3 = elem["attr3"]

        response = (
            client.query
            .get(name)
            .with_near_vector({
            "vector": vec
            })
            .with_limit(k)
            .with_additional(["id"])
            .with_where({
            "operator": "And",
            "operands": [
            {
                "path": ["attr1"],
                "operator": "Equal",
                "valueBoolean": attr1
            },
            {
                "path": ["attr2"],
                "operator": "Equal",
                "valueBoolean": attr2
            },
            {
                "path": ["attr3"],
                "operator": "Equal",
                "valueBoolean": attr3
            }
            ]
            })  
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
    print(f"Throughput: of {name} {throughput:.2f} queries per second")
    print(f"Average Latency: of {name} {average_latency:.4f} seconds")
    print(f"k is {k}")


    true_positives = 0
    n_classified = 0
    for i,elem in enumerate(result_indexes):
        true_positives_iter = len(np.intersect1d(truth[i], elem))
        true_positives += true_positives_iter
        n_classified += len(elem)
    print(f'Average recall: of {name} {true_positives/n_classified}')
    
    with open("results_attribute_weviate.csv", mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([name, k, (true_positives/n_classified), throughput, average_latency])
    # sum = 0
    # for i,j in enumerate(result_indexes):
    #     intersection = np.intersect1d(truth[i], j)
    #     sum = sum + len(intersection)
    # print(f'The average accuracy is {sum/len(result_indexes)}')

k_number = [1, 10, 100]
for i in names:
    for j in range(3):
        run_query(i, k_number[j], query_vectors_with_attributes, base_vectors_with_attributes)

run_query("Siftsmall_flat_index" , 100, query_vectors_with_attributes, base_vectors_with_attributes)
# Specify the file name
csv_file_name = "results_attribute_weviate.csv"
# Define the column names
columns = ["name", "k", "throughput", "latency", "average_recall"]
# Create an empty CSV file with header
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header to the CSV file
    writer.writerow(columns)

print(f"Empty CSV file '{csv_file_name}' with columns {columns} has been created.")

