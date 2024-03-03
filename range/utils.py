import numpy as np
import pandas as pd
import h5py

def read_fvecs(file_path):
    # Source: http://corpus-texmex.irisa.fr/
    # in fvecs files, for each vector we have:
    #   1. 4 bytes int (32 bits) that represents the dimension of the vector
    #   2. n elements of 4 bytes floats (32 bits) * dimension. 
    # Then, each vector will have 1+dimension elements, but the first element will be 
    # the same for all of them and does not represent info about that record.

    print(f'Loading file: {file_path.split("/")[-1]}')

    dimension = np.fromfile(file_path, dtype=np.int32, count=1)[0]
    print(f'    The dimension of the vectors in the file is: {dimension}')

    # Bulk contains the raw data as 4 byte floats
    bulk = np.fromfile(file_path, dtype=np.float32)

    # Reshape it into dimension+1 groups
    reshaped_bulk = bulk.reshape(-1, dimension+1)

    # For each group, drop the first element (it is dimension which we already extracted)
    final_dataframe = reshaped_bulk[:, 1:]
    print(f'    The final shape of the loaded dataset {file_path.split("/")[-1]} is {final_dataframe.shape}.')
    return final_dataframe



def read_ivecs(file_path):
    # Source: http://corpus-texmex.irisa.fr/
    # (From source):
    # The groundtruth files contain, for each query, the identifiers (vector number, starting at 0) 
    # of its k nearest neighbors, ordered by increasing (squared euclidean) distance. 
    #   • k=100 for the dataset ANN_SIFT10K, ANN_SIFT1M and ANN_GIST1M
    #   • k=1000 for the big ANN_SIFT1B dataset
    # Therefore, the first element of each integer vector is the nearest neighbor identifier 
    # associated with the query. 

    # ivecs files are identical to fvecs files, but contain 4 byte int values instead. Check read_fvecs function to know how it works
    # only difference is that since its all ints, we don't need to load the first element differently
    print(f' Loading file: {file_path.split("/")[-1]}')
    bulk = np.fromfile(file_path, dtype=np.int32)
    dimension = bulk[0]
    print(f'    The dimension of the vectors in the file is: {dimension}')

    reshaped_bulk = bulk.reshape(-1, dimension+1)

    final_dataframe = reshaped_bulk[:, 1:]
    print(f'    The final shape of the loaded dataset is {final_dataframe.shape}.')
    return final_dataframe

def read_h5vs(file_path):
    with h5py.File(file_path, 'r') as hdf_file:
        ground_truth = []
        query_vectors = []
        base_vectors = []
        for a in hdf_file['train']:
            base_vectors.append(list(a))
        for a in hdf_file['test']:
            query_vectors.append(list(a))
        for a in hdf_file['neighbors']:
            ground_truth.append(list(a))
    
    return base_vectors, query_vectors, ground_truth

def calculate_distance(vector1, vector2, similarity_function='euclidean'):
    if similarity_function == 'euclidean':
        return np.linalg.norm(np.array(vector1) - np.array(vector2), ord=2)
    elif similarity_function == 'cosine':
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    else:
        raise ValueError("Unsupported similarity function")

def range_truth(query_vectors, base_vectors, threshold = 300, function='euclidean'):

    distances_df = pd.DataFrame(index=query_vectors.index, columns=base_vectors.index)


    for i, vector1 in enumerate(query_vectors['vector']):
        for j, vector2 in enumerate(base_vectors['vector']):
            distance = calculate_distance(vector1, vector2, similarity_function='euclidean')
            distances_df.at[i, j] = distance

    mask = distances_df < threshold
    ids = np.empty((len(query_vectors),), dtype=object)
    for i in range(len(query_vectors)):
        ids[i] = np.where(mask.iloc[i])[0]


    return ids

def range_truth_iterative(query_vector, base_vectors, threshold=300, similarity_function='euclidean'):
    ids_list = []

    distances = []
    for base_vector in base_vectors['vector']:
        distance = calculate_distance(query_vector, base_vector, similarity_function)
        distances.append(distance)
    distances = np.array(distances)
    ids = np.where(distances < threshold)[0]
    ids_list.append(ids)

    return ids_list
