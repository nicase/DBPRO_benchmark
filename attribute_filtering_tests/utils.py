import numpy as np

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

def euclidean_distance(query_vector, base_vectors):
    distances = np.sqrt(np.sum((base_vectors - query_vector) ** 2, axis=1))
    return distances


def top_k_neighbors(query_vectors, base_vectors, k=100, function = 'euclidean'):
    '''
        Calculates the top k neighbors (ground truth). For now only with euclidean distance. 
        TODO: add other functions (cosine similarity, etc).
    '''
    top_k_indices = []
    for query_vector in query_vectors:
        distances = euclidean_distance(query_vector, base_vectors)
        
        k_indices = np.argsort(distances)[:k]
        top_k_indices.append(k_indices)
    
    return np.array(top_k_indices)