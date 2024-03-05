import numpy as np
import h5py


def read_fvecs(file_path):
    # Source: http://corpus-texmex.irisa.fr/
    # in fvecs files, for each vector we have:
    #   1. 4 bytes int (32 bits) that represents the dimension of the vector
    #   2. n elements of 4 bytes floats (32 bits) * dimension. 
    # Then, each vector will have 1+dimension elements, but the first element will be 
    # the same for all of them and does not represent info about that record.

    # print(f'Loading file: {file_path.split("/")[-1]}')

    dimension = np.fromfile(file_path, dtype=np.int32, count=1)[0]
    # print(f'    The dimension of the vectors in the file is: {dimension}')

    # Bulk contains the raw data as 4 byte floats
    bulk = np.fromfile(file_path, dtype=np.float32)

    # Reshape it into dimension+1 groups
    reshaped_bulk = bulk.reshape(-1, dimension+1)

    # For each group, drop the first element (it is dimension which we already extracted)
    final_dataframe = reshaped_bulk[:, 1:]
    # print(f'    The final shape of the loaded dataset {file_path.split("/")[-1]} is {final_dataframe.shape}.')
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
    # print(f' Loading file: {file_path.split("/")[-1]}')
    bulk = np.fromfile(file_path, dtype=np.int32)
    dimension = bulk[0]
    # print(f'    The dimension of the vectors in the file is: {dimension}')

    reshaped_bulk = bulk.reshape(-1, dimension+1)

    final_dataframe = reshaped_bulk[:, 1:]
    # print(f'    The final shape of the loaded dataset is {final_dataframe.shape}.')
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
    elif similarity_function == 'dot':
        return np.dot(vector1, vector2)
    else:
        raise ValueError("Unsupported similarity function")

def select_k_closest_elements(df, reference_vector, k, similarity_function='euclidean'):
    df = df.copy() 
    df['distance'] = df['vector'].apply(lambda v: calculate_distance(v, reference_vector, similarity_function))
    if similarity_function == 'euclidean':
        result_df = df.nsmallest(k, 'distance').drop(columns=['distance'])
    else:
        result_df = df.nlargest(k, 'distance').drop(columns=['distance'])

        
    return result_df.index.values


def filter_by_attributes(qvec, bvecs):
    columns_to_match = bvecs.columns[1:]

    mask = (bvecs[columns_to_match] == qvec[columns_to_match]).all(axis=1)

    filtered_df = bvecs[mask]

    return filtered_df

def top_k_neighbors(query_vectors, base_vectors, k=100, function='euclidean', filtering = True):
    '''
        Calculates the top k neighbors (ground truth), if filtering = True, we filter all boolean attributes as well.
    '''
    if function not in ['euclidean', 'cosine', 'dot']:
        raise NotImplementedError("Other distance functions are not yet implemented")
    
    top_k_indices = []
    
    for _, elem in query_vectors.iterrows():
        if(filtering):
            filtered_df = filter_by_attributes(elem, base_vectors)
        else:
            filtered_df = base_vectors
             
        result = select_k_closest_elements(filtered_df, elem["vector"], k, similarity_function=function)
        top_k_indices.append(result)
    
    return top_k_indices

