import os
import tarfile
import numpy as np
import struct
import urllib.request
import time
import faiss
from sklearn.neighbors import NearestNeighbors
import lancedb
from lance.vector import vec_to_table
import pyarrow as pa
import matplotlib.pyplot as plt

# Step 1: Load the test and train data

def download(url, dest):
    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))
    if not os.path.exists(dest):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"File {dest} already exists.")

def _load_texmex_vectors(f, n, k):
    v = np.zeros((n, k), dtype='float32')
    for i in range(n):
        f.read(4)  # length info, not used
        v[i] = struct.unpack('f' * k, f.read(k * 4))
    return v

def _get_irisa_matrix(t, fn):
    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack('i', f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)

def load_gist():
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.gz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        X_train = _get_irisa_matrix(t, "gist/gist_base.fvecs")
        X_test = _get_irisa_matrix(t, "gist/gist_query.fvecs")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    return X_train, X_test

# Compute ground truth
def compute_ground_truth(X_train, X_test, k):
    print("Computing ground truth nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=-1).fit(X_train)
    distances, indices = nbrs.kneighbors(X_test)
    ground_truth = indices.tolist()
    print("Ground truth computation completed.")
    return ground_truth

# Step 2: Create a FAISS HNSW index and run the benchmark

def build_faiss_index(X_train, M_hnsw=32, efConstruction=100):
    d = X_train.shape[1]
    index = faiss.IndexHNSWFlat(d, M_hnsw)
    index.hnsw.efConstruction = efConstruction
    print(f"\nBuilding Faiss index with M_hnsw={M_hnsw}, efConstruction={efConstruction}")
    index.add(X_train)
    return index

def benchmark_faiss_queries(index, X_test, ground_truth, k=10, efSearch=50):
    total_queries = len(X_test)
    total_recall = 0.0

    print(f"\nStarting benchmarking with Faiss (efSearch={efSearch})...")
    index.hnsw.efSearch = efSearch  # Set efSearch parameter
    start_time = time.time()

    for idx, query_vector in enumerate(X_test):
        if (idx + 1) % 100 == 0:
            print(f"Processing query {idx + 1}/{len(X_test)}...")
        D, I = index.search(query_vector.reshape(1, -1), k)
        neighbors = I[0].tolist()
        # Compute recall
        gt_neighbors = set(ground_truth[idx])
        retrieved_neighbors = set(neighbors)
        intersection_count = len(gt_neighbors & retrieved_neighbors)
        recall = intersection_count / len(gt_neighbors)
        total_recall += recall

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_query = total_time / total_queries
    avg_recall = total_recall / total_queries
    QPS = total_queries / total_time

    print(f"Average search time per query: {avg_time_per_query:.6f} seconds.")
    print(f"Average recall per query: {avg_recall:.4f}")

    results = {
        "average_search_time_per_query_seconds": avg_time_per_query,
        "average_recall": avg_recall,
        "total_queries": total_queries,
        "neighbors_retrieved_per_query": k,
        "total_time_seconds": total_time,
        "queries_per_second": QPS,
        "efSearch": efSearch
    }
    return results

# Step 3: Create a LanceDB index with all the training data and run the benchmark

def initialize_lancedb(uri='data/sample-lancedb'):
    print("Connecting to LanceDB...")
    if not os.path.exists(uri):
        os.makedirs(uri)
    catalog = lancedb.connect(uri)
    return catalog

def create_lancedb_table(catalog, table_name, vectors, index_params):
    print(f"\nCreating table '{table_name}'...")
    table = vec_to_table(vectors)
    num_rows = table.num_rows
    id_data = pa.array(range(num_rows))
    table = table.append_column('id', id_data)

    if table_name in [tbl.name for tbl in catalog.tables()]:
        catalog.drop_table(table_name)

    tbl = catalog.create_table(table_name, data=table)

    print("Creating index...")
    tbl.create_index(
        column="vector",
        **index_params
    )
    return tbl

def benchmark_lancedb_queries(tbl, X_test, ground_truth, k=10, query_args=None):
    total_queries = len(X_test)
    total_recall = 0.0

    print(f"\nStarting benchmarking with LanceDB (query_args={query_args})...")
    start_time = time.time()
    for idx, query_vector in enumerate(X_test):
        if (idx + 1) % 100 == 0:
            print(f"Processing query {idx + 1}/{len(X_test)}...")
        # Perform the query
        result = tbl.search(query_vector, metric='L2', **(query_args or {})).limit(k).to_df()
        neighbors = result['id'].values.tolist()
        # Compute recall
        gt_neighbors = set(ground_truth[idx])
        retrieved_neighbors = set(neighbors)
        intersection_count = len(gt_neighbors & retrieved_neighbors)
        recall = intersection_count / len(gt_neighbors)
        total_recall += recall

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_query = total_time / total_queries
    avg_recall = total_recall / total_queries
    QPS = total_queries / total_time

    print(f"Average search time per query: {avg_time_per_query:.6f} seconds.")
    print(f"Average recall per query: {avg_recall:.4f}")

    results = {
        "average_search_time_per_query_seconds": avg_time_per_query,
        "average_recall": avg_recall,
        "total_queries": total_queries,
        "neighbors_retrieved_per_query": k,
        "total_time_seconds": total_time,
        "queries_per_second": QPS,
        "query_args": query_args
    }
    return results

# Step 4: Data augmentation, train LanceDB index, and run benchmark

def augment_data(X_train, factor=3):
    np.random.seed(42)  # For reproducibility
    augmented_data = [X_train]
    for _ in range(factor - 1):
        noise = np.random.normal(0, 0.01, X_train.shape).astype('float32')
        augmented = X_train + noise
        augmented_data.append(augmented)
    augmented_data = np.vstack(augmented_data)
    return augmented_data

def main():
    # Load data
    X_train, X_test = load_gist()

    # Compute ground truth
    k = 10
    ground_truth = compute_ground_truth(X_train, X_test, k)

    # Build FAISS index
    index = build_faiss_index(X_train, M_hnsw=32, efConstruction=100)

    # Benchmark FAISS with varying efSearch values
    faiss_results_list = []
    efSearch_values = [10, 20, 40, 80, 120, 200, 400, 800]
    for ef in efSearch_values:
        res = benchmark_faiss_queries(index, X_test, ground_truth, k=10, efSearch=ef)
        faiss_results_list.append(res)

    # Initialize LanceDB
    catalog = initialize_lancedb(uri="data/sample-lancedb")

    # Index parameters for LanceDB
    index_params = {
        "index_type": "HNSW",
        "metric": "L2",
        "m": 32,
        "ef_construction": 100
    }

    # Create LanceDB table
    table = create_lancedb_table(catalog, "table", X_train, index_params)

    # Benchmark LanceDB with varying ef_search values
    lancedb_results_list = []
    ef_values = [10, 20, 40, 80, 120, 200, 400, 800]
    for ef in ef_values:
        query_args = {'ef_search': ef}
        res = benchmark_lancedb_queries(table, X_test, ground_truth, k=10, query_args=query_args)
        res['ef_search'] = ef
        lancedb_results_list.append(res)

    # Augment data to 3x
    X_train_augmented = augment_data(X_train, factor=3)
    print(f"\nAugmented X_train shape: {X_train_augmented.shape}")

    # Create LanceDB table with augmented data
    table_augmented = create_lancedb_table(catalog, "table_augmented", X_train_augmented, index_params)

    # Benchmark LanceDB with augmented data
    lancedb_augmented_results_list = []
    for ef in ef_values:
        query_args = {'ef_search': ef}
        res = benchmark_lancedb_queries(table_augmented, X_test, ground_truth, k=10, query_args=query_args)
        res['ef_search'] = ef
        lancedb_augmented_results_list.append(res)

    # Collect data for plotting
    faiss_recalls = [res['average_recall'] for res in faiss_results_list]
    faiss_qps = [res['queries_per_second'] for res in faiss_results_list]

    lancedb_recalls = [res['average_recall'] for res in lancedb_results_list]
    lancedb_qps = [res['queries_per_second'] for res in lancedb_results_list]

    lancedb_aug_recalls = [res['average_recall'] for res in lancedb_augmented_results_list]
    lancedb_aug_qps = [res['queries_per_second'] for res in lancedb_augmented_results_list]

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(faiss_recalls, faiss_qps, marker='o', label='Faiss (1M vectors)')
    plt.plot(lancedb_recalls, lancedb_qps, marker='s', label='LanceDB (1M vectors)')
    plt.plot(lancedb_aug_recalls, lancedb_aug_qps, marker='^', label='LanceDB Augmented (3M vectors)')

    plt.xlabel('Recall')
    plt.ylabel('Queries per Second (QPS)')
    plt.title('Recall vs QPS Tradeoff Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('benchmark_results.png')
    plt.show()

if __name__ == "__main__":
    main()
