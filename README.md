## README File (`README.md`)

# GIST1M ANN Benchmark: FAISS vs. LanceDB

This repository contains benchmarking scripts comparing FAISS HNSW and LanceDB HNSW implementations on the GIST1M dataset. The benchmarks evaluate performance based on recall and queries per second (QPS), exploring how each system behaves when the dataset fits in memory and when it exceeds memory capacity.

## Introduction

High-dimensional nearest neighbor search (ANN) is crucial in various applications like recommendation systems, image retrieval, and natural language processing. This benchmark aims to compare the performance of:

- **FAISS HNSW**: An in-memory index.
- **LanceDB HNSW**: A disk-based index optimized for large-scale datasets.

The benchmarks demonstrate how LanceDB performs relative to FAISS, especially when dealing with datasets that exceed available memory.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher.
- pip package manager.
- Approximately 5 GB of free disk space for the dataset and indices.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/ann-benchmark.git
   cd ann-benchmark
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** If you encounter issues with `faiss-cpu`, ensure that you have a compatible version of Python and that your environment meets the build requirements.

## Running the Benchmark

Simply run the `benchmark.py` script:

```bash
python benchmark.py
```

The script will:

1. **Download the GIST1M Dataset**: If not already present in the `data/` directory.
2. **Compute Ground Truth Nearest Neighbors**: Using exact k-NN search with Scikit-learn.
3. **Build and Benchmark FAISS HNSW Index**: Over varying `efSearch` values.
4. **Build and Benchmark LanceDB HNSW Index**: Over varying `ef_search` values.
5. **Augment Data to Simulate Out-of-Memory Scenario**: Increase dataset size by 3x.
6. **Benchmark LanceDB with Augmented Data**: To analyze performance when data exceeds memory.
7. **Plot and Save Results**: As `benchmark_results.png`.


# License


This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
