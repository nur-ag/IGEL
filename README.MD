# IGEL: Beyond 1-WL with Local Ego-Network Encodings

## Brief Description

This repository contains the code and experiments for 
IGEL, a graph encoding method for unattributed graphs that relies only on local ego-network information.

We include our IGEL implementation, as well as the boilerplate code to run the different 
experiments in a reproducible manner. We provide a quick 'drag and drop' IGEL BFS implementation (without log-1p 
normalization, mapping (distance, degree) pairs to sparse vector indices, relative degrees, and assuming the
[python-igraph](https://igraph.readthedocs.io/en/0.10.2/index.html) library) below:

```python
import igraph as ig
from collections import Counter


def igel_bfs(input_node, alpha=2):
    if alpha == 0:
        return {(0, input_node.degree()): 1}
    to_visit = [input_node]
    degrees = {}
    distances = {input_node: 0}
    while to_visit:
        node = to_visit[0]
        del to_visit[0]
        current_distance = distances[node]
        current_degree = 0
        for neigh in node.neighbors():
            if neigh not in distances:
                distances[neigh] = current_distance + 1
            if distances[neigh] <= alpha:
                current_degree += 1
                if neigh not in degrees:
                    to_visit.append(neigh)
        degrees[node] = current_degree
    counts = Counter((distances[node], degree) for node, degree in degrees.items())
    return {igel_pair: count for igel_pair, count in sorted(counts.most_common())}


G = ig.Graph.Tree(28, 3)
print(igel_bfs(G.vs[0]))
# prints: {(0, 3): 1, (1, 4): 3, (2, 1): 9}
```

## Replicating the Results

**Note**: The following applies to Link Prediction, Vertex Classification, Clustering and 
Scalability results. For the GNN results, see the [companion 
repository](https://github.com/nur-ag/gnn-matlang) based on [the repo for "Breaking the Limits 
of Message Passing Graph Neural Networks" published in ICML 2021](https://github.com/balcilar/gnn-matlang):

All runs are seeded and all training configurations are tracked my means of json files. Replicating the results should be a matter of:

1. **Installing the dependencies.**

```bash
$ python -m pip install -r requirements.txt
```

It might be advisable to create a [virtual environment](https://virtualenv.pypa.io/en/latest/) before doing this to avoid polluting your whole python environment. In principle, all dependencies to reproduce our results require calling `pip` on `requirements.txt`. 

2. **Downloading the datasets.**

In principle, all the data is included in the repo. The same caveat as in the previous point applies. 

3. **Running the code.**

```bash
$ ./replication.sh
```

The replication script should replicate the results sequentially and very slowly if you don't have a GPU. If you already have resulting outputs in the `output/` folder, the experiments will NOT run! Take this into consideration.

## Hardware Requirements

As per the paper and the last section, it is recommended that you run the experiments using a GPU. A refresher: For Link Prediction and Node Classification, we run each experiment on a machine with 8 cores, 16GB of memory and a single GPU with 8GB of device memory. For graph analysis, we simply use the provided notebook in the notebook folder, running on a 2015 MacBook Pro with 8 cores and 16GB of memory. Finally, the scalability analysis is ran on a CPU-only setting in a 48 cpu SLURM cluster reserving 200GB of memory per run. We realize exactly reproducing the scalability analysis is not realistic, given the workload of a SLURM cluster. The other experiments, however, are seeded and should be easy to replicate with common 'researcher-with-a-laptop' hardware.

## Author

This project is maintained by Nurudín Álvarez. You can get in touch at https://nur.wtf/

