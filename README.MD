# IGEL: Inductive Graph Encodings of Local Information

## Brief Description

This repository contains the code and experiments for 
IGEL, a graph encoding method for unattributed graphs that relies only on locality information.

We include PyTorch layer implementations, as well as the boilerplate code to run the different 
experiments in a reproducible manner. 

## Replicating the Results

**Note**: The following applies to Link Prediction, Vertex Classification, Clustering and 
Scalability results. For the GNN results, see the [companion 
repository](https://github.com/nur-ag/gnn-matlang) based on [the repo for "Breaking the Limits 
of Message Passing Graph Neural Networks" published in ICML 2021](https://github.com/balcilar/gnn-matlang):

We have tried to keep the project fully replicable from the first moment. All runs are seeded and all training configurations are tracked my means of json files. However, this code is messy, has no unit testing in place, and might make your computer explode. Special researcher gloves are recommended as protection, should your device catch on fire. If you are properly equipped, replicating the results should be a matter of:

1. **Installing the dependencies.**

```bash
$ python -m pip install -r requirements.txt
```

It might be advisable to create a [virtual environment](https://virtualenv.pypa.io/en/latest/) before doing this to avoid polluting your whole python environment.

In principle, all dependencies as they are found in the author's computer should be installed by calling `pip` on `requirements.txt`. In practice, this might not be the case, so please open an issue if you get stuck so we figure out what the problem is. The code has been tested on a single Mac OS laptop and two different Linux servers. This should be better than 'works in my machine', but perhaps not.

2. **Downloading the datasets.**

In principle, all the data is included in the repo. The same caveat as in the previous point applies. 

3. **Running the code.**

```bash
$ ./replication.sh
```

The replication script should replicate the results sequentially and very slowly if you don't have a GPU. If you already have resulting outputs in the `output/` folder, the experiments will NOT run! Take this into consideration.

It might be a good idea to append `&` to each of the executions to run them in the background at the same time, if you have enough resources and don't mind it burning down your machine. It will try to take all the available CPUs, which might not be a great idea given Python's threading model and how it plays with memory. Your mileage might vary, and PRs are more than welcome to improve this point. 

## Hardware Requirements

As per the paper and the last section, it is recommended that you run the experiments using a GPU. A refresher: For Link Prediction and Node Classification, we run each experiment on a machine with 8 cores, 16GB of memory and a single GPU with 8GB of device memory. For graph analysis, we simply use the provided notebook in the notebook folder, running on a 2015 MacBook Pro with 8 cores and 16GB of memory. Finally, the scalability analysis is ran on a CPU-only setting in a 48 cpu SLURM cluster reserving 200GB of memory per run. We realize exactly reproducing the scalability analysis is not realistic, given the workload of a SLURM cluster. The other experiments, however, are seeded and should be easy to replicate with common 'researcher-with-a-laptop' hardware.

## Author

This project is maintained by Nurudín Álvarez. You can get in touch at https://nur.wtf/

