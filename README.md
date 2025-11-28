# Identification-of-Interleaved-Routines-for-RPA

This repository provides an implementation of a graph-based technique
for identifying **routine types** from **interleaved UI logs** in
Robotic Process Automation (RPA). 

## Overview

Given a UI log the method follows two main phases:

### 1. Normalized Directly-Follows Graph (NDFG)


### 2. Community Detection for Routine Identification


## Repository Structure

    graph_functions.py     # NDFG construction, normalization, and clustering
    utils.py               # Log processing, evaluation, exports, helper routines
    notebooks/             # Experimental notebooks
    README.md              # Project documentation



## Installation

    pip install pm4py networkx infomap-python python-louvain numpy pandas matplotlib

## Basic Usage

``` python
from graph_functions import discover_dfg, get_Network_Graph, infomap_clustering
import pandas as pd

log = pd.read_csv("logs/1.csv")
dfg = discover_dfg(log)
G = get_Network_Graph(dfg)
clusters, _ = infomap_clustering(G)
print(clusters)
```

## Purpose

This repository supports research on the discovery of **interleaved
routines** in RPA environments.




