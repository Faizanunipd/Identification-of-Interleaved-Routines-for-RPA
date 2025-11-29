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



# 2nd Option

This work analyzes UI logs generated from three randomly selected routine types.  
The following processing pipeline is used:

---

### 1. UI Log Construction
- Select 3 routine types at random.
- Generate UI event logs by merging their event traces.
- Preserve the internal event order of each routine.

---

### 2. Directly-Follows Graph (DFG) Frequency Computation
- Build a DFG from the UI logs.
- Count frequencies of all directly-following event pairs:
  - (A → B), (B → C), (C → D), etc.
- Store edge frequencies as:  
  - `DFG[(event_i, event_j)] = count`

**Example:**
- A → B : 12  
- B → C : 8  
- C → A : 5

---

### 3. Frequency Normalization using Maximum Likelihood Estimation (MLE)
- Convert raw frequencies to probabilities using MLE:
  
  `P(event_j | event_i) = freq(event_i → event_j) / sum(freq of all outgoing edges from event_i)`

- Produces a normalized transition probability matrix.

**Example:**
- A → B : 0.48  
- A → C : 0.52

---

### 4. Community Detection with Infomap
- Convert the normalized DFG into a weighted directed graph.
- Apply **Infomap** to detect functional communities.
- Identifies clusters of events frequently appearing together in interleaved routines.

---

### Output Artifacts
- Normalized DFG (probability-based)
- Weighted graph representation
- Infomap community partitions
- Visualizations (optional): graph plots, transition heatmaps, cluster diagrams






