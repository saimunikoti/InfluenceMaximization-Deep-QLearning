# InfluenceMaximization-Deep-QLearning
A scalable deep reinforcement learning framework for activation informed influence maximization on Large graphs.

## Files structure

### data
1.raw graphs: power-law cluster
2. processed graphs: edge weights-influence probability, node features-degree, node lables- candidate/participant labels.

### src
1. src/data/config.py: project directory paths
2. src/data/utils.py: utility functions
3. src/data/make_graph.py: preprocess raw graphs 
4. src/data/make_dglgraph.py: prepare graphs for dgl liblary
5. src/data/load_graph.py: load graph in dgl liblary

6. src/models/CandidateIMnodes/CandidateIMnodes_NodeClassPth.py : Node Classification model for identifying Im candiidate nodes. 
7. src/models/CandidateIMnodes/CandidateIMnodes_NodeClassPth_test.py: Test trained Node Classification model on new graphs.

8. src/models/traindqn.py: Main function for training deep Q learning agent for identifying IM seed nodes.
9.  src/models/dqnAgent.py: utility functions for dqn agent.
10. src/models/models.py: Q network model

11. src/models/testdqn.py: Test trained dqn agent.

12. src/models/Baseline_models.py: Functions for Greedy Hill climbing seeds and Influence capapcity scores.

### models
1. Trained node classification (candidiate node identiifcation) pytorch models.\
2. Trained dqn agent models.

Note: There are other files that are developed on fly but are not needed for the output generation.