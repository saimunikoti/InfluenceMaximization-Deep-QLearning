# InfluenceMaximization-Deep-QLearning

A scalable deep reinforcement learning framework for activation informed influence maximization on Large graphs.

Anonymized GitHub repo for KDD 2022 Submission.

This code is provided solely for the purpose of peer review for the KDD 2022 conference.

===================== File specification =========================

### data
1. data/ca-CSphd/: Train and val graphs from different families.

### script files
1. src/data/config.py: project directory paths
2. src/data/utils.py: utility functions
3. src/data/make_graph.py: preprocess raw graphs 
4. src/data/make_dglgraph.py: prepare graphs for dgl liblary
5. src/data/load_graph.py: load graph in dgl liblary

6. src/models/CandidateIMnodes/CandidateIMnodes_NodeClassPth.py : Node Classification model for identifying AIM candiidate nodes. 

7. src/models/CandidateIMnodes/CandidateIMnodes_NodeClassPth_test.py: Test trained Node Classification model on new graphs.

8. src/models/traindqn.py: Main function for training deep Q learning agent for identifying AIM seed nodes.
9.  src/models/dqnAgent.py: utility functions for Q learning agent.
10. src/models/models.py: Q network model.

11. src/models/testdqn.py: Test trained dqn agent and generate plots of performance.

12. src/models/Baseline_models.py: Functions for getting ground truths from Greedy Hill climbing and Influence capapcity scores.

### models
1. Trained node classification (candidiate node identiifcation)  models in torch format.
2. Trained QLearning agent models.

Note: There are other files that are developed on fly but are not needed for the output generation.

==================================== end =========================