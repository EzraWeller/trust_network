from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD.configuration import Configuration
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

### HELPER FUNCTIONS ###
## Policies
# choose partners
def choose_random_pairings(nodes):
  nodeList = []
  for node in nodes:
    nodeList.append(node)
  random.shuffle(nodeList)
  pairings = []
  for i in range(len(nodeList)):
    if i % 2 == 0 and i < (len(nodeList)-1):
      pairings.append((nodeList[i], nodeList[i+1]))
  return pairings

# play games
def play_with_random_strategies(moves, result_matrix, nodePair):
  move_0 = moves[random.randint(0,1)]
  move_1 = moves[random.randint(0,1)]
  return result_matrix[move_0 + move_1]

# find new edge weights (create edge if doesn't exist)
def new_edge_weights(PD, pairings, network):
  moves = PD['moves']
  result_matrix = PD['result_matrix']
  play_function = PD['play_function']
  max_result_sum = PD['max_result_sum']
  for pair in pairings:
    result = play_function(moves, result_matrix, pair)
    result_sum = result[0] + result[1]
    if [pair[0], pair[1]] not in network.edges:
      network.add_edge(pair[0], pair[1], games=0, sum=0, weight=0)
    network[pair[0]][pair[1]]['games'] += 1
    network[pair[0]][pair[1]]['sum'] += result_sum
    network[pair[0]][pair[1]]['weight'] = round((
      network[pair[0]][pair[1]]['sum'] / 
      (max_result_sum * network[pair[0]][pair[1]]['games']
      * 1.1**network[pair[0]][pair[1]]['games'])
    ), 4)
  return network

### GLOBAL VARIABLES ###
iterations = 10
starting_nodes = 10

### INITIALIZE STATE VARIABLES ###
# prisoner's dilemma config
moves = ('c', 'd')
result_matrix = {
  'cc': [2, 2],
  'cd': [0, 3],
  'dc': [3, 0],
  'dd': [1, 1]
}
max_result_sum = 4

# graph with nodes
network = nx.Graph()
network.add_nodes_from(range(starting_nodes))

# trust score matrix: for each node, a set of trust scores for all other nodes
for n in network.nodes:
  initial_trusts = {}
  for i in range(starting_nodes):
    if i != n:
      initial_trusts[i] = 0
  network.nodes[n]['trusts'] = initial_trusts
  
  ### future/optional ###
    # nodes each choose an initial set of trusted nodes, 
    # which inform their set of trust scores.

    # nodes have set behaviors
      # simple: always defect, always coop
      # medium: version of tit-for-tat

# wrap variables
initial_conditions = { 'network': network }

### POLICY FUNCTIONS ###
# nodes play prisoner's dilemmas to determine changes to edge weights
def play(params, step, sL, s):
  network = s['network']
  PD = params[0]['PD']
  trust = params[0]['trust']
  pairings = PD['matchmaking_function'](network.nodes)
  network_with_new_weights = trust['weight_function'](PD, pairings, network)
  return { 'network': network_with_new_weights }

  ### future/optional ###
    # nodes may only play with nodes they trust
    # nodes

# find trust scores
def shortest_path_trust_scores(params, step, sL, s):
  network = s['network']
  for node_a in network.nodes:
    for node_b in network.nodes:
      if node_a != node_b:
        try: 
          path = nx.shortest_path(network, source=node_a, target=node_b)
          trust = 0
          path_weight = 0
          path_length = len(path) - 1
          for i in range(path_length):
            if i < path_length:
              path_weight += network[path[i]][path[i+1]]['weight']
          if path_length > 0:
            trust += path_weight / path_length
          network.nodes[node_a]['trusts'][node_b] = trust
        except nx.exception.NetworkXNoPath:
          continue
  return { 'network': network }

### STATE UPDATE FUNCTIONS ###
# update edge weights
def update_edges(params, step, sL, s, _input):
  return ('network', _input['network'])

# update trust scores
def update_trusts(params, step, sL, s, _input):
  return ('network', _input['network'])

### UPDATE BLOCKS ###
partial_state_update_blocks = [
  # block 1: play dilemmas, change edge weights
  {
    'policies': {
      'action': play
    },
    'variables': {
      'network': update_edges
    }
  },
  # block 2: determine trust scores, update trust scores
  {
    'policies': {
      'action': shortest_path_trust_scores
    },
    'variables': {
      'network': update_trusts
    }
  }
]

### SIMULATION PARAMS ###
simulation_parameters = {
  'T': range(iterations),
  'N': 1,
  'M': {
    'PD': {
      'moves': moves,
      'result_matrix': result_matrix,
      'max_result_sum': max_result_sum,
      'matchmaking_function': choose_random_pairings,
      'play_function': play_with_random_strategies
    },
    'trust': {
      'weight_function': new_edge_weights,
    }
  },
}

### ASSEMBLE CONFIG ###
config = Configuration(
  initial_state=initial_conditions,
  partial_state_update_blocks=partial_state_update_blocks,
  sim_config=simulation_parameters
)

### RUN SIMULATIONS ###
exec_mode = ExecutionMode()
exec_context = ExecutionContext(exec_mode.single_proc)
executor = Executor(exec_context, [config])
raw_result, tensor = executor.execute()

### PREPARE RESULTS ###
graph = raw_result[len(raw_result)-1]['network']

# Weight ratio
max_total_result = 0
for g in graph.edges.data('games'):
  max_total_result += g[2] * max_result_sum
total_result = 0
for s in graph.edges.data('sum'):
  total_result += s[2]

### PLOT RESULTS ###
print('Sums', graph.edges.data('sum'))
print('Weight ratio', total_result, '/', max_total_result, '=', total_result/max_total_result)
# nx.draw_kamada_kawai(graph)
# plt.show()

### possible measures of success
# From trust rank:
  # give each node an "actual" trust score, then check how well the trust scores are discovering those?

# original
  # Maximize the weight ratio of the network (the result sum / max possible result sum)
  # this approximates how successful the network is at facilitating "successful" interactions (coop/coop)
    # for this to make sense, nodes must be greedy (they want to defect/coop when they can get away with it)
    # to stop nodes from just defecting all the time.
    # You could also compare this ratio to the expected ratio of the result matrix if
      # players played randomly
      # players played nash perfectly

  # Maximize the median return on a game?

  