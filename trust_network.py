from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD.configuration import Configuration
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

### HELPER FUNCTIONS ###
## Initial conditions
def set_network(starting_nodes, strategy_key, strategies):
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
    # set strategies by weight instead
    strat_name = return_strategy_by_ratio(strategy_key, strategies)
    network.nodes[n]['strategy'] = strategies[strat_name]
    network.nodes[n]['strategy_name'] = strat_name
    
    ### future/optional ###
      # nodes each choose an initial set of trusted nodes, 
      # which inform their set of trust scores.

      # nodes have set behaviors
        # simple: always defect, always coop
        # medium: version of tit-for-tat

  # wrap variables
  return network

# set strategy according to 
def return_strategy_by_ratio(strategy_key, strategies):
  roll = random.random()
  for k in strategy_key:
    if roll < strategy_key[k]:
      return k

## Policies
# matchmaking functions
def choose_random_pairings(network):
  node_list = []
  for n in network.nodes:
    node_list.append(n)
  random.shuffle(node_list)

  pairings = []
  for i in range(len(node_list)):
    if i % 2 == 0 and i < (len(node_list)-1):
      pairings.append((node_list[i], node_list[i+1]))
  return pairings

def most_trusted_pairings(network):
  node_list = []
  nodes_left = []
  for n in network.nodes:
    node_list.append(n)
    nodes_left.append(n)
  random.shuffle(node_list)
  pairings = []
  paired_nodes = []
  for i in range(len(node_list)):
    if node_list[i] in nodes_left:
      most_trusted = (
        0, # node
        -1 # trust score
      )
      if len(network[node_list[i]]) > 0:
        for t in network.nodes[node_list[i]]['trusts']:
          if (
            network.nodes[node_list[i]]['trusts'][t] > most_trusted[1] and 
            t in nodes_left
          ):
            most_trusted = (t, network.nodes[node_list[i]]['trusts'][t])
        if most_trusted[1] >= 0.7:
          pairings.append((node_list[i], most_trusted[0]))
          nodes_left.pop(nodes_left.index(node_list[i]))
          nodes_left.pop(nodes_left.index(most_trusted[0]))
      if most_trusted[1] < 0.7 and len(nodes_left) > 1:
        nodes_left.pop(nodes_left.index(node_list[i]))
        nodes_left_index = random.randint(0, len(nodes_left)-1)
        pairings.append((node_list[i], nodes_left[nodes_left_index]))
        nodes_left.pop(nodes_left_index)
  return pairings

# play games
def play_with_random_strategies(node_0, node_1, moves, edge, result_matrix):
  move_0 = moves[random.randint(0,1)]
  move_1 = moves[random.randint(0,1)]
  return result_matrix[move_0 + move_1]

def play_with_node_strategies(node_0, node_1, moves, edge, result_matrix):
  move_0 = node_0['strategy'](moves, edge, result_matrix)
  move_1 = node_1['strategy'](moves, edge, result_matrix)
  return result_matrix[move_0 + move_1]

# find new edge weights (create edge if doesn't exist)
def new_edge_weights(PD, pairings, network):
  moves = PD['moves']
  result_matrix = PD['result_matrix']
  play_function = PD['play_function']
  max_result_sum = PD['max_result_sum']
  for pair in pairings:
    if [pair[0], pair[1]] not in network.edges:
      network.add_edge(
        pair[0], 
        pair[1], 
        games=0, 
        sums={
          pair[0]: 0,
          pair[1]: 0
        }, 
        weight=0
      )
    edge = network[pair[0]][pair[1]]
    node_0 = network.nodes[pair[0]]
    node_1 = network.nodes[pair[1]]
    result = play_function(node_0, node_1, moves, edge, result_matrix)
    edge['games'] += 1
    edge['sums'][pair[0]] += result[0]
    edge['sums'][pair[1]] += result[1]
    edge['weight'] = (
      (edge['sums'][pair[0]] + edge['sums'][pair[1]])
      / (max_result_sum * edge['games'] * 1.01**edge['games'])
    )
  return network

## Strategies
def random_move(moves, edge, result_matrix):
  return moves[random.randint(0,1)]

def history_based(moves, edge, result_matrix):
  # if they haven't played, default to cooperate
  if edge['games'] == 0:
    return 'c'
  max_result = result_matrix['cc'][0] + result_matrix['cc'][1]
  mid_result = result_matrix['cd'][0] + result_matrix['cd'][1]
  max_sum = edge['games'] * max_result
  mid_sum = edge['games'] * mid_result
  edge_sum = 0
  for k in edge['sums']:
    edge_sum += edge['sums'][k]
  # if on average they're cooperating, cooperate
  if (edge_sum / max_sum) > (mid_sum / max_sum):
    return 'c'
  # if on average they're defecting, defect
  else:
    return 'd'

def always_defect(moves, edge, result_matrix):
  return 'd'

def always_coop(moves, edge, result_matrix):
  return 'c'

# Group by strategy
def group_by_strategy(graph):
  nodes_by_strat = {}
  strat_names = []

  for n in range(len(graph.nodes)):
    strategy = graph.nodes.data('strategy_name')[n]
    if strategy not in nodes_by_strat:
      nodes_by_strat[strategy] = []
      strat_names.append(strategy)
    nodes_by_strat[strategy].append(n)

  returns_by_strat = {}

  for s in strat_names:
    returns_by_strat[s] = { 'sum': 0, 'games': 0 }
    for n in nodes_by_strat[s]:
      for k in graph[n]:
        returns_by_strat[s]['sum'] += graph[n][k]['sums'][n]
        returns_by_strat[s]['games'] += graph[n][k]['games']
        num = returns_by_strat[s]['sum']
        denom = returns_by_strat[s]['games'] * max_individual_result
        returns_by_strat[s]['success_ratio'] = num / denom

  success_ratio_denom = 0

  for k in returns_by_strat:
    success_ratio_denom += returns_by_strat[k]['success_ratio']

  for k in returns_by_strat:
    returns_by_strat[k]['normalized_ratio'] = (
      returns_by_strat[k]['success_ratio'] / success_ratio_denom
    )
  
  return returns_by_strat

### GLOBAL VARIABLES ###
iterations = 100
starting_nodes = 10
strategy_update_loops = 30

### INITIAL CONDITIONS ###
# prisoner's dilemma config
moves = ('c', 'd')
result_matrix = {
  'cc': [2, 2],
  'cd': [0, 3],
  'dc': [3, 0],
  'dd': [1, 1]
}
max_result_sum = result_matrix['cc'][0] + result_matrix['cc'][1]
max_individual_result = result_matrix['dc'][0]
strategies = {
  'random_move': random_move,
  'history_based': history_based,
  'always_coop': always_coop,
  'always_defect': always_defect
}
strategy_key = {}
original_key_sum = 0
for k in strategies:
  original_key_sum += 1/len(strategies)
  strategy_key[k] = original_key_sum

initial_conditions = {}

### POLICY FUNCTIONS ###
# nodes play prisoner's dilemmas to determine changes to edge weights
def play(params, step, sL, s):
  network = s['network']
  PD = params[0]['PD']
  trust = params[0]['trust']
  pairings = PD['matchmaking_function'](network)
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
      'matchmaking_function': most_trusted_pairings,
      'play_function': play_with_node_strategies
    },
    'trust': {
      'weight_function': new_edge_weights,
    }
  },
}

#### STRATEGY UPDATE LOOPS
results = {}

for l in range(strategy_update_loops):
  loop_results = {}

  ### ASSEMBLE CONFIG ###
  initial_conditions = { 'network': set_network(starting_nodes, strategy_key, strategies) }
  print(initial_conditions)

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

  # Overall Weight ratio
  max_total_result = 0
  for g in graph.edges.data('games'):
    max_total_result += g[2] * max_result_sum
  total_result = 0
  for s in graph.edges.data('sums'):
    total_result += s[2][s[0]] + s[2][s[1]]

  # By strategy
  returns_by_strat = group_by_strategy(graph)

  ### PLOT RESULTS ###
  print('Overall weight ratio:')
  print('---', total_result, '/', max_total_result, '=', total_result/max_total_result)
  print('Returns by strat:')
  for k in returns_by_strat:
    print(
      '---', 
      k,
      returns_by_strat[k]['sum'],
      '/',
      returns_by_strat[k]['games'] * max_individual_result,
      '= raw', 
      returns_by_strat[k]['success_ratio'], 
      '| normalized', 
      returns_by_strat[k]['normalized_ratio']
    )

  results[l] = { 'Total result ratio': str(total_result)+'/'+str(max_total_result) }

  # adjust initial conditions for next loop
  strategy_key = {}
  new_strategy_sum = 0
  for k in returns_by_strat:
    results[l][k] = returns_by_strat[k]['normalized_ratio']
    new_strategy_sum += returns_by_strat[k]['normalized_ratio']
    strategy_key[k] = new_strategy_sum


data = pd.DataFrame(results)
print(data)

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

  # Check the average return for each node's games and average by their strategy
    # which strategy leads to the highest average return?
    # Todo

  