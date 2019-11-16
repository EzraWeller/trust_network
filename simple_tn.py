import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import pandas as pd
from operator import itemgetter

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Globals
#
result_matrix = {
  'cc': (2, 2),
  'cd': (0, 3),
  'dc': (3, 0),
  'dd': (1, 1)
}

repeat_game_decay = 0.2
path_length_decay = 0.2

  # Strategy choice functions
  #
def half_good_half_bad(network, node):
  if node >= (len(network.nodes)-1)/2:
    return always_c
  return always_d

  # Strategy functions
  #
def random_move(network):
  moves = ('c', 'd')
  return moves[random.randint(0,1)]

def always_c(network):
  return 'c'

def always_d(network):
  return 'd'

  # Pairing functions
  #
def random_pairs(network):
  node_list = []
  for n in network.nodes:
    node_list.append(n)
  random.shuffle(node_list)

  pairings = []
  for i in range(len(node_list)):
    if i % 2 == 0 and i < (len(node_list)-1):
      pairings.append((node_list[i], node_list[i+1]))
  return pairings

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trust Network class
#

class TrustNetwork:
  def __init__(self, node_amount, result_matrix):
    self.network = nx.Graph()
    self.network.add_nodes_from(range(node_amount))

    self.PD = {
      'result_matrix': result_matrix,
      'max_result_sum': result_matrix['cc'][0] + result_matrix['cc'][1],
      'max_individual_result': result_matrix['dc'][0]
    }

    self.pairings = []

  def set_strategies(self, strategy_choice_function):
    for n in self.network.nodes:
      self.network.nodes[n]['strategy'] = strategy_choice_function(self.network, n)

  def set_pairings(self, pairing_function):
    self.pairings = pairing_function(self.network)

  def play_round(self):
    for p in self.pairings:
      move_0 = self.network.nodes[p[0]]['strategy'](self.network)
      move_1 = self.network.nodes[p[1]]['strategy'](self.network)
      result = self.PD['result_matrix'][move_0 + move_1]
      if (p[0], p[1]) not in self.network.edges:
        self.network.add_edge(
            p[0], 
            p[1],
            results=[]
          )
      self.network[p[0]][p[1]]['results'].append(result)

  def play_rounds(
    self, 
    rounds, 
    strategy_choice_function,
    pairing_function
  ):
    for r in range(rounds):
      self.set_strategies(strategy_choice_function)
      self.set_pairings(pairing_function)
      self.play_round()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Data processing functions
#
def find_trust(current_path_trust, current_path_max, path_length, c, results):
  edge_trust = 0
  max_edge_trust = 0
  number_of_results = len(results)

  for r in range(number_of_results):
    edge_trust += (results[r][0] + results[r][1] - 2) * (1 - repeat_game_decay)**r
    max_edge_trust += 2 * (1 - repeat_game_decay)**r
  
  path_max = current_path_max + max_edge_trust * (1 - path_length_decay)**path_length
  path_trust = (current_path_trust * current_path_max) + edge_trust * (1 - path_length_decay)**path_length
  print(c, path_trust)
  return path_trust / path_max

def found_trust_score(network, source, target):
  current_node = source
  unvisited_nodes = []
  paths = {}
  for n in network.nodes:
    unvisited_nodes.append(n)
    paths[n] = {
      'nodes': [],
      'tentative_trust': 0,
      'max_trust': 0
    }

  searching = True
  while searching == True:
    for neighbor in network[current_node]:
      if neighbor not in paths[current_node]['nodes']:
        results = network[current_node][neighbor]['results']
        tentative_trust = find_trust(
          paths[current_node]['tentative_trust'], 
          paths[current_node]['max_trust'],
          len(paths[current_node]['nodes']),
          current_node, 
          results)

        if tentative_trust > paths[neighbor]['tentative_trust']:
          paths[neighbor]['tentative_trust'] = tentative_trust
          paths[neighbor]['nodes'] = paths[current_node]['nodes'].copy()
          paths[neighbor]['nodes'].append(neighbor)

    unvisited_nodes.pop(unvisited_nodes.index(current_node))

    if target not in unvisited_nodes:
      return paths[target]['tentative_trust']
    else: 
      next_best_trust = -1
      for n in unvisited_nodes:
        if paths[n]['tentative_trust'] > next_best_trust:
          current_node = n
          next_best_trust = paths[n]['tentative_trust']
      if next_best_trust == -1:
        return paths[target]['tentative_trust']

def true_trust_score(network, PD, n, m):
# should this function try to find expected value in a general way?
  strategy_1 = network.nodes[n]['strategy']
  strategy_2 = network.nodes[m]['strategy']
  expected_value = 0
  if strategy_1 == random_move:
    if strategy_2 == random_move:
      expected_value = 3
    elif strategy_2 == always_c:
      expected_value = 3.5
    elif strategy_2 == always_d:
      expected_value = 2.5
  elif strategy_1 == always_c:
    if strategy_2 == random_move:
      expected_value = 3.5
    elif strategy_2 == always_c:
      expected_value = 4
    elif strategy_2 == always_d:
      expected_value = 3
  elif strategy_1 == always_d:
    if strategy_2 == random_move:
      expected_value = 2.5
    elif strategy_2 == always_c:
      expected_value = 3
    elif strategy_2 == always_d:
      expected_value = 2
  if expected_value == 0:
    print(strategy_1, strategy_2)
  return (expected_value - 2) / 2

def found_trust_scores(network):
  found_trusts = []
  for n in network.nodes:
    trust = (n, {})
    for m in network.nodes:
      if n != m:
        trust[1][m] = found_trust_score(network, n, m)
    found_trusts.append(trust)
  return found_trusts

def true_trust_scores(network, PD):
  true_trusts = []
  for n in network.nodes:
    trust = (n, {})
    for m in network.nodes:
      if n != m:
        trust[1][m] = true_trust_score(network, PD, n, m)
    true_trusts.append(trust)
  return true_trusts

def compare_trust_scores(found_trusts, true_trusts):
  differences = []
  for n in found_trusts:
    for t in n[1]:
      dif = n[1][t] - true_trusts[n[0]][1][t]
      if dif < 0:
        dif = dif * -1
      differences.append({ 'difference': dif })
  return differences

def mean_difference(differences):
  total = 0
  for d in differences:
    total += d['difference']
  return total / len(differences)

def ordered_trusts(trusts, node):
  trust_list = []
  for t in trusts:
    if t[0] != node:
      trust_list.append((
        t[0],
        t[1][node]
      ))
  return sorted(trust_list, key=itemgetter(1), reverse=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Run sims
#
tn = TrustNetwork(10, result_matrix)
tn.play_rounds(
  10,
  half_good_half_bad,
  random_pairs
)
found_trusts = found_trust_scores(tn.network)
true_trusts = true_trust_scores(tn.network, tn.PD)
print('found trusts', found_trusts)
print('true trusts', true_trusts)
differences = compare_trust_scores(found_trusts, true_trusts)
difference_data = pd.DataFrame(differences)
mean = round(mean_difference(differences), 3)
difference_data.plot(kind='hist', xlim=(0, 1.0), title='Mean difference = '+str(mean))
plt.show()