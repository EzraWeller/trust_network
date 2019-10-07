import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import pandas as pd

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
def test_strategy_choice_function(network, node):
  return random_move

def half_good_half_bad(network, node):
  if node >= len(network.nodes)/2:
    return always_c
  return always_d

  # Strategy functions
  #
def test_strategy_function(network):
  return 'c'

def random_move(network):
  moves = ('c', 'd')
  return moves[random.randint(0,1)]

def always_c(network):
  return 'c'

def always_d(network):
  return 'd'

  # Pairing functions
  #
def test_pairing_function(network):
  return [(0, 1)]

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

def trusted_pairs(network):
  
  return

  # Trust score functions
  #
def test_trust_score_function(network, source, target):
  return 1

def shortest_path_trust(network, source, target):
  # this is working right now!
  # BUT, it's leading to 0 trust between some nodes even though their are connections between every
  # node, because there are sometimes no *results* on a given edge, and it's only looking for one path
  # solutions: look for a few non-overlapping paths, only draw edges where there are results
  paths_to_find = 1
  best_trusts = []
  unused_nodes = []

  current_node = source
  unvisited_nodes = []
  paths = {}
  for n in network.nodes:
    unvisited_nodes.append(n)
    unused_nodes.append(n)
    paths[n] = {'nodes': [n], 'weight': 0, 'tentative_trust': -1}

  searching = True
  while searching == True:
    for neighbor in network[current_node]:
      if (neighbor in unused_nodes) and (neighbor not in paths[current_node]['nodes']):
        results = network[current_node][neighbor]['results']
        edge_weight = 0
        number_of_results = len(results)
        for r in range(number_of_results):
          edge_weight += (results[r][0] + results[r][1]) * (1 - repeat_game_decay)**r
        weight = paths[current_node]['weight'] + edge_weight
        distance = len(paths[current_node]['nodes'])
        trust = weight / distance * (1 - path_length_decay)**(distance - 1)
        if trust > paths[neighbor]['tentative_trust']:
          paths[neighbor]['weight'] = paths[current_node]['weight'] + edge_weight
          paths[neighbor]['nodes'] = paths[current_node]['nodes'].copy()
          paths[neighbor]['nodes'].append(neighbor)
          paths[neighbor]['tentative_trust'] = trust
    unvisited_nodes.pop(unvisited_nodes.index(current_node))
    if target not in unvisited_nodes:
      for n in paths[target]['nodes']:
        if n != (source or target):
          unused_nodes.pop(unused_nodes.index(n))
      best_trusts.append(round(paths[target]['tentative_trust'], 2))
      print('Best path', source, paths[target]['nodes'])
      searching = False
    else: 
      next_best_trust = -1
      for n in unvisited_nodes:
        if paths[n]['tentative_trust'] > next_best_trust:
          current_node = n
          next_best_trust = paths[n]['tentative_trust']
      if next_best_trust == -1:
        searching = False
  return best_trusts
    
#  
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
  
  def set_trust_scores(self, trust_score_function):
    for n in self.network.nodes:
      self.network.nodes[n]['trusts'] = {}

    for n in self.network.nodes:
      for m in self.network.nodes:
        if 'trusts' in self.network.nodes[m] and n in self.network.nodes[m]['trusts']:
          self.network.nodes[n]['trusts'][m] = self.network.nodes[m]['trusts'][n]
        elif n != m:
          self.network.nodes[n]['trusts'][m] = trust_score_function(self.network, n, m)

  def play_rounds(
    self, 
    rounds, 
    strategy_choice_function, 
    pairing_function,
    trust_score_function
  ):
    for r in range(rounds):
      print('round', r)
      self.set_strategies(strategy_choice_function)
      self.set_pairings(pairing_function)
      self.play_round()
      self.set_trust_scores(trust_score_function)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Run sims
#
tn = TrustNetwork(5, result_matrix)
tn.play_rounds(
  20,
  half_good_half_bad,
  random_pairs,
  shortest_path_trust
)
print(pd.DataFrame(tn.network.nodes.data('trusts')))
print(pd.DataFrame(tn.network.edges.data('results')))
nx.draw_kamada_kawai(tn.network)
plt.show()
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #