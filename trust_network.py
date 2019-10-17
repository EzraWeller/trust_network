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

repeat_game_decay = 0.1
path_length_decay = 0.3

  # Helper functions
  #
def find_trust_rate(paths, current_node, neighbor, results, PD):
  edge_weight = 0
  max_edge_weight = 0
  min_edge_weight = 0
  number_of_results = len(results)

  for r in range(number_of_results):
    edge_weight += results[r][0] + results[r][1]
    max_edge_weight += PD['max_result_sum']
    min_edge_weight += (PD['result_matrix']['dd'][0] + PD['result_matrix']['dd'][1])
  
  edge_weight = edge_weight * (1 - repeat_game_decay)**number_of_results
  max_edge_weight = max_edge_weight * (1 - repeat_game_decay)**number_of_results
  min_edge_weight = min_edge_weight * (1 - repeat_game_decay)**number_of_results

  weight = paths[current_node]['weight'] + edge_weight
  max_weight = paths[current_node]['max_weight'] + max_edge_weight
  min_weight = paths[current_node]['min_weight'] + min_edge_weight

  distance = len(paths[current_node]['nodes'])
  tentative_trust_weight = weight / distance * (1 - path_length_decay)**(distance - 1)
  tentative_trust_rate = (weight - min_weight) / (max_weight - min_weight) * (1 - path_length_decay)**(distance - 1)

  return [weight, max_weight, min_weight, tentative_trust_rate, tentative_trust_weight]

def best_path_value(path_type, value_function, network, PD, source, target, worst=False):
  def conditional_greater_than(a, b, worst):
    if worst == True:
      return a < b
    return a > b

  best_value = (0,0)

  current_node = source
  unvisited_nodes = []
  paths = {}
  for n in network.nodes:
    unvisited_nodes.append(n)
    paths[n] = {
      'nodes': [n], 
      'weight': 0, 
      'max_weight': 0, 
      'min_weight': 0,
      path_type: -1,
      'tentative_weight': 0
    }
    if worst == True:
      paths[n][path_type] = 1000000

  searching = True
  while searching == True:

    for neighbor in network[current_node]:
      if neighbor not in paths[current_node]['nodes']:
        results = network[current_node][neighbor]['results']

        values = value_function(paths, current_node, neighbor, results, PD)

        if conditional_greater_than(values[3], paths[neighbor][path_type], worst):
          paths[neighbor]['weight'] = values[0]
          paths[neighbor]['max_weight'] = values[1]
          paths[neighbor]['min_weight'] = values[2]
          paths[neighbor][path_type] = values[3]
          paths[neighbor]['tentative_weight'] = values[4]
          paths[neighbor]['nodes'] = paths[current_node]['nodes'].copy()
          paths[neighbor]['nodes'].append(neighbor)

    unvisited_nodes.pop(unvisited_nodes.index(current_node))

    if target not in unvisited_nodes:
      best_value = (round(paths[target][path_type], 2), round(paths[target]['tentative_weight'], 2))
      searching = False
    else: 
      next_best_trust = -1
      if worst == True:
        next_best_trust = 1000000
      for n in unvisited_nodes:
        if conditional_greater_than(paths[n][path_type], next_best_trust, worst):
          current_node = n
          next_best_trust = paths[n][path_type]
      if next_best_trust == -1 or next_best_trust == 1000000:
        searching = False

  return best_value

  # Strategy choice functions
  #
def half_good_half_bad(network, node):
  if node >= (len(network.nodes)-1)/2:
    return always_c
  return always_d

def even_split(network, node):
  list_of_strategy_functions = [always_c, always_d, random_move]
  for n in range(len(list_of_strategy_functions)):
    if node / len(network.nodes) <= (n + 1) / len(list_of_strategy_functions):
      return list_of_strategy_functions[n]

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

def trusted_pairs(network):
  node_list = []
  for n in network.nodes:
    node_list.append(n)
  random.shuffle(node_list)

  pairings = []
  used_nodes = []
  for i in node_list:
    if i not in used_nodes:
      # take most trusted if possible
      if 'trusts' in network.nodes[i]:
        trusts = []
        for t in network.nodes[i]['trusts']:
          trusts.append((
            t,
            network.nodes[i]['trusts'][t]['cumulative'],
            network.nodes[i]['trusts'][t]['rate']
          ))
        sorted_trusts = sorted(trusts, key=itemgetter(1,2))
        found = False
        index = -1
        while found == False:
          if sorted_trusts[index][2] >= 0.7 and sorted_trusts[index][0] not in used_nodes:
            pairings.append((i, sorted_trusts[index][0]))
            used_nodes.append(i)
            used_nodes.append(sorted_trusts[index][0])
            found = True
          index -= 1
          if index * -1 > len(sorted_trusts):
            for m in node_list:
              if m not in used_nodes and i != m and found == False:
                pairings.append((i, m))
                used_nodes.append(i)
                used_nodes.append(m)
                found = True
            found = True
      # otherwise take random
      else:
        found = False
        while found == False:
          for m in node_list:
            if m not in used_nodes and i != m and found == False:
              pairings.append((i, m))
              used_nodes.append(i)
              used_nodes.append(m)
              found = True
          found = True

  return pairings

  # Trust score functions
  #
def strongest_path_trust(network, PD, source, target):
  trust = best_path_value('trust_rate', find_trust_rate, network, PD, source, target)
  return { 'cumulative': trust[1], 'rate': trust[0] }

# def strongest_path_through_friends()
  # for each friend
    # find the strongest path to the friend, then from the friend to the target
  # take the strongest of those paths

def best_and_worst_trust(network, PD, source, target):
  best_trust = best_path_value('trust_rate', find_trust_rate, network, PD, source, target)
  worst_trust = best_path_value('trust_rate', find_trust_rate, network, PD, source, target, worst=True)
  best_trust_weight = 0
  if best_trust[1] != 0:
    best_trust_weight = best_trust[1]/(best_trust[1] + worst_trust[1])
  worst_trust_weight = 0
  if worst_trust[1] != 0:
    worst_trust_weight = worst_trust[1]/(best_trust[1] + worst_trust[1])
  print('best_trust_weight', best_trust_weight, 'best_trust', best_trust[0], 'worst_trust_weight', worst_trust_weight, 'wort_trust', worst_trust[0])
  return { 'rate': best_trust_weight * best_trust[0] + worst_trust_weight * worst_trust[0] }

# Maybe a trust rating that finds best-rated path AND worst-rated path, then averages 
# them weighted by their cumulative trust

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
      # Todo: result might need to specify by player so we can tell who got 0 and who got
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
          self.network.nodes[n]['trusts'][m] = trust_score_function(
            self.network, 
            self.PD,
            n, 
            m
          )

  def play_rounds(
    self, 
    rounds, 
    strategy_choice_function, 
    pairing_function,
    trust_score_function
  ):
    for r in range(rounds):
      self.set_strategies(strategy_choice_function)
      self.set_pairings(pairing_function)
      self.play_round()
      self.set_trust_scores(trust_score_function)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Data processing functions
#
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
  return {'rate': (expected_value - 2) / 2 }

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
      if t > n[0]:
        dif = n[1][t]['rate'] - true_trusts[n[0]][1][t]['rate']
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
    trust_list.append((
      t,
      trusts[t]['rate']
    ))
  return sorted(trust_list, key=itemgetter(1), reverse=True)
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Run sims
#
tn = TrustNetwork(10, result_matrix)
tn.play_rounds(
  2,
  even_split,
  random_pairs,
  strongest_path_trust
)
found_trusts = tn.network.nodes.data('trusts')
true_trusts = true_trust_scores(tn.network, tn.PD)
print('found trusts', ordered_trusts(found_trusts[0], 0))
print('true trusts', ordered_trusts(true_trusts[0][1], 0))
differences = compare_trust_scores(found_trusts, true_trusts)
difference_data = pd.DataFrame(differences)
mean = round(mean_difference(differences), 3)
difference_data.plot(kind='hist', xlim=(0, 1.0), title='Mean difference = '+str(mean))
plt.show()

# success metrics
  # 1. how close the trust scores are to the "true" trust scores for the strategies
    # e.g. "always defect" should lead to a trust score of ....?
  # 2. the success rate of the network: 
    # how much trust has been generated compared to the amount of trust that could have been generated with the same games
  # 3. the total trust weight of the network
  # 2. and 3. but for specific strategies

# Idea:
  # random pairings
  # only use strategies with calculable "true" trust scores (how much they should be trusted)
    # e.g. 'always defect' has an expected result of 'dd' against 'always defect': 
    # from this we can calculate the "true" trust between these two nodes
    # three simple strategies: random move, always defect, always coop
  # x rounds
  # generate estimated trust scores using history
  # compare to actual trust scores using expected values
    # compare by comparing the *order*: for node n, 
    # in what order should they trust other nodes, and what order did we find