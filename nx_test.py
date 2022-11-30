import random

import networkx as nx
import numpy as np
from numpy import mean
import networkx.algorithms.community as nxcom

G = nx.MultiGraph()
G.add_node(0)
G.add_node(1)
G.add_node(2)
G.add_node(3)

nx.set_node_attributes(G, {0: {'x': 1, 'y': 1, 'label':'person', 'innate_x':-1, 'innate_y':-1}})
nx.set_node_attributes(G, {1: {'x': 1, 'y': 1, 'label':'person', 'innate_x':-1, 'innate_y':-1}})
nx.set_node_attributes(G, {2: {'x': -1, 'y': -1, 'label':'person', 'innate_x':-1, 'innate_y':-1}})
nx.set_node_attributes(G, {3: {'x': -1, 'y': -1, 'label':'item', 'step':10}})

G.add_edge(0, 1, key=0, label='friend',rejections=0)
G.add_edge(1, 2, key=0, label='friend',rejections=0)

items = [a for a, b in G.nodes(data=True) if b['label'] == 'item' and b['step'] <= 10 and G.degree[a] == 0]

print('items={}'.format(items))




# G.add_edge(1, 3, key=0, label='friend',rejections=0)
# G.add_edge(2, 3, key=0, label='friend',rejections=0)

# sub_graphs = nx.connected_components(G)
#
# H = G.copy()
#
# bumba = [x for x,y in G.nodes(data=True) if y['label'] == 'item']
# print('bumba', bumba)
#
# while True:
#     try:
#         print(next(sub_graphs))
#     except StopIteration:
#         break
#
# G_karate = nx.karate_club_graph()
# # Find the communities
# communities = sorted(nxcom.greedy_modularity_communities(G_karate), key=len, reverse=True)
# # Count the communities
# print(f"The karate club has {len(communities)} communities.")
#
# def detect_comm():
#     import matplotlib.pyplot as plt
#     import networkx as nx
#     from networkx.algorithms.community.centrality import girvan_newman
#
#     G = nx.karate_club_graph()
#     communities = girvan_newman(G)
#
#     node_groups = []
#     for com in next(communities):
#         node_groups.append(list(com))
#
#     print(node_groups)
#
#     color_map = []
#     for node in G:
#         if node in node_groups[0]:
#             color_map.append('blue')
#         else:
#             color_map.append('green')
#     nx.draw(G, node_color=color_map, with_labels=True)
#     plt.show()
#
# detect_comm()
#
# def get_polarisation(G):
#     items = [(b['x'], b['y']) for a,b in G.nodes(data=True) if b['label'] == 'person']
#     average = get_average_xy(G)
#     polarisation = [ ( (x - average[0]) ** 2, (y - average[1]) ** 2 ) for x, y in items]
#     return tuple(map(sum, zip(*polarisation)))
#     return polarisation
#
# def get_average_xy(G):
#     items = [(b['x'], b['y']) for a,b in G.nodes(data=True) if b['label'] == 'person']
#     print('items',items)
#     return tuple(map(mean, zip(*items)))
#
# print('polarisation', get_polarisation(G))
# print('items', get_average_xy(G, 'person'))


# def update_infolink(G, source, destination):
#     key = 1
#     if G[source][destination].get(key):
#         G[source][destination].get(key)['refs'] += 1
#     else:
#         G.add_edge(source, destination, key=key, label='infolink', refs=1)

# def add_fof(G, id):
#     friends = get_links(G,id, 0, 'friend')
#     if len(friends) == 0:
#         return
#     random_friend = friends[random.randrange(0,len(friends))]
#
#     fof = [x for x in get_links(G, random_friend, 0, 'friend') if x != id]
#     if len(fof) == 0:
#         return
#     random_fof = fof[random.randrange(0, len(fof))]
#
#     G.add_edge(id, random_fof, key=0, label='friend', rejections=0)


# def get_links(G, id, key, label, include_origin=False):
#     """ Get links of a person with nodes containing a specific label
#     :param G: The graph
#     :param key: The edge key
#     :param label: The label (e.g. 'itemlink','friend')
#     :return: a list of links
#     """
#     itemlinks = []
#     for (u, v) in G.edges(id):
#         if G[u][v].get(key) and G[u][v][key]['label'] == label:
#             itemlinks.append(v if include_origin == False else (u,v))
#     return itemlinks
#
#
# update_infolink(G,1,2)
# update_infolink(G,1,2)
# update_infolink(G,1,2)

# def remove_infolink(G, id, item):
#     key = 1
#     links = [x for x in get_links(G, item, 0, 'itemlink') if x != id]
#     for link in links:
#         if G.has_edge(link, id, key):
#             if G[link][id].get(key)['refs'] > 1:
#                 G[link][id].get(key)['refs'] -= 1
#             else:
#                 G.remove_edge(link,id,key)
#
# remove_infolink(G, 1, 3)
# remove_infolink(G, 1, 3)
# remove_infolink(G, 1, 3)
#
# add_fof(G,0)
#
# print(G.edges(data=True))

#
# import statistics
#
# def calc_avg_dist_links_all_nodes(source_label, target_key, target_label, G):
#     return statistics.mean( calc_avg_dist_links(a, target_key, target_label, G) for a, b in G.nodes.data() if b['label'] == source_label )
#
# def calc_avg_dist_links(origin, target_key, target_label, G):
#     distances = [get_distance(G, origin, i) for i in get_links(origin, G, target_key, target_label)]
#     if not distances:
#         return None
#     else:
#         return statistics.mean(distances)
#
# def get_links(node,G, key, label, include_origin=False):
#     """ Get links of a person with nodes containing a specific label
#     :param G: The graph
#     :param key: The edge key
#     :param label: The label (e.g. 'itemlink','friend')
#     :return: a list of links
#     """
#     itemlinks = []
#     for (u, v) in G.edges(node):
#         if G[u][v].get(key) and G[u][v][key]['label'] == label:
#             itemlinks.append(v if include_origin == False else (u,v))
#     return itemlinks
#
# def get_distance(G, u, v):
#     x_u = G.nodes[u]['x']
#     y_u = G.nodes[u]['y']
#     x_v = G.nodes[v]['x']
#     y_v = G.nodes[v]['y']
#     return math.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)
#
# print(list(get_links(a,G,0,'itemlink',True) for a, b in G.nodes.data() if b['label'] == 'person' ))
#
# print( list(itertools.chain.from_iterable(get_links(a,G,0,'itemlink',True) for a, b in G.nodes.data() if b['label'] == 'person' )))
#
#
#
# # print('position(0)=({},{})'.format(G.nodes[0]['x'],G.nodes[0]['y']))
# # print('get_links(0)=',get_links(0,G,0,'itemlink'))
# # print(calc_avg_dist_links(0,0,'itemlink',G))
# # print(calc_avg_dist_links(0,0,'friend',G))
# # print(calc_avg_dist_links_all_nodes('person',0, 'itemlink',G))
# # print(calc_avg_dist_links_all_nodes('person',0, 'friend',G))
#
