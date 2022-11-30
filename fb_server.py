import numpy as np
from mesa.visualization.modules import TextElement
from fb_model import FBModel
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule

from visualization.modules.XYScaledNetworkModule import XYScaledNetworkModule


def network_portrayal(model):

    def edge_color(link_type):
        if link_type == 'friend':
            return '#D3D3D3'
        elif link_type == 'itemlink':
            return '#008000'
        else:
            return '#0B66E9'

    G = model.G

    datacollector_data = model.datacollector.get_model_vars_dataframe().iloc[-1:]

    portrayal = dict()

    portrayal["nodes"] = [
        {
            "id": node_id,
            "size": 7 if G.nodes[node_id]['label'] == 'person' else 3,
            "color": "#07218a" if G.nodes[node_id]['label'] == 'person' else '#FFA500' if G.nodes[node_id]['label'] == 'news_engine' else '#e6acac',
            "fx": G.nodes[node_id]['x'],
            "fy": G.nodes[node_id]['y'],
            "tooltip": "id: {}<br>x={},y={},<br> characteristics={}".format(node_id, G.nodes[node_id]['x'], G.nodes[node_id]['y'], G.nodes[node_id]['agent'][0].get_characteristics() if G.nodes[node_id]['label'] == 'person' else "") if G.nodes[node_id]['label'] == 'person' else "id: {}<br>x={},y={}".format(node_id, G.nodes[node_id]['x'], G.nodes[node_id]['y'])
        }
        for node_id in G.nodes()
            if G.nodes[node_id]['label'] == 'person'
               or G.nodes[node_id]['label'] == 'news_engine'
               or G.nodes[node_id]['label'] == 'item' # include only people or documents
    ]

    portrayal["edges"] = [
        {"id": str(source) + '_' + str(target) + '_' + label['label'],
         "source": source,
         "target": target,
         "color": edge_color(label['label'])}
        for source, target, label in G.edges(data=True) if label['label'] == 'friend'
                                                           or label['label'] == 'itemlink'
                                                           or label['label'] == 'infolink'
    ]
    if len(datacollector_data["dist_between_friends"]) > 0:
        portrayal["datacollector_data"] = [
        {
            "dist_between_friends": str(datacollector_data["dist_between_friends"].iloc[-1]),
            "dist_between_users_and_items": str(datacollector_data["dist_between_users_and_items"].iloc[-1]),
            "dist_between_sharers": str(datacollector_data["dist_between_sharers"].iloc[-1]),
            "innate_dist": str(datacollector_data["dist_between_sharers"].iloc[-1])
        }
    ]

    return portrayal

class InfoTextElement(TextElement):
    def render(self, model):
        # days = model.current_step / model.num_steps_per_day
        return "number of nodes: {}".format(model.G.number_of_nodes())  #"Number of elapsed days so far: {:.1f}".format(days)

grid = XYScaledNetworkModule(network_portrayal, 730, 730)

number_of_agents_slider = UserSettableParameter('slider','Number of users', 30, 2, 500, 1)
num_friends_slider = UserSettableParameter('slider','Number of friends', 5, 0, 100, 1)
num_steps_per_day_slider = UserSettableParameter('slider','steps to count a day', 1, 0, 100, 1)
num_items_per_day_slider = UserSettableParameter('slider','Produced items per day', 10, 0, 100, 1)
max_num_item_links_slider = UserSettableParameter('slider','Maximum number of item links',20, 1, 100, 1)
num_exposed_items_person_day_slider = UserSettableParameter('slider','Number of exposed items per person, per day', 5, 1, 100, 1)
latitude_acceptance_slider = UserSettableParameter('slider','Latitude of acceptance', 0.3, 0, 2, 0.02)
sharpness_slider = UserSettableParameter('slider','Sharpness', 20, 0, 50, 0.5)
probability_post_social_net_slider = UserSettableParameter('slider','Probability of posting items on the social network', 0.3, 0, 1, 0.01)
user_behaviour_choice = UserSettableParameter('choice','User Behaviour',value='RS_Social_Network',\
                                              choices=['Individual_Search',\
                                                       'Individual_Search_Social_Network', \
                                                       'RS_Social_Network', \
                                                       'Recommendation'])
rs_alg_choice = UserSettableParameter('choice','User Behaviour',value='2', choices=[1,2])
rs_challenge_choice = UserSettableParameter('choice','Challenge',value='False', choices=[False,True])
network_choice = UserSettableParameter('choice','Network',value='predefined',\
                                              choices=['predefined',\
                                                       'socfb-Haverford76.mtx',
                                                       'socfb-Haverford76-copy.mtx'])
hide_friends_choice = UserSettableParameter('checkbox', 'Hide Friends', value=False)
hide_item_choice = UserSettableParameter('checkbox', 'Hide Itemlinks', value=False)

dist_between_friends_graph = ChartModule(
    [
        { "Label": "dist_between_friends", "Color": "Red"},
        { "Label": "dist_between_users_and_items", "Color": "Green"},
        { "Label": "dist_between_sharers", "Color": "Orange"},
        { "Label": "innate_dist", "Color": "Blue"}
        # { "Label": "polarisation", "Color": "Red"}
        # { "Label": "sub_graphs", "Color": "Red"},
        # { "Label": "communities", "Color": "Red"}
    ], data_collector_name='datacollector')

server = ModularServer(FBModel, [grid, InfoTextElement(), dist_between_friends_graph], 'Agent-Based Simulation - Filter Bubbles',
                       {'network': network_choice,
                        'N': number_of_agents_slider,
                        'num_friends': num_friends_slider,
                        'num_steps_per_day': num_steps_per_day_slider,
                        'num_items_per_day': num_items_per_day_slider,
                        'max_num_item_links': max_num_item_links_slider,
                        'num_exposed_items_person_day': num_exposed_items_person_day_slider,
                        'latitude_acceptance': latitude_acceptance_slider,
                        'sharpness': sharpness_slider,
                        'probability_post_social_net': probability_post_social_net_slider,
                        'user_behaviour': user_behaviour_choice,
                        'rs_alg': rs_alg_choice,
                        'rs_challenge': rs_challenge_choice,
                        'hide_friends': hide_friends_choice,
                        'hide_itemlinks': hide_item_choice})