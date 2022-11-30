import abc

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid, NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
from numpy import mean
import networkx.algorithms.community as nxcom
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from collections import defaultdict
from networkx.readwrite import json_graph
import json

import random
import statistics
import math
import timeit
import csv

data = []

# First train an SVD algorithm on the movielens dataset.
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def add_fof(G, id, friend, step, lot):
    """
    Adding a random friend of a friend
    :param G: The graph object
    :param id: The agent id
    :return:
    """
    # getting a list of friends
    friends = get_links(G,id, 0, 'friend')
    if len(friends) == 0:
        return
    
    random_friend = friends[random.randrange(0,len(friends))]
    friends_of_friend_likeminded = [x for x in get_links(G, random_friend, 0, 'friend') if x != id and x not in friends and get_distance(G, id, x) <= lot]
    # print('fof of ',id,fof,'friend',friend)
    if len(friends_of_friend_likeminded) == 0:
        return

    random_likeminded_fof = friends_of_friend_likeminded[random.randrange(0, len(friends_of_friend_likeminded))]
    G.add_edge(id, random_likeminded_fof, key=0, label='friend', rejections=0)
    # print("{}: I added {} as friend that is close!".format(id, random_likeminded_fof))
    data.append({ "step": step, "action":"addLink", "source": id, "target" : random_likeminded_fof, "label": "friend", "trace":"add_fof"})

def addInfoLink(self):
    #check if I have read items to begin with
    itemlinks = get_links(self.model.G, self.unique_id, 0, 'itemlink')
    if len(itemlinks) < 1:
        return

    #for each article I have read
    for item in itemlinks:
        infoLinks = get_links(self.model.G, self.unique_id, 0, 'infolink')
        #Get the people who have also read this article
        people = [x for x in get_links(self.model.G, item, 0, "itemlink") if x != self.unique_id and x not in infoLinks]
        if len(people) < 1:
            return
        
        #Pick a random person and add them as a friend!
        for person in people:
            #print("{}: I'm adding {} as an infolink for item {}".format(self.unique_id, person, item))
            self.model.G.add_edge(self.unique_id, person, key=1, label='infolink', refs=1)
            data.append({"step": self.model.current_step, "action":"addLink", "source": self.unique_id, "target" : person, "label": "infolink", "trace":"add_info_link"})

def update_infolink(G, source, destination, current_step):
    key = 1
    if G[source][destination].get(key):
        G[source][destination].get(key)['refs'] += 1
    else:
        G.add_edge(source, destination, key=key, label='infolink', refs=1)
        data.append({ "step": current_step, "action":"addInfoLink", "source": source, "target" : destination, "label": "infolink", "trace":"update_infolink"})

def remove_infolink(G, id, item, current_step):
    key = 1
    links = [x for x in get_links(G, item, 0, 'itemlink') if x != id]
    for link in links:
        if G.has_edge(link, id, key):
            if G[link][id].get(key)['refs'] > 1:
                G[link][id].get(key)['refs'] -= 1
            else:
                G.remove_edge(link,id,key)
                data.append({"step": current_step, "action": "removeInfoLink", "source": link, "target": id,"label": "infolink", "trace": "remove_infolink"})

def get_items(G, label):
    items = [x for x,y in G.nodes(data=True) if y['label'] == label]
    return items

def get_random_item(G, list_of_read_items):
    items = get_items(G, 'item')

    if not items or len(items) == 0:
        return None

    pos = random.randint(0, len(items) - 1)
    if not items[pos] in list_of_read_items:
        return items[pos]
    else:
        return None

def get_links(G, id, key, label, include_origin=False):
    """ Get links of a person with nodes containing a specific label
    :param G: The graph
    :param key: The edge key
    :param label: The label (e.g. 'itemlink','friend')
    :return: a list of links
    """
    itemlinks = []
    # print('@get_links id={}'.format(id))
    for (u, v) in G.edges(id):
        if G[u][v].get(key) and G[u][v][key]['label'] == label:
            itemlinks.append(v if include_origin == False else (u,v))
    return itemlinks

def has_link(G, id, other, key, label):
    """ Get links of a person with nodes containing a specific label
    :param G: The graph
    :param key: The edge key
    :param label: The label (e.g. 'itemlink','friend')
    :return: a list of links
    """
    itemlinks = []
    # print('@get_links id={}'.format(id))
    for (u, v) in G.edges(id):
        if G[u][v].get(key) and G[u][v][key]['label'] == label and v == other:
            return True
    return False

def get_distance(G, u, v):
    return math.sqrt((G.nodes[u]['x'] - G.nodes[v]['x']) ** 2 + (G.nodes[u]['y'] - G.nodes[v]['y']) ** 2)

def get_innate_distance(G, v):
    return math.sqrt((G.nodes[v]['x'] - G.nodes[v]['innate_x']) ** 2 + (G.nodes[v]['y'] - G.nodes[v]['innate_y']) ** 2)

def get_polarisation(G):
    items = [(b['x'], b['y']) for a,b in G.nodes(data=True) if b['label'] == 'person']
    average = get_average_xy(G)
    polarisation = [ ( (x - average[0]) ** 2, (y - average[1]) ** 2 ) for x, y in items]
    total_polarisation = tuple(map(sum, zip(*polarisation)))
    return (total_polarisation[0] + total_polarisation[1]) / 2

def get_average_xy(G):
    items = [(b['x'], b['y']) for a,b in G.nodes(data=True) if b['label'] == 'person']
    return tuple(map(mean, zip(*items)))

class PersonAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.timeline = []
        self.characteristics = {}

    def post_item(self, friend, item):
        if (friend,item) not in self.timeline:
            self.timeline.append((friend,item))

    @abc.abstractmethod
    def get_item(self, list_of_read_items):
        pass

    def get_average_latitude_infolinks_and_node(self,G):
        itemlinks = get_links(G, self.unique_id, 0, 'itemlink')
        sum_x = 0
        sum_y = 0
        count = 0
        for item in itemlinks:
            sum_x += G.nodes[item]['x']
            sum_y += G.nodes[item]['y']
            count += 1

        #adding own node position as part of the average
        sum_x += G.nodes[self.unique_id]['x']
        sum_y += G.nodes[self.unique_id]['y']
        count += 1

        return sum_x / count, sum_y / count

    def generate_characteristics(self):
        #This generates characteristics for an individual agent
        self.latitude_of_acceptance = self.model.latitude_acceptance # np.clip(np.random.normal(self.model.latitude_acceptance, 0.2), 0, 2)
        self.sharpness = self.model.sharpness # np.clip(np.random.normal(self.model.sharpness, 5), 0, 50)
        self.probability_post_social_net = self.model.probability_post_social_net # np.clip(np.random.normal(self.model.probability_post_social_net, 0.1), 0, 1)
        self.num_friends = np.floor(np.clip(np.random.normal(self.model.num_friends, 5), 1, self.model.G.number_of_nodes()))
        self.max_num_item_links = self.model.max_num_item_links # np.floor(np.clip(np.random.normal(self.model.max_num_item_links, 2), 1, 100))
        
        self.characteristics["lattitude_of_acceptance"] = self.latitude_of_acceptance
        self.characteristics["sharpness"] = self.sharpness
        self.characteristics["probability_post_social_net"] = self.probability_post_social_net
        self.characteristics["num_max_num_item_links"] = self.max_num_item_links

    def generate_friends(self, G):
        #Get a list of agents where each agent can only appear once in the list up to my allowed amount of friends
        possibleFriends = random.sample(list(G.nodes(data=False)) , int(self.num_friends))

        #If I appear in my friendslist, remove me
        if(self.unique_id in possibleFriends):
            possibleFriends.remove(self.unique_id)

        #Create a link between me and all my friends
        for new_friend in possibleFriends:
            G.add_edge(self.unique_id, new_friend, key=0, label='friend', rejections=0)

    def get_characteristics(self):
        #Use this function to retrieve the characteristics of the agent
        return self.characteristics

    def get_acceptance(self, distance):
        latitude_of_acceptance = self.characteristics["lattitude_of_acceptance"]
        sharpness = self.characteristics["sharpness"]
        return (latitude_of_acceptance ** sharpness) / (distance ** sharpness + latitude_of_acceptance ** sharpness)

    def update_latitude(self, G, x, y):
        nx.set_node_attributes(G, {self.unique_id: {'x': x, 'y': y, 'label': 'person'}})
        data.append({"step": self.model.current_step, "action": "changePosition", "id": self.unique_id, 'x': x, 'y':y})

    def step(self):
        self.timeline = self.timeline[-50:]
        start = timeit.default_timer()

        # getting the maximum number of item links
        max_num_item_links = self.characteristics["num_max_num_item_links"]

        # getting the number of items to read per day
        num_exposed_items_person_step = math.ceil(self.model.num_exposed_items_person_day / self.model.num_steps_per_day)

        # adds potential infolinks through content i've interacted with
        addInfoLink(self)

        # for each item of the day
        il=0
        gi=0
        for i in range(num_exposed_items_person_step):
            # get item links
            start_il = timeit.default_timer()
            il += timeit.default_timer() - start_il
             #All links with items the user has at the moment
            itemlinks = get_links(self.model.G, self.unique_id, 0, 'itemlink')

            # if the memory is full, forget (the oldest) item
            if len(itemlinks) >= max_num_item_links:
                self.model.G.remove_edge(self.unique_id,itemlinks[0])
                data.append({"step": self.model.current_step, "action": "removeItemLink", "source": self.unique_id, "target": itemlinks[0] ,"label": "itemlink", "trace": "step(removeitem)"})
                remove_infolink(self.model.G,self.unique_id,itemlinks[0], self.model.current_step)

            # based on the strategy , get an article
            start_gi = timeit.default_timer()
            item = self.get_item(itemlinks)

            if type(item) != list:
                items = [item]
            else:
                items = item

            for item in items:
                # if it contains social net info
                if(type(item) == tuple):
                    friend = item[0]
                    item = item[1]
                else:
                    friend = None

                gi+= timeit.default_timer() - start_gi

                if item:
                    # calculate the distance between the user and item
                    distance = get_distance(self.model.G, self.unique_id, item)

                    # get the latitude of acceptance for the item
                    acceptance = self.get_acceptance(distance)

                    if random.uniform(0,1) <= acceptance:
                        # if(item in itemlinks):
                        #     print("oops dup ",item, items, data)

                        self.model.G.add_edge(self.unique_id, item, key=0, label='itemlink')
                        data.append({"step": self.model.current_step, "action": "addItemLink", "source": self.unique_id, "target": item,"label": "itemlink", "trace": "step(additem)"})
                        # print('step - added item',item)

                        # recalculate position of user based on the average of his memory
                        x,y = self.get_average_latitude_infolinks_and_node(self.model.G)
                        self.update_latitude(self.model.G, x, y)

                        # if friend and self.model.G.has_edge(self.unique_id, friend, 0):
                        #     update_infolink(self.model.G,self.unique_id, friend, self.model.current_step)

                        # if user posts to the social network
                        if self.characteristics["probability_post_social_net"] > 0:
                            # probabilistic (yes)
                            will_post = random.uniform(0,1) <= self.characteristics["probability_post_social_net"]
                            if will_post:
                                friends = get_links(self.model.G, self.unique_id, 0, 'friend')
                                # print('len(friends)={} item={}'.format(len(friends), item))
                                # for each friend
                                for friend in friends:
                                    # post the item in the agent's timeline
                                    agent = self.model.G.nodes[friend]['agent'][0]
                                    agent.post_item(self.unique_id,item)
                    else:
                        # item was rejected (count rejections if this is a known friend that is not an infosharer
                        if friend and self.model.G.has_edge(self.unique_id,friend,0) and not self.model.G.has_edge(self.unique_id,friend,1):
                            self.model.G[self.unique_id][friend].get(0)['rejections'] += 1
                            # rejection_limit = random.uniform(1,3)
                            # user could unfriend people sharing confronting items
                            if self.model.G[self.unique_id][friend].get(0)['rejections'] >= 5 and random.uniform(0,1) <= 15:
                                # if self.model.G[self.unique_id][friend].get(0)['rejections'] >= rejection_limit:
                                # print('rejecte')
                                print("{}:I hate you, {}! Unfriended!".format(self.unique_id, friend))
                                self.model.G.remove_edge(self.unique_id,friend,0)
                                data.append({"step": self.model.current_step, "action": "removeLink","source": self.unique_id, "target": friend, "label": "friend", "trace": "step(unfriend)"})
                                add_fof(self.model.G, self.unique_id, friend, self.model.current_step, self.latitude_of_acceptance)

        # clear timeline
        self.timeline = []

        tot = timeit.default_timer() - start
        # print('@PersonAgent.step time: ', tot * 1000, il/tot, gi/tot)

class IndividualSearchPersonAgent(PersonAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def get_item(self, list_of_read_items):
        return get_random_item(self.model.G, list_of_read_items)

class RecommendationEnginePersonAgent(PersonAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def get_item(self, list_of_read_items):

        using_rs = random.uniform(0,1) <= 0.75

        if using_rs and self.model.news_item_engine.ready:
            start = timeit.default_timer()
            rec = self.model.news_item_engine.recommend_user_based(self.unique_id,10,challenge=self.model.rs_challenge)
            ellapsed = timeit.default_timer() - start
            # print("@rs getitem took {:f} s".format(ellapsed))
            if rec and len(rec) >= 1:
                if random.uniform(0,1) >= 0.5: # flipping a coin
                    # print('rec',rec)
                    for i in range(len(rec)):
                        pos = random.randint(0, len(rec) - 1)
                        if not rec[pos] in list_of_read_items:
                            # print('an item was recommended', rec[pos], list_of_read_items)
                            return rec[pos]
                    return None
                else:
                    return None
            else:
                return None
        else:
            # getting a random item to read
            items = get_items(self.model.G, 'item')

            if not items or len(items) == 0:
                return None
            pos = random.randint(0, len(items)-1)
            if not items[pos] in list_of_read_items:
                # print('an item was randomly picked', items[pos], list_of_read_items)
                return items[pos]
            else:
                return None

class IndividualSearchSocialNetPersonAgent(PersonAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def get_item(self, list_of_read_items):

        # flipping a coin
        if random.uniform(0, 1) >= 0.2: # read random item
            start = timeit.default_timer()
            ri = get_random_item(self.model.G, list_of_read_items)
            ellapsed = timeit.default_timer() - start
            # print("@IndividualSearchSocialNetPersonAgent.get_item random took {} s".format(ellapsed))
            return ri
        else: # read social network timeline item
            # print('reading timeline {}'.format(len(self.timeline)))
            if len(self.timeline) == 0:
                return None
            start = timeit.default_timer()
            items = []
            random.shuffle(self.timeline)
            for item in self.timeline:
                if not item[1] in list_of_read_items:
                    items.append(item)
                    break
            ellapsed = timeit.default_timer() - start
            # print("@IndividualSearchSocialNetPersonAgent.get_item timeline took {} s".format(ellapsed))
            return items

class RSSocialNetPersonAgent(RecommendationEnginePersonAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def get_item(self, list_of_read_items):

        # flipping a coin
        if random.uniform(0, 1) >= 0.5: # read rs item
            return super().get_item(list_of_read_items)
        else: # read social network timeline item
            # print('reading timeline {}'.format(len(self.timeline)))
            if len(self.timeline) == 0:
                return None
            start = timeit.default_timer()
            items = []
            random.shuffle(self.timeline)
            for item in self.timeline:
                if not item[1] in list_of_read_items:
                    # print('get_item(RSSocialNetPersonAgent) adding item',item, self.timeline, list_of_read_items)
                    items.append(item)
                    break
            ellapsed = timeit.default_timer() - start
            # print("@IndividualSearchSocialNetPersonAgent.get_item timeline took {} s".format(ellapsed))
            return items

class NewsItemEngine(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.current_step = 0
        self.item_count = 0
        self.ready = False

    def build_cf_rs(self,read_log):
        if not read_log:
            return
        ratings = pd.DataFrame(((x,y,5) for x,y in read_log),columns=['user_id','item_id','rating'])

        if self.model.rs_alg == 1:
            self.user_item_m = ratings.pivot('user_id', 'item_id', 'rating').fillna(0)
            self.X_user = cosine_similarity(self.user_item_m)
            self.X_item = cosine_similarity(self.user_item_m.T)

        elif self.model.rs_alg == 2:
            reader = Reader(rating_scale=(0, 5))
            data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

            trainset = data.build_full_trainset()
            algo = SVD()
            algo.fit(trainset)

            testset = trainset.build_anti_testset()
            predictions = algo.test(testset)

            self.top_n = get_top_n(predictions, n=10)
            # print('@build_cf_rs top_n={}'.format(self.top_n))

        self.ready = True

    def recommend_user_based(self, user, top_n,challenge=False):
        if not self.ready:
            return []

        if challenge and random.uniform(0,1) <= 1:
            for i in range(50):
                item = get_random_item(self.model.G,[])
                distance = get_distance(self.model.G, user, item)
                if distance > self.model.latitude_acceptance: ## full challenge
                # if distance < self.model.latitude_acceptance and distance > 0.20 and distance < 0.25:
                # if distance < self.model.latitude_acceptance and (self.model.latitude_acceptance - distance) / self.model.latitude_acceptance < 0.05:
                #     print('challenging person: dist={}, lat={}, dif={}, prop={}'.format(distance, self.model.latitude_acceptance,(
                #             self.model.latitude_acceptance - distance),
                #             (self.model.latitude_acceptance - distance) / self.model.latitude_acceptance  ))
                    return [item]

        if self.model.rs_alg == 1:
            try:
                ix = self.user_item_m.index.get_loc(user)
                # Use it to index the User similarity matrix
                u_sim = self.X_user[ix]
                # obtain the indices of the top k most similar users (2d array with position and value )
                d = np.array([(a, b) for a, b in enumerate(u_sim) if b > 0])
                # sorting by similarity
                d = d[d[:, 1].argsort()[::-1]]

                most_similar = self.user_item_m.index[d.T[0].astype(int)]
                # Obtain the mean ratings of those users for all items
                rec_items = self.user_item_m.loc[most_similar].mean(0).sort_values(ascending=False)

                # Discard already seen items
                # already seen items
                seen_mask = self.user_item_m.loc[user].gt(0)
                seen = seen_mask.index[seen_mask].tolist()
                rec_items = rec_items.drop(seen).head(top_n)
                rec_items = rec_items[rec_items > 0]
                # return recommendations - top similar users rated items
                return (rec_items.index.to_frame()
                        .reset_index(drop=True)
                        .head(top_n)['item_id'].to_list())
            except KeyError:
                return None
        elif self.model.rs_alg == 2:
            items = [x[0] for x in self.top_n[user]]
            # print('@recommend_user_based items={}'.format(items))
            return items

    def generate_items(self):
        self.current_step += 1
        # print('current step={}'.format(self.current_step))
        number_of_items = math.ceil(self.model.num_items_per_day / self.model.num_steps_per_day)
        for i in range(number_of_items):
            # add the item as a graph node
            self.item_count += 1
            item_id = self.model.num_agents + self.item_count
            # print('itemid: {}'.format(item_id))
            self.model.G.add_node(item_id)
            my_x = random.uniform(-1,1)
            my_y = random.uniform(-1,1)
            nx.set_node_attributes(self.model.G, {item_id:{'x': my_x, 'y': my_y, 'label': 'item', 'step': self.current_step}})
            data.append({"step": self.model.current_step, "action": "addItem", "id": item_id, "x": my_x, "y": my_y ,"trace": "generate_items"})

    def update_rs_cf(self):
        start = timeit.default_timer()
        read_log = list(itertools.chain.from_iterable(
            get_links(self.model.G, a, 0, 'itemlink', True) for a, b in self.model.G.nodes.data() if b['label'] == 'person'))
        ellapsed = timeit.default_timer() - start
        # print("@update_rs_cf read_log took {} s".format(ellapsed))

        start = timeit.default_timer()
        self.build_cf_rs(read_log)
        ellapsed = timeit.default_timer() - start
        # print("@update_rs_cf build_cf_rs took {} s".format(ellapsed))


    def clean_up(self):
        items_to_remove = [a for a, b in self.model.G.nodes(data=True) if b['label'] == 'item' and b['step'] <= self.current_step - 10 and self.model.G.degree[a] == 0]
        # print('@clean_up len(items_to_remove)={} {}'.format(len(items_to_remove),items_to_remove))
        if items_to_remove:
            for n in items_to_remove:
                # print('docs before={}'.format(get_items(self.model.G,'item')))
                self.model.G.remove_node(n)
                # print('docs after={}'.format(get_items(self.model.G, 'item')))

    def step(self):
        self.generate_items()

        # cleanup?
        # if self.current_step % 10 == 0:
        #     print('running cleaning for step {}'.format(self.current_step))
        #     self.clean_up()

        if self.model.user_behaviour == "Recommendation":
            self.update_rs_cf()


class FBModel(Model):
    def __init__(self, network, N, num_friends, num_steps_per_day, num_items_per_day, max_num_item_links,
                 num_exposed_items_person_day, latitude_acceptance, sharpness, probability_post_social_net, 
                 user_behaviour, rs_alg, rs_challenge, hide_friends, hide_itemlinks):
        super().__init__(self)

        data = []

        np.random.seed(958)
        random.seed(958)

        # storing parameters
        self.network = network
        self.running = True
        self.num_agents = N
        self.num_friends = num_friends
        self.num_steps_per_day = num_steps_per_day
        self.num_items_per_day = num_items_per_day
        self.max_num_item_links = max_num_item_links
        self.num_exposed_items_person_day = num_exposed_items_person_day
        self.latitude_acceptance = latitude_acceptance
        self.sharpness = sharpness
        self.probability_post_social_net = probability_post_social_net
        self.user_behaviour = user_behaviour
        self.rs_alg = rs_alg
        self.rs_challenge = rs_challenge
        self.hide_friends = hide_friends
        self.hide_itemlinks = hide_itemlinks

        # initialising control variables
        self.current_step = 0

        # create the users network
        self.build_users_network()

        # preparing the grid and schedule
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)

        # creating people
        for i, node in enumerate(self.G.nodes()):
            if self.user_behaviour == 'Recommendation':
                a = RecommendationEnginePersonAgent(i, self)
                a.generate_characteristics()
                a.generate_friends(self.G)
            elif self.user_behaviour == 'Individual_Search':
                a = IndividualSearchPersonAgent(i, self)
                a.generate_characteristics()
                a.generate_friends(self.G)
            elif self.user_behaviour == 'Individual_Search_Social_Network':
                a = IndividualSearchSocialNetPersonAgent(i, self)
                a.generate_characteristics()
                a.generate_friends(self.G)
            elif self.user_behaviour == 'RS_Social_Network':
                a = RSSocialNetPersonAgent(i, self)
                a.generate_characteristics()
                a.generate_friends(self.G)

            self.schedule.add(a)
            self.grid.place_agent(a, node)

        # creating news item engine
        self.news_item_engine = self.create_news_item_engine()

        # configuring the datacollector
        self.datacollector = DataCollector(
                                model_reporters={
                                    "dist_between_friends": calculate_dist_between_friends,
                                    "dist_between_users_and_items": calculate_dist_between_users_and_items,
                                    "dist_between_sharers": calculate_dist_between_sharers,
                                    "innate_dist": calculate_average_innate_distance
                                    # "polarisation": calculate_polarisation,
                                    # "sub_graphs": count_sub_graphs,
                                    # "communities": detect_comm
                                }, agent_reporters={})

    def create_news_item_engine(self):
        news_item_id = self.G.number_of_nodes() # adding the agent at the bottom of the network list
        news_item_engine = NewsItemEngine(news_item_id,self)

        self.G.add_node(news_item_id)
        nx.set_node_attributes(self.G,{news_item_id: {'x': 0, 'y': 0, 'label': 'news_engine', 'agent':[]}})

        self.schedule.add(news_item_engine)
        self.grid.place_agent(news_item_engine, news_item_id)

        return news_item_engine

    def build_network_from_file(self, network_file_name):

        net_file = csv.reader(open('data/' + network_file_name,"r"), delimiter=' ')
        next(net_file)
        content = next(net_file)
        num_nodes = 300 # int(content[0]) + 1
        self.num_agents = num_nodes
        # print(num_nodes)
        # G = nx.erdos_renyi_graph(n=num_nodes, p=0.05)
        G = nx.erdos_renyi_graph(n=num_nodes, p=0)
        # print('g',G.number_of_nodes())
        for line in net_file:
            x, y = line[:2]
            x = int(x)
            y = int(y)
            if x < 300 and y < 300:
                # print(x,y, G.number_of_nodes())
                G.add_edge(x,y)
        return G

    def build_users_network(self):
        if self.network == 'predefined':
            G = nx.erdos_renyi_graph(n=self.num_agents, p=0)
        else:
            G = self.build_network_from_file(self.network)

        # updating nodes position
        for node in G:
            x = random.uniform(-1,1)
            y = random.uniform(-1,1)
            # print(node)
            nx.set_node_attributes(G, {node:{'x': x, 'y': y, 'label': 'person','innate_x': x, 'innate_y': y}})

        # from graph to MultiGraph
        self.G = nx.MultiGraph(G)
        self.G.clear_edges()

        # transfering edges to the multigraph
        for u, v in G.edges():
            self.G.add_edge(u, v, label='friend',rejections=0)

        # print('network {}'.format(json_graph.node_link_data(self.G)))

        # with open("/Users/mfknr/graph.json", "w") as outfile:
        #     print(json_graph.node_link_data(self.G),file=outfile)

    def step(self):
        # allow agents to run through the scheduler
        self.current_step += 1
        start = timeit.default_timer()
        self.schedule.step()
        time_taken = timeit.default_timer() - start
        if self.current_step % 1 == 0:
            # print('s: {} took {} s'.format(self.current_step, time_taken))
            self.datacollector.collect(self)

        # if self.current_step == 100:
        #     with open("/Users/mfknr/actions.json","w") as outfile:
        #         json.dump(data, outfile)

def calc_avg_dist_links_all_nodes(source_label, target_key, target_label, G):
    dist_links = [calc_avg_dist_links(a, target_key, target_label, G) for a, b in G.nodes.data() if b['label'] == source_label]
    if dist_links:
        return statistics.mean(dist_links)
    else:
        return 0

def calc_avg_dist_links(origin, target_key, target_label, G):
    distances = [get_distance(G, origin, i) for i in get_links(G, origin, target_key, target_label)]
    if not distances:
        return 0
    else:
        return statistics.mean(distances)

def calculate_dist_between_friends(model):
    return calc_avg_dist_links_all_nodes('person',0, 'friend', model.G)

def calculate_dist_between_sharers(model):
    return calc_avg_dist_links_all_nodes('person',1, 'infolink', model.G)

def calculate_dist_between_users_and_items(model):
    return calc_avg_dist_links_all_nodes('person',0, 'itemlink', model.G)

def calculate_average_innate_distance(model):
    distances = [get_innate_distance(model.G, v) for v in get_items(model.G, 'person')]
    if not distances:
        return 0
    else:
        return statistics.mean(distances)

def calculate_polarisation(model):
    return get_polarisation(model.G)

def count_sub_graphs(model):


    sub_graphs = nx.connected_components(h)
    count = 0
    while True:
        try:
            next(sub_graphs)
            count += 1
        except StopIteration:
            break
    return count

def detect_comm(model):
    G = model.G.copy()
    to_remove = [x for x,y in G.nodes(data=True) if y['label'] == 'item']
    G.remove_nodes_from(to_remove)

    communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
    return len(communities)

# def detect_comm(model):
#
#     G = model.G.copy()
#     to_remove = [x for x,y in G.nodes(data=True) if y['label'] == 'item']
#     # to_remove = [node for node,degree in dict(G.degree()).items() if degree == 0]
#     print('to_remove',to_remove)
#     G.remove_nodes_from(to_remove)
#
#     communities = girvan_newman(G)
#
#     node_groups = []
#     for com in next(communities):
#         print('list(com)',list(com))
#         node_groups.append(list(com))
#
#     print('len(node_groups)',len(node_groups))
#
#     color_map = []
#     for node in G:
#         if node in node_groups[0]:
#             color_map.append('blue')
#         elif node in node_groups[1]:
#             color_map.append('green')
#     nx.draw(G, node_color=color_map, with_labels=True)
#     plt.show()
#
#     return len(node_groups)