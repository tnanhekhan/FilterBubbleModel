from matplotlib import pyplot as plt
from mesa.batchrunner import BatchRunner
import timeit

from fb_model import FBModel, calculate_dist_between_friends, calculate_dist_between_users_and_items, \
    calculate_dist_between_sharers, calculate_average_innate_distance

fixed_params = {'N': 200,
                'network': 'predefined', #Haverford76.mtx
                'num_friends': 16,
                'num_steps_per_day': 1,
                'num_items_per_day': 20,
                'max_num_item_links': 20,
                'num_exposed_items_person_day': 20,
                'latitude_acceptance': 0.3,
                'sharpness': 13,
                'probability_post_social_net': 0.1,
                'user_behaviour': 'Recommendation', # , Individual_Search, Individual_Search_Social_Network, RS_Social_Network, Recommendation
                'rs_alg':2,
                'rs_challenge': False}

batch_run = BatchRunner(FBModel,
                        None,
                        fixed_params,
                        iterations=1,
                        max_steps=500,
                        model_reporters={
                                    "dist_between_friends": calculate_dist_between_friends,
                                    "dist_between_users_and_items": calculate_dist_between_users_and_items,
                                    "dist_between_sharers": calculate_dist_between_sharers,
                                    "innate_dist": calculate_average_innate_distance

                        })

print("params",fixed_params)

start = timeit.default_timer()

batch_run.run_all()

print('took {}'.format(timeit.default_timer() - start))

run_data = batch_run.get_model_vars_dataframe()

for index, row in run_data.iterrows():
    print (row["dist_between_friends"], row["dist_between_users_and_items"],row['dist_between_sharers'], row['innate_dist'])