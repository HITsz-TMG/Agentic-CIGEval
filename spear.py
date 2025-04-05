from util import *

human_ratings = "path_to_ImagenHub_human_eval_results"


def read_human_sc(task, model, sample):
    import pandas as pd
    import numpy as np
    import ast

    mm = task.replace("ImagenHub_","")

    df1 = pd.read_csv(os.path.join(human_ratings,f"{task}/{mm}_rater1.tsv"), sep="\t")
    df2 = pd.read_csv(os.path.join(human_ratings,f"{task}/{mm}_rater1.tsv"), sep="\t")
    df3 = pd.read_csv(os.path.join(human_ratings,f"{task}/{mm}_rater1.tsv"), sep="\t")

    cell_value_1 = df1.loc[df1['uid'] == sample, model].values
    cell_value_2 = df2.loc[df2['uid'] == sample, model].values
    cell_value_3 = df3.loc[df3['uid'] == sample, model].values

    sc_1 = ast.literal_eval(cell_value_1[0])[0]
    sc_2 = ast.literal_eval(cell_value_2[0])[0]
    sc_3 = ast.literal_eval(cell_value_3[0])[0]
    
    
    return np.mean([sc_1,sc_2,sc_3])
    # return [sc_1,sc_2,sc_3]


def preprocess(_list):
    temp_list = []
    for scores in _list:
        if isinstance(scores, (int, float)):
            temp_list.append(map_to_nearest_higher(scores/10.0))
        else:
            scores = [int(score) for score in scores]
            # temp_list.append(map_to_nearest_higher(min(scores)))
            temp_list.append(map_to_nearest_higher(min(scores)/10.0))
    return temp_list


def sigfig(number, sigfigs=4, digit_mode=True):
    if digit_mode:
        string_mode = '{:#.{sigfigs}f}'
    else:
        string_mode = '{:#.{sigfigs}g}'
    if isinstance(number, list):
        new_numbers = []
        for num in number:
            new_num = string_mode.format(num, sigfigs=sigfigs)
            new_numbers.append(float(new_num))
        return new_numbers
    else:
        return float(string_mode.format(number, sigfigs=sigfigs))


def map_to_nearest_higher(number, target_numbers=[0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0], not_mapping=True):
    if not_mapping:
        if number > 1.0:
            return 1.0
        if number < 0.0:
            return 0.0
        return number
    
    # Find the nearest higher number
    for target in target_numbers:
        if target >= number:
            return target
    return target_numbers[-1]  # Return the maximum if no higher number is found


def average_correlation(z_scores):
    import math
    # Calculate the average Z score
    z_avg = sum(z_scores) / len(z_scores)

    # Convert the average Z score back to a correlation coefficient
    r_avg = (math.exp(2 * z_avg) - 1) / (math.exp(2 * z_avg) + 1)
    return r_avg




from scipy.stats import spearmanr
import numpy as np
import ast
task=""
model=""
# Read the identifiers of the evaluation images under each task/model and organize them into a list
keys=[]
# Read the automated evaluation results and organize them into a list
SC_gpt4o = []

SC_human = [read_human_sc(task, model, key) for key in keys]
SC_rho, _ = spearmanr(SC_gpt4o, SC_human)
print(task, model, "SC|", sigfig(SC_rho))
