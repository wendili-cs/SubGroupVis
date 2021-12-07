import pandas as pd 
import numpy as np
import itertools
from sklearn import preprocessing
import random

def random_categories(categories, num = 3):
  N = len(categories)
  index_array = np.random.choice(N, replace = False, size = num)
  default_list = []
  for i in index_array:
    default_list.append(categories[i])
  return default_list


def standardizer(df, numerics):
  numeric_segment = df[numerics]
  numeric_segment_copy = df[numerics].to_numpy()
  normalizer = preprocessing.StandardScaler().fit(numeric_segment_copy)
  normalized_numeric_segment = pd.DataFrame(normalizer.transform(numeric_segment_copy),
                                          columns = numerics)
  return normalized_numeric_segment

def get_subgroups(df: pd.DataFrame, categories: list):
  '''
  args:
    df: dataset dataframe
    categories: a list of names of selected categories
  returns: a list subgroups
  '''
  cat_list = []
  for c in categories:
    cat_list.append(set(df[c]))
  subgroups = sorted(list(itertools.product(*cat_list)))
  subgroups_dict = {}
  for sg in subgroups:
    idx = pd.Series(np.ones([len(df)], dtype=bool))
    for i, feat in enumerate(sg):
      idx = idx & (df[categories[i]]==feat)
    subgroups_dict[sg] = pd.Index(idx)
  return subgroups_dict

def retrieve_levels(data, categories):
  '''
  Retrieve the levels of categorical variables
  A rather silly way though
  args:
    data: the dataset
    categories: the full list of categorical variables
  return:
    a dictionary containing levels.
  '''
  level_dict = {}
  for c in categories:
    level_dict[c] = data[c].unique()
  return level_dict