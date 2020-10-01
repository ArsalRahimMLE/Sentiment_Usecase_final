import pymongo
import pandas as pd
from pymongo import MongoClient
import scipy as sp
import numpy as np
import redis
from .redis import in_redis, fetch_from_redis, store_in_redis


def establish_conn(coll_name='women_reviews'):
    client = pymongo.MongoClient()
    db = client['review_data']
    coll = db[coll_name]
    return coll


def fetch_data(collection='women_reviews'):
    if in_redis(collection):
        # fetch from redis #
        df = fetch_from_redis(collection)
    else:
        conn = establish_conn(collection)
        df = conn.find({}, {"_id": 0})
        df = pd.DataFrame(list(df))
        store_in_redis(df, collection)
    return df
