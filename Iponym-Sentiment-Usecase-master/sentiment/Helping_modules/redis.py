import redis
import pyarrow as pa
from django.conf import settings
from pandas.tests.test_downstream import df

context = pa.default_serialization_context()


# redis_instance = redis.StrictRedis(host=settings.REDIS_HOST, port= settings.REDIS_PORT, db=0)

def connect_redis():
    conn = redis.Redis()
    return conn


def in_redis(cache_var):
    redis_conn = connect_redis()
    redis_flag = redis_conn.exists(cache_var)
    return redis_flag


def fetch_from_redis(cache_var='women_reviews'):
    redis_conn = connect_redis()
    df = context.deserialize(redis_conn.get(cache_var))
    return df


def store_in_redis(df, cache_var='women_reviews'):
    # connect with redis#
    redis_conn = connect_redis()
    redis_conn.set(cache_var, context.serialize(df).to_buffer().to_pybytes())
