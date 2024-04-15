import pandas as pd
import numpy as np
import plotly.express as px
from Console.package import Functions
 
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
 
conf = SparkConf().set('spark.executor.memory', '1G')
spark = SparkSession \
    .builder \
    .config(conf=conf) \
    .appName("Summary Statistic") \
    .getOrCreate()
 
f = Functions()
 
df = f.get_args("df")
 
psdf = ps.from_pandas(df)
 
table = psdf.describe().to_pandas().rename_axis('Metric').reset_index()
 
plt = psdf['cltv'].plot.box()
 
print(table)
 
f.save_table(table, name="Table")
 
f.save_graph(plt, name="Graph")
# your code here
