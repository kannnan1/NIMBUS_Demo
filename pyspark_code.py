import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
spark = SparkSession.builder.getOrCreate()
 
f = Functions()
 
df = f.get_args("df")
 
psdf = ps.from_pandas(df)
 
table = psdf.describe().to_pandas().rename_axis('Metric').reset_index()
 
plt = psdf['cltv'].plot.box()
 
f.save_table(table, name="Table")
 
f.save_graph(plt)
# your code here
