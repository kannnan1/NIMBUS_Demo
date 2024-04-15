
import pandas as pd
import numpy as np
import plotly.express as px
from Console.package import Functions
from sklearn import datasets
from sklearn.decomposition import PCA


f = Functions()

iris = datasets.load_iris()
iris = pd.DataFrame(iris.data)
iris = iris.rename(columns = {'0':'SepalLengthCm','1':'SepalWidthCm','2':'PetalLengthCm','3':'PetalWidthCm'})

f.save_table(iris, name="IRIS DataSet")
summary = iris.describe()
f.save_table(summary, name="Summary Statistics")
