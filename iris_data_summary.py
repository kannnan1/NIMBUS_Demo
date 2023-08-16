
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

print(iris)
print(iris.describe())
