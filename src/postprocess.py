import pandas
import plotly.express as px
from PIL import Image

pos_weights = pandas.read_csv("results/weights.csv")
all = pandas.read_csv("datasets/combined.csv")

all.loc[:, "weight"] = pos_weights
all.to_csv("results/final.csv", index=False)
