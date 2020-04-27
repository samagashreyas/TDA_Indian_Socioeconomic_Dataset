"""
Mapper analysis of Statewise socio-economic data
"""

from io import BytesIO
import sys
import base64

try:
    from scipy.misc import imsave, toimage
except ImportError as e:
    print("imsave requires you to install pillow. Run `pip install pillow` and then try again.")
    sys.exit()

try:
    import pandas as pd
except ImportError as e:
    print("pandas is required for this example. Please install with `pip install pandas` and then try again.")
    sys.exit()

import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import umap
import csv



df = pd.read_csv("../datasets/data_mapper.csv")
feature_names = [c for c in df.columns if c!="State"]


# The data
y = np.array(df["State"])
X = np.array(df[feature_names])  # quick and dirty imputation
#scaled=MinMaxScaler().fit_transform(X)
print("shape:",  X.shape)

# Call Kmapper
mapper = km.KeplerMapper(verbose=3)

# We create a custom 1-D lens. These are the four different filter functions.
#lens1 = mapper.fit_transform(X, projection=sklearn.manifold.Isomap())
#lens2 = mapper.fit_transform(X, projection="l2norm")
#lens3 = mapper.fit_transform(X, projection=PCA(n_components=2, random_state=1))
lens4 = mapper.fit_transform(X, projection=umap.UMAP(n_components=2, random_state=1))

#One can also combine two filter functions to get a 2D filter function.
#lens = np.c_[lens1, lens2]

# Create the simplicial complex
graph = mapper.map(lens=lens4, X=X, 
							   clusterer=sklearn.cluster.DBSCAN(eps=0.5, min_samples=1), 
							   cover=km.Cover(10, 0.60), remove_duplicate_nodes=True)

#Visualization
mapper.visualize(graph,
                 path_html="states_4_indices_umap_di.html",
                title="States data visualization", color_function=np.array(df["DI"]), nbins=10, custom_tooltips=y)

