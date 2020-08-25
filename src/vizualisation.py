import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def dataloader():
    raw_df = pd.read_csv('./dataset/train.csv')
    #print(raw_df.head())
    #print(raw_df.loc[:, raw_df.columns != 'Id'])
    clean_df = raw_df[raw_df.bare_nuclei>=1]
    clean_df = clean_df.reset_index(drop=True)
    return clean_df

def correlation():
    correlation = dataloader(). drop('Id', axis=1).corr(method='pearson')
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
    plt.title("Correlation Matrix")
    plt.show()

def sum():
    df = dataloader()
    sum = df.loc[df['class'] == 4].sum()

    return sum

def pcanalysis():
    df = dataloader()
    df = df.loc[:, df.columns != 'Id']
    pca = PCA(n_components=1)
    pca.fit(df)
    return pca.components_
