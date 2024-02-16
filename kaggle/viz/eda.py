import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_theme(style="whitegrid", palette="muted", context="talk", font_scale=1.2)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})

import warnings
warnings.filterwarnings("ignore")

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## EDA

def plot_categorical(data, column_name):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data[column_name].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title(column_name)
    ax[0].set_ylabel('')
    sns.countplot(x=column_name, data=data, ax=ax[1])
    ax[1].set_title(column_name)
    plt.show()
    
def plot_correlation_heatmap(df):
    corr = df.corr()
    mask = np.triu(corr)
    plt.figure(figsize=(15, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f")
    plt.show()
    
    
def plot_pairplot(data, numerical_cols, target_col):
    pairplot = sns.pairplot(data=data[numerical_cols + [target_col]], 
                            hue=target_col, 
                            corner=True, 
                            plot_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'k'},
                            palette='Set1', 
                            diag_kws={'edgecolor':'k'})
    pairplot.fig.suptitle("Pairplot of Numerical Variables", y=1.02)
    plt.show()
    
def plot_boxplot(data, numerical_col, target_col):
    sns.boxplot(x=target_col, y=numerical_col, data=data)
    plt.title(f'Box Plot of {numerical_col} by {target_col}')
    plt.show()

    
def plot_violinplot(data, numerical_col, target_col):
    sns.violinplot(x=target_col, y=numerical_col, data=data)
    plt.title(f'Violin Plot of {numerical_col} by {target_col}')
    plt.show()
    
def plot_histograms(data, continuous_vars, target_col):
    for column in continuous_vars:
        if data[column].dtype == 'float16':
            data[column] = data[column].astype('float32')

        fig, ax = plt.subplots(figsize=(18, 4))
        sns.histplot(data=data, x=column, hue=target_col, bins=50, kde=True)
        plt.show()

def plot_countplot(data, column_name):
    sns.countplot(x=column_name, data=data)
    plt.title(f'Count Plot of {column_name}')
    plt.show()