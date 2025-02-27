#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[107]:


sheet1_data = pd.read_excel("Merged_F1 (music_emotion_featureextraxted dataset).xlsx", sheet_name='Sheet1')


# In[108]:


sheet1_data.head()


# In[109]:


def get_base_features_timbre():
    # Define base feature names for selecting the appropriate columns
    return [
        'Spectral Rolloff Point Overall Average',
        'Zero Crossings Overall Average', 
        'Spectral Flux Overall Average',
        'Spectral Centroid Overall Average',
    ]

def get_base_features_rhythmic():
    # Define base feature names for selecting the appropriate columns
    return [
        'Strength Of Strongest Beat Overall Average',
        'Strongest Beat Overall Average', 
        'Root Mean Square Overall Average',
    ]

def get_base_features_dynamics():
    # Define base feature names for selecting the appropriate columns
    return [
        'Root Mean Square Overall Average',
        'Fraction Of Low Energy Windows Overall Average', 
    ]

def heatmap(emotion_feature_correlation, title=""):
    # Plot heatmap for better visualization
    plt.figure(figsize=(20, 11))
    # sns.heatmap(emotion_feature_correlation, annot=True, cmap='coolwarm', fmt=".2f")
    sns. heatmap(emotion_feature_correlation, cmap="YlGnBu", annot=True)

    plt.title(f'Correlation Between {title}')
    return plt.show()


# In[110]:



# Select emotion and feature columns (all features)
def get_main_columns_all(sheet1_data, deviation = False, allcontent = False):
    emotion_columns = ['Joyful', 'Happy', 'Amusing', 'Energizing', 'Dreamy', 'Relaxing', 'Neutral', 'Sad', 'Annoying', 'Anxious']
    
    if deviation:
        if allcontent:
            feature_columns = [
                col for col in sheet1_data.columns 
                if col not in ['fileName', 'SongID', 'Genre', 'SongName'] + emotion_columns
            ]
        else:
            feature_columns = [
                col for col in sheet1_data.columns 
                if col not in ['fileName', 'SongID', 'Genre', 'SongName'] + emotion_columns and 'Overall Average' not in col
            ]
        
    else:
        feature_columns = [
            col for col in sheet1_data.columns 
            if col not in ['fileName', 'SongID', 'Genre', 'SongName'] + emotion_columns and 'Deviation' not in col
        ]

    return emotion_columns, feature_columns

# Function to select emotion and feature columns (specific base features and corresponding deviations)
def get_main_columns_each(base_features, sheet1_data, deviation=False):
    emotion_columns = ['Joyful', 'Happy', 'Amusing', 'Energizing', 'Dreamy', 'Relaxing', 'Neutral', 'Sad', 'Annoying', 'Anxious']
    
    if deviation:
        # Include both base features and their corresponding deviations
        feature_columns = [
            col for base in base_features
            for col in sheet1_data.columns
            if col == base or col == base.replace('Overall Average', 'Overall Standard Deviation')
        ]
    else:
        # Only include base features
        feature_columns = [
            col for col in sheet1_data.columns 
            if col in base_features and col not in ['fileName', 'SongID', 'Genre', 'SongName'] + emotion_columns
        ]

    return emotion_columns, feature_columns


# Function to compute the correlation matrix between emotions and features
def correlation_matrix(sheet1_data, emotion_columns, feature_columns):
    # Ensure that only numeric data is passed for correlation computation
    numeric_columns = emotion_columns + feature_columns
    numeric_data = sheet1_data[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN
    
    # Drop rows with NaN values
    numeric_data = numeric_data.dropna()

    # Compute the correlation matrix between emotions and features
    correlation_matrix = numeric_data.corr()
    # correlation_matrix = numeric_columns.corr()
    return correlation_matrix

# Function to plot the heatmap for better visualization
def heatmap(correlation_matrix, title=""):
    plt.figure(figsize=(20, 18))  # Adjust figure size as needed
    sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
    plt.title('Correlation Between ' + title)
    plt.show()


# In[111]:


#Using all features (All features):
emotion_columns, feature_columns = get_main_columns_all(sheet1_data)
emotion_columnsD, feature_columnsD = get_main_columns_all(sheet1_data,deviation=True)
# Compute correlation matrix for all emotions and all features combined
correlations_full = correlation_matrix(sheet1_data, emotion_columns, feature_columns)
correlations_fullD = correlation_matrix(sheet1_data, emotion_columnsD, feature_columnsD)
# Visualize the correlations using a heatmap
heatmap(correlations_full, title="Emotions and All Features")
heatmap(correlations_fullD, title="Emotions and All Standard Deviation Features")


# In[112]:


# 1. Using base features (Timbre features):
base_features_timbre = get_base_features_timbre()
emotion_columns_timbre, feature_columns_timbre = get_main_columns_each(base_features_timbre, sheet1_data)
emotion_columns_timbreD, feature_columns_timbreD = get_main_columns_each(base_features_timbre, sheet1_data,deviation=True)
# Compute correlation matrix for all emotions and timbre features combined
correlations_full = correlation_matrix(sheet1_data, emotion_columns_timbre, feature_columns_timbre)
correlations_fullD = correlation_matrix(sheet1_data, emotion_columns_timbreD, feature_columns_timbreD)
# Visualize the correlations using a heatmap
heatmap(correlations_full, title="Emotions and Timbre Features")
heatmap(correlations_fullD, title="Emotions and Timbre Features with Standard Deviation")


# In[113]:


# 2. Using base features (Rhythmic features):
base_features_rhythmic = get_base_features_rhythmic()
emotion_columns_rhythmic, feature_columns_rhythmic = get_main_columns_each(base_features_rhythmic, sheet1_data)
emotion_columns_rhythmicD, feature_columns_rhythmicD = get_main_columns_each(base_features_rhythmic, sheet1_data,deviation=True)
# Compute correlation matrix for all emotions and timbre features combined
correlations_full = correlation_matrix(sheet1_data, emotion_columns_rhythmic, feature_columns_rhythmic)
correlations_fullD = correlation_matrix(sheet1_data, emotion_columns_rhythmicD, feature_columns_rhythmicD)
# Visualize the correlations using a heatmap
heatmap(correlations_full, title="Emotions and Rhythmic Features")
heatmap(correlations_fullD, title="Emotions and Rhythmic Features with Standard Deviation")


# In[114]:


# 3. Using base features (Rhythmic features):
base_features_dynamics = get_base_features_dynamics()
emotion_columns_dynamics, feature_columns_dynamics = get_main_columns_each(base_features_dynamics, sheet1_data)
emotion_columns_dynamicsD, feature_columns_dynamicsD = get_main_columns_each(base_features_dynamics, sheet1_data,deviation=True)
# Compute correlation matrix for all emotions and timbre features combined
correlations_full = correlation_matrix(sheet1_data, emotion_columns_dynamics, feature_columns_dynamics)
correlations_fullD = correlation_matrix(sheet1_data, emotion_columns_dynamicsD, feature_columns_dynamicsD)
# Visualize the correlations using a heatmap
heatmap(correlations_full, title="Emotions and Rhythmic Features")
heatmap(correlations_fullD, title="Emotions and Rhythmic Features with Standard Deviation")


# In[ ]:


import librosa
import numpy as np

# Load the audio file
audio_path = '1.mp3'
y, sr = librosa.load(audio_path, sr=None)  # Load the audio file with the original sample rate

# Perform STFT on the audio (this would require n_fft and hop_length, which we want to identify)
D = librosa.stft(y)

# Get the number of frequency bins and the number of frames (time steps)
n_fft = D.shape[0] * 2 - 2  # The number of frequency bins is half the FFT size, minus 2 for symmetry
n_frames = D.shape[1]  # The number of time steps (frames)

# Estimate hop size based on the audio length and number of frames
hop_length = len(y) // n_frames

print(f"Identified Frame Size (n_fft): {n_fft} samples")
print(f"Identified Hop Size: {hop_length} samples")


# In[ ]:




