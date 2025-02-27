#!/usr/bin/env python
# coding: utf-8

# Imports

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Read Data

# In[2]:


def read_data():
    return pd.read_excel("Merged_F1 (music_emotion_featureextraxted dataset).xlsx", sheet_name='Sheet1')


# In[3]:


def show_head(sheet1_data):
    return sheet1_data.head()


# Main Columns

# In[4]:


def get_main_columns(sheet1_data, deviation=False):
    # Select emotion columns and audio feature columns for correlation
    emotion_columns = ['Joyful', 'Happy', 'Amusing', 'Energizing', 'Dreamy', 'Relaxing','Neutral','Sad','Annoying','Anxious']
    if deviation:
        feature_columns = [col for col in sheet1_data.columns if col not in ['fileName', 'SongID', 'Genre', 'SongName'] + emotion_columns]
    else:
        feature_columns = [
            col for col in sheet1_data.columns 
            if col not in ['fileName', 'SongID', 'Genre', 'SongName'] + emotion_columns and 'Deviation' not in col
        ]

    return emotion_columns,feature_columns


# Correlation

# In[5]:


def correlation_matrix(sheet1_data,emotion_columns,feature_columns):
    # Compute the correlation matrix between emotions and audio features
    correlation_matrix = sheet1_data[emotion_columns + feature_columns].corr()
    
    # Extract the correlation of emotions with audio features
    emotion_feature_correlation = correlation_matrix.loc[emotion_columns, feature_columns]
    return emotion_feature_correlation


# In[6]:


def show_columns_names(sheet1_data):
    return list(sheet1_data.columns)


# HeatMap

# In[7]:


def heatmap(emotion_feature_correlation):
    # Plot heatmap for better visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(emotion_feature_correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Between Audio Features and Emotions')
    return plt.show()
    


# Features

# In[8]:


def features(base_features,sheet1_data, emotion_columns, deviation=False):
    feature_columns = []
    
    # Determine feature columns based on the deviation flag
    for base_feature in base_features:
        # Always add the base feature
        if base_feature in sheet1_data.columns:
            feature_columns.append(base_feature)
            
            # If deviation is True, add the corresponding standard deviation feature
            if deviation:
                deviation_feature = base_feature.replace('Average', 'Standard Deviation')
                if deviation_feature in sheet1_data.columns:
                    feature_columns.append(deviation_feature)

    # Exclude non-feature columns (if any)
    feature_columns = [
        col for col in feature_columns 
        if col not in ['fileName', 'SongID', 'Genre', 'SongName']
    ]

    # Extract features and target data
    X = sheet1_data[feature_columns]  # Features DataFrame
    y = sheet1_data[emotion_columns]   # Target DataFrame

    return X, y, feature_columns, emotion_columns


# In[9]:


def clean_data(X,y):
    # Step 1: Check for NaN values in the features and target data
    print("Checking for NaN values in features:")
    print(X.isnull().sum())
    print("\nChecking for NaN values in emotions:")
    print(y.isnull().sum())
    
    # Step 2: Handle NaN values
    # Option A: Drop rows with any NaN values
    X = X.dropna()
    y = y.loc[X.index]  # Align y with remaining rows in X
    
    # Drop rows with NaN in the Neutral emotion column
    y = y.dropna(subset=['Neutral'])
    X = X.loc[y.index]  # Align X with remaining rows in y
    return X,y


# In[10]:


def clean_data_nocomment(X,y):
    # Step 2: Handle NaN values
    # Option A: Drop rows with any NaN values
    X = X.dropna()
    y = y.loc[X.index]  # Align y with remaining rows in X
    
    # Drop rows with NaN in the Neutral emotion column
    y = y.dropna(subset=['Neutral'])
    X = X.loc[y.index]  # Align X with remaining rows in y
    return X,y


# In[11]:


def importance_from_features(sheet1_data,emotion_columns,X, y,feature_columns):
    # Store feature importance for each emotion
    importance_results = {}
    
    for emotion in emotion_columns:
        # Define target variable
        y = sheet1_data[emotion]
        
        # Split the data into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get feature importance scores
        feature_importances = model.feature_importances_
        
        # Store the importance scores in a DataFrame
        importance_results[emotion] = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
    
    # Display the feature importance for each emotion
    for emotion, importance_df in importance_results.items():
        print(f"\nFeature Importance for {emotion}:")
        print(importance_df)

    
    return importance_results
    


# In[12]:


def plotimportanceresult(importance_results):
    for emotion, importance_df in importance_results.items():
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance Score')
        plt.title(f'Feature Importance in Predicting {emotion}')
        plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
        plt.show()


# In[13]:


def importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns):
    # Store feature importance for each emotion
    importance_results = {}
    
    for emotion in emotion_columns:
        # Define target variable
        y = sheet1_data[emotion]
        
        # Split the data into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        # Get feature importance scores
        feature_importances = model.feature_importances_
        
        # Store the importance scores in a DataFrame
        importance_results[emotion] = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
    
    return importance_results
    


# In[14]:


def get_base_features_timbre():
    # Define base feature names for selecting the appropriate columns
    return [
        'Spectral Rolloff Point Overall Average',
        'Zero Crossings Overall Average', 
        'Spectral Flux Overall Average',
        'Spectral Centroid Overall Average',
    ]
    


# In[15]:


def get_base_features_rhythmic():
    # Define base feature names for selecting the appropriate columns
    return [
        'Strength Of Strongest Beat Overall Average',
        'Strongest Beat Overall Average', 
        'Root Mean Square Overall Average',
    ]


# In[16]:


def get_base_features_dynamics():
    # Define base feature names for selecting the appropriate columns
    return [
        'Root Mean Square Overall Average',
        'Fraction Of Low Energy Windows Overall Average', 
    ]


# In[ ]:





# In[ ]:





# In[ ]:




