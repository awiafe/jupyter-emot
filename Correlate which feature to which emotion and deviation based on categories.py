#!/usr/bin/env python
# coding: utf-8

# Some outputs are commented for readability.
# 
# To run file, 
# 1. kindly make sure the notebook and the helper notebook are in the same location.
# 2. Run the Helper notebook
# 3. then Run this
# 
# NB: All functions are called from the helper, so you can cross check in case of any adjustments,
# 
# If you are using python environment to run code, then restart it if you make an adjustement for it to reflect. 

# In[1]:


import nbimporter
import helper as hp


# In[2]:


sheet1_data = hp.read_data()


# In[3]:


# hp.show_head(sheet1_data)


# In[4]:


emotion_columns, feature_columns = hp.get_main_columns(sheet1_data)


# In[5]:


emotion_feature_correlation = hp.correlation_matrix_t(sheet1_data,emotion_columns,feature_columns)
# hp.heatmap(emotion_feature_correlation)


# In[6]:


emotion_feature_correlation = hp.correlation_matrix(sheet1_data,emotion_columns,feature_columns)
# print(emotion_feature_correlation)
# hp.heatmap(emotion_feature_correlation)


# In[7]:


correlation_matrix_feature,correlation_matrix_emotion = hp.correlation_matrix_per(sheet1_data,emotion_columns,feature_columns)
# hp.heatmap(correlation_matrix_emotion)
# hp.heatmap(correlation_matrix_feature)


# In[8]:


# hp.show_columns_names(sheet1_data)


# In[9]:


emotion_columns,feature_columns = hp.get_main_columns(sheet1_data, deviation= True)

X, y, feature_columns, emotion_columns = hp.features(feature_columns,sheet1_data, emotion_columns, deviation = True)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results,"all_deviation")


# Timbre Features with Deviation

# In[10]:


base_features = hp.get_base_features_timbre()

X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns, deviation = True)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results,"timbre_deviation")


# Timbre Features without Deviation

# In[11]:


X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results,"timbre")


# rhythmic features ( beat and tempo detection) with Deviation

# In[12]:


base_features = hp.get_base_features_rhythmic()

X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns, deviation = True)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results,"rhythmic_deviation")


# rhythmic features ( beat and tempo detection) without Deviation

# In[13]:


X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results,"rhythmic")


# Dynamic features ( beat and tempo detection) with Deviation

# In[14]:


base_features = hp.get_base_features_dynamics()

X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results, "dynamics_deviation")


# Dynamic features ( beat and tempo detection) without Deviation

# In[15]:


X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results, "dynamics")


# Harmony With Deviation

# In[16]:


base_features = hp.get_base_features_harmony()

X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns, deviation = True)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results,"harmony_deviation")


# Harmony Without Deviation

# In[17]:


X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns)
X, y = hp.clean_data_nocomment(X,y)
importance_results = hp.importance_from_features_nocomment(sheet1_data,emotion_columns,X, y,feature_columns)
hp.plot_importance_result(importance_results, "harmony")


# In[ ]:





# ........................................................Stop HERE......................................................

# Using SVM 

# In[12]:



base_features = hp.get_base_features_timbre()
X, y, feature_columns, emotion_columns = hp.features(base_features,sheet1_data, emotion_columns, deviation = True)


# In[13]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


# In[29]:


# Ensure y_train and y_test are NumPy arrays (or use pd.DataFrame.values if using pandas DataFrames)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Check the dimensions of y_train and emotion_columns
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Number of emotion columns: {len(emotion_columns)}")

# Define a function to convert each row of emotion columns to a list of active emotions
def convert_labels_to_list(row):
    emotions = []
    for i, emotion in enumerate(emotion_columns):
        if row[i] == 1:  # If the emotion is marked as 1 (present)
            emotions.append(emotion)
    return emotions

# Convert each row in y_train and y_test to a list of emotions
y_train_list = [convert_labels_to_list(row) for row in y_train]
y_test_list = [convert_labels_to_list(row) for row in y_test]


# In[30]:


# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Binarize the labels (multi-label output)
y_train_binarized = mlb.fit_transform(y_train_list)
y_test_binarized = mlb.transform(y_test_list)


# In[31]:


# Create a Support Vector Classifier (SVC) for multi-label classification
classifier = OneVsRestClassifier(SVC(kernel='linear'))


# In[32]:



# Train the classifier
classifier.fit(X_train, y_train_binarized)

# Predict on the test set
y_pred = classifier.predict(X_test)


# In[44]:


accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(f"Accuracy: {accuracy}")


# Ensure y_test and y_pred have the same shape
y_test_classes = y_test.argmax(axis=1)
y_pred_classes = y_pred.argmax(axis=1)

# Get unique classes and their counts
unique_y_test = np.unique(y_test_classes)
unique_y_pred = np.unique(y_pred_classes)

print(f"Unique classes in y_test: {unique_y_test}")
print(f"Unique classes in y_pred: {unique_y_pred}")

# Generate the classification report
# Use the labels parameter to specify the classes in emotion_columns
# Generate the classification report
report = classification_report(
    y_test_classes,
    y_pred_classes,
    target_names=emotion_columns,
    labels=np.arange(len(emotion_columns)),
    zero_division=0  # Set undefined metrics to 0
)
print(report)


# In[36]:


# Print the accuracy score
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Print a detailed classification report
# print(classification_report(y_test, y_pred, target_names=emotion_columns))


# In[ ]:


# import joblib

# Save the trained model to a file
# joblib.dump(classifier, 'emotion_classifier_model.pkl')


# In[45]:


import pandas as pd

# Assuming you have a y_train variable
y_train_classes = y_train.argmax(axis=1)
train_counts = pd.Series(y_train_classes).value_counts()
print("Class distribution in training data:")
print(train_counts)


# In[ ]:





# In[47]:


get_ipython().system('pip install imbalanced-learn')


# In[64]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MultiLabelBinarizer

# Use MultiLabelBinarizer to convert labels to a binary format
mlb = MultiLabelBinarizer()
y_binarized = mlb.fit_transform(y)  # Ensure labels are in binary format

# Initialize a list to store models and predictions
models = []
y_preds = np.zeros_like(y_binarized)  # Initialize predictions for all samples

# Loop through each class (emotion)
for i in range(y_binarized.shape[1]):
    # Extract the current label (emotion)
    current_label = y_binarized[:, i]

    # Resample to handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, current_label)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the SVM classifier for the current label
    svm_model = SVC(kernel='rbf', class_weight='balanced')
    svm_model.fit(X_train, y_train)

    # Make predictions for the current label
    y_pred = svm_model.predict(X_test)

    # Store the model
    models.append(svm_model)

    # Find indices of the test set in the resampled data
    test_indices = X.index.isin(X_test.index)

    # Store the predictions for the current emotion in the appropriate places
    y_preds[test_indices, i] = y_pred

# Evaluate the models
print(classification_report(y_binarized, y_preds, target_names=mlb.classes_))
print(confusion_matrix(y_binarized.argmax(axis=1), y_preds.argmax(axis=1)))


# In[ ]:





# In[70]:


import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'your-file-path.xlsx'  # Replace with your file path
data = pd.read_excel("Merged_F1 (music_emotion_featureextraxted dataset).xlsx", sheet_name='Sheet1')

# Extract the audio features (all numerical columns except metadata and emotion labels)
metadata_columns = ['fileName', 'SongID', 'Genre', 'SongName']
emotion_columns = ['Joyful', 'Happy', 'Amusing', 'Energizing', 'Dreamy', 'Relaxing','Neutral','Sad','Annoying','Anxious']  # Modify based on actual emotion columns
feature_columns = data.columns.difference(metadata_columns + emotion_columns)

# Prepare the target emotions directly without grouping
X = data[feature_columns]
y = data[emotion_columns].idxmax(axis=1)  # This assumes the emotions are binary 0/1, adjust if needed

# Convert the emotion labels from strings to numerical values using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# SVM Classification
svm_classifier = svm.SVC(kernel='rbf', C=1, gamma='scale')
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(svm_classifier, X, y_encoded, cv=kfold, scoring='accuracy')

# Train the classifier on the full dataset and predict for the confusion matrix
svm_classifier.fit(X, y_encoded)
y_pred = svm_classifier.predict(X)

# Calculate confusion matrix and classification report
conf_matrix = confusion_matrix(y_encoded, y_pred)
class_report = classification_report(y_encoded, y_pred, zero_division=0)

# Print classification results
print("Mean Accuracy (10-fold CV):", accuracy_scores.mean())
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# SVM Regression (for predicting numerical emotion ratings)
svm_regressor = svm.SVR(kernel='rbf', C=1, gamma='scale')

# Perform cross-validation to predict ratings
predictions = cross_val_predict(svm_regressor, X, y_encoded, cv=kfold)

# Calculate Mean Squared Error for regression model
mse = mean_squared_error(y_encoded, predictions)

# Print regression results
print("Mean Squared Error for SVM Regression:", mse)


# In[71]:


import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

# Load the dataset
file_path = 'your-file-path.xlsx'  # Replace with your file path
data = pd.read_excel("Merged_F1 (music_emotion_featureextraxted dataset).xlsx", sheet_name='Sheet1')

# Extract the audio features (all numerical columns except metadata and emotion labels)
metadata_columns = ['fileName', 'SongID', 'Genre', 'SongName']
emotion_columns = ['Joyful', 'Happy', 'Amusing', 'Energizing', 'Dreamy', 'Relaxing','Neutral','Sad','Annoying','Anxious']  # Modify based on actual emotion columns
feature_columns = data.columns.difference(metadata_columns + emotion_columns)

# Prepare the target emotions directly without grouping
X = data[feature_columns]
y = data[emotion_columns].idxmax(axis=1)  # This assumes the emotions are binary 0/1, adjust if needed

# Convert the emotion labels from strings to numerical values using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create a pipeline for scaling and SVM classification with class weights
svm_classifier = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced'))

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(svm_classifier, X, y_encoded, cv=kfold, scoring='accuracy')

# Train the classifier on the full dataset and predict for the confusion matrix
svm_classifier.fit(X, y_encoded)
y_pred = svm_classifier.predict(X)

# Calculate confusion matrix and classification report
conf_matrix = confusion_matrix(y_encoded, y_pred)
class_report = classification_report(y_encoded, y_pred, zero_division=0)

# Print classification results
print("Mean Accuracy (10-fold CV):", accuracy_scores.mean())
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# SVM Regression (for predicting numerical emotion ratings)
svm_regressor = make_pipeline(StandardScaler(), svm.SVR(kernel='rbf', C=1, gamma='scale'))

# Perform cross-validation to predict ratings
predictions = cross_val_predict(svm_regressor, X, y_encoded, cv=kfold)

# Calculate Mean Squared Error for regression model
mse = mean_squared_error(y_encoded, predictions)

# Print regression results
print("Mean Squared Error for SVM Regression:", mse)

print("Mapping of numerical labels to emotion names:")
for i, emotion in enumerate(label_encoder.classes_):
    print(f"{i} = {emotion}")


# In[73]:


from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

print(X)
print('------------------------------------')
print(y)
# Replace these with your actual data (features and labels)
X = X # Features (200 audio features)
y = y  # Labels (emotions)

# Create a dictionary to store correlations for each emotion category
categories = ['Joyful', 'Happy', 'Amusing', 'Energizing', 'Dreamy', 'Relaxing','Neutral','Sad','Annoying','Anxious']  # Update if needed
svm_corr = []
linear_corr = []

# Iterate through each emotion category to calculate correlations
for category in categories:
    # Assuming y_train has multiple columns, one for each emotion category
    y_category = y[category]

    # SVM Regression
    svr = SVR(kernel='rbf')
    svm_scores = cross_val_score(svr, X, y_category, cv=10, scoring='r2')  # R-squared correlation
    svm_corr.append(np.mean(svm_scores))

    # Linear Regression
    lr = LinearRegression()
    linear_scores = cross_val_score(lr, X, y_category, cv=10, scoring='r2')
    linear_corr.append(np.mean(linear_scores))

# Create a DataFrame to display the correlation results
corr_df = pd.DataFrame({
    'Category': categories,
    'SVM (corr)': svm_corr,
    'Linear Regression (corr)': linear_corr
})

# Display the correlation table
print("Correlation Table for SVM and Linear Regression:")
print(corr_df)

# If you want to save the correlation table as an Excel or CSV file
corr_df.to_excel('correlation_table.xlsx')  # or use .to_csv('correlation_table.csv')


# In[74]:


import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Assuming X is your features DataFrame
# y is currently a Series with emotion labels, convert to one-hot encoding
y_one_hot = pd.get_dummies(y, prefix='Emotion')

# Create a dictionary to store correlations for each emotion category
categories = y_one_hot.columns.tolist()  # Get the new column names
svm_corr = []
linear_corr = []

# Iterate through each emotion category to calculate correlations
for category in categories:
    y_category = y_one_hot[category]  # Access the one-hot encoded column

    # SVM Regression
    svr = SVR(kernel='rbf')
    svm_scores = cross_val_score(svr, X, y_category, cv=10, scoring='r2')  # R-squared correlation
    svm_corr.append(np.mean(svm_scores))

    # Linear Regression
    lr = LinearRegression()
    linear_scores = cross_val_score(lr, X, y_category, cv=10, scoring='r2')
    linear_corr.append(np.mean(linear_scores))

# Create a DataFrame to display the correlation results
corr_df = pd.DataFrame({
    'Category': categories,
    'SVM (corr)': svm_corr,
    'Linear Regression (corr)': linear_corr
})

# Display the correlation table
print("Correlation Table for SVM and Linear Regression:")
print(corr_df)

# If you want to save the correlation table as an Excel or CSV file
# corr_df.to_excel('correlation_table.xlsx')  # or use .to_csv('correlation_table.csv')


# In[84]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample correlation data
data = {
    'Category': [
        'Emotion_Amusing',
        'Emotion_Annoying',
        'Emotion_Anxious',
        'Emotion_Dreamy',
        'Emotion_Energizing',
        'Emotion_Happy',
        'Emotion_Joyful',
        'Emotion_Neutral',
        'Emotion_Relaxing',
        'Emotion_Sad'
    ],
    'SVM (corr)': [
        -0.039558,
        -0.060321,
        -0.080887,
        -0.046379,
        -0.033525,
        -0.045322,
        -0.051358,
        -0.040340,
        -0.043941,
        -0.096738
    ],
    'Linear Regression (corr)': [
        -0.128015,
        -0.001293,
        -0.166611,
        -0.272015,
        -0.217309,
        -0.201862,
        -0.377663,
        -0.261473,
        -0.171236,
        -0.087360
    ]
}

# Create DataFrame
correlation_df = pd.DataFrame(data)

# Set Category as index
correlation_df.set_index('Category', inplace=True)

# Display the DataFrame
print(correlation_df)

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Table for SVM and Linear Regression')
plt.xlabel('Models')
plt.ylabel('Emotion Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# The data you provided represents the correlation coefficients between two machine learning models—**Support Vector Machine (SVM)** and **Linear Regression**—for different emotion categories. Here’s a breakdown of what each part of the data means:
# 
# ### Correlation Coefficient
# 
# 1. **Definition**: The correlation coefficient (often denoted as \( r \)) is a statistical measure that describes the strength and direction of a relationship between two variables. It ranges from -1 to 1:
#    - **1**: Perfect positive correlation (as one variable increases, the other also increases).
#    - **0**: No correlation (no linear relationship).
#    - **-1**: Perfect negative correlation (as one variable increases, the other decreases).
# 
# ### Emotion Categories
# 
# - The table contains various **emotion categories** (e.g., *Emotion_Amusing*, *Emotion_Sad*, etc.) that represent different emotional responses that your models are trying to predict. Each row corresponds to a specific emotion.
# 
# ### Correlation Values
# 
# - **SVM (corr)**: The correlation coefficients for the predictions made by the Support Vector Machine model for each emotion category. 
#   - For instance, a value of **-0.039558** for *Emotion_Amusing* suggests that there is a very weak negative correlation between the SVM predictions and the actual values for this emotion.
#   
# - **Linear Regression (corr)**: The correlation coefficients for the predictions made by the Linear Regression model for each emotion category. 
#   - For example, a value of **-0.128015** for *Emotion_Amusing* indicates a weak negative correlation for Linear Regression predictions.
# 
# ### Interpretation of the Values
# 
# - **Negative Correlations**: 
#   - Most values in your table are negative, indicating that as the predicted values for these emotions increase, the actual values tend to decrease. This could suggest that both models are poorly predicting these emotions, potentially due to a lack of relevant features, noise in the data, or the models' inherent limitations.
# 
# - **Weak Correlations**: 
#   - The absolute values of the correlations are relatively low, typically below 0.4. This suggests that the models may not be capturing a strong linear relationship between the predicted and actual values for these emotions.
#   
# - **Comparison Between Models**: 
#   - By comparing the correlation coefficients for SVM and Linear Regression, you can determine which model is performing better for each emotion. For example, *Emotion_Joyful* has a correlation of **-0.051358** for SVM and **-0.377663** for Linear Regression, suggesting that Linear Regression may be more strongly related to the actual values for this emotion despite both being negative.
# 
# ### Conclusions
# 
# - The correlations give you insight into how well each model is able to predict the emotions based on the features provided. Low or negative correlation values indicate that the models may not be effective for this task as they do not align well with the actual emotional responses.
# 
# - Further analysis may be needed to improve model performance, such as:
#   - **Feature Engineering**: Adding more relevant features that may help improve predictions.
#   - **Hyperparameter Tuning**: Adjusting model parameters to optimize performance.
#   - **Different Models**: Experimenting with other machine learning models or ensemble methods. 
# 
# Understanding these correlations is crucial for model evaluation and making informed decisions about future model improvements or feature selections.

# In[88]:


import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your features and target variables
# Assuming X is your features DataFrame and y is your target Series
# X = pd.read_csv('features.csv')  # Load your features
# y = pd.read_csv('labels.csv')     # Load your labels

# Example:
# X = pd.DataFrame(...)  # Replace with actual data
# y = pd.Series(...)     # Replace with actual data

# Check the shapes of X and y
print(f"Initial shapes - X: {X.shape}, y: {y.shape}")

# Convert y to one-hot encoding
y_one_hot = pd.get_dummies(y, prefix='Emotion')

# Check the shape of one-hot encoded y
print(f"One-hot encoded y shape: {y_one_hot.shape}")

# Ensure X and y have the same number of samples
if len(X) != len(y_one_hot):
    print("Warning: The lengths of X and one-hot encoded y do not match.")
    min_length = min(len(X), len(y_one_hot))
    X = X.iloc[:min_length]
    y_one_hot = y_one_hot.iloc[:min_length]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a dictionary to store correlations for each emotion category
categories = y_one_hot.columns.tolist()  # Get the new column names
svm_corr = []
linear_corr = []

# Iterate through each emotion category to calculate correlations
for category in categories:
    y_category = y_one_hot[category]  # Access the one-hot encoded column
    print(f"Processing category: {category}, y_category shape: {y_category.shape}")

    # Ensure that the lengths match
    if len(y_category) != len(X_scaled):
        print(f"Length mismatch for {category}: X_scaled ({len(X_scaled)}) vs y_category ({len(y_category)})")
        continue  # Skip this category if there's a mismatch

    # SVM Regression
    svr = SVR(kernel='rbf')
    svm_scores = cross_val_score(svr, X_scaled, y_category, cv=10, scoring='r2')  # R-squared correlation
    svm_corr.append(np.mean(svm_scores))

    # Linear Regression
    lr = LinearRegression()
    linear_scores = cross_val_score(lr, X_scaled, y_category, cv=10, scoring='r2')
    linear_corr.append(np.mean(linear_scores))

# Create a DataFrame to display the correlation results
corr_df = pd.DataFrame({
    'Category': categories,
    'SVM (corr)': svm_corr,
    'Linear Regression (corr)': linear_corr
})

# Display the correlation table
print("Correlation Table for SVM and Linear Regression:")
print(corr_df)

# Save the correlation table as an Excel or CSV file if needed
# corr_df.to_excel('correlation_table.xlsx')  # or use .to_csv('correlation_table.csv')


# In[ ]:




