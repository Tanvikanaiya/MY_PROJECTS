#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

file_path = 'greendestination (1).csv'
data = pd.read_csv(file_path)
print(data.head())


# In[2]:


from sklearn.preprocessing import LabelEncoder

categorical_columns = [
    'Attrition', 'BusinessTravel', 'Department', 'EducationField', 
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

label_encoder = LabelEncoder()

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
print(data.head())


# In[3]:


summary_statistics = data.describe()
print(summary_statistics)


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Attrition', data=data)
plt.title('Attrition Distribution')
plt.show()


# In[5]:


sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()


# In[6]:


sns.histplot(data['YearsAtCompany'], kde=True)
plt.title('Years at Company Distribution')
plt.show()


# In[7]:


sns.histplot(data['MonthlyIncome'], kde=True)
plt.title('Monthly Income Distribution')
plt.show()


# In[8]:


attrition_rate = data['Attrition'].mean() * 100
print(f'Attrition Rate: {attrition_rate:.2f}%')


# In[12]:


# Check data types of all columns
print(data.dtypes)

# Identify non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print(f"Non-numeric columns: {non_numeric_columns}")

# Inspect unique values of each non-numeric column
for column in non_numeric_columns:
    print(f"Unique values in {column}: {data[column].unique()}")

# Clean the data by replacing 'Y' and 'N' with 1 and 0, respectively
data = data.replace({'Y': 1, 'N': 0})

# Apply label encoding to each remaining non-numeric column
label_encoder = LabelEncoder()
for column in non_numeric_columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Verify data types to ensure all columns are now numeric
print(data.dtypes)


# In[13]:


# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[17]:





# In[18]:


# Feature importance
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance in Predicting Attrition')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[19]:


# Features and target variable
X = data.drop(columns=['Attrition'])
y = data['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression model with 'liblinear' solver
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




