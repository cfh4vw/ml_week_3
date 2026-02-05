# %% [markdown]
#  # Setup 
# Imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from io import StringIO 
import requests

# %% [markdown]
# # Step 1: Data Loading and Idea Development
#
# Key Questions:
# - How well can we predict if a university's four year graduation rate is 
# above or below the median?
# - An idependent business measure could be the graduation rate of the university.

# %%
# Load the data
college = pd.read_csv('college.csv')
college.info()

# %% [markdown]
# # Step 2: Data Preparation and Initial Exploration

# %%
# Establish the target variable:
college['grad_ontime_above_median'] = (
    college.grad_100_percentile > 50
).astype(int)

college.grad_ontime_above_median.value_counts()
# %%
# Find the prevalence
prevalence = college.grad_ontime_above_median.mean()
print(f'Prevalence of above median graduation rates: {prevalence:.2%}')
# %%
# Drop unnecessary columns
drop = ["index", "unitid", "chronname", "city", "site", 
        'hbcu','flagship', "nicknames", "similar", 'state', 
        "counted_pct", "long_x", "lat_y", 'vsa_year',
        "vsa_grad_after4_first", "vsa_grad_elsewhere_after4_first",
        "vsa_enroll_after4_first", "vsa_enroll_elsewhere_after4_first",
        "vsa_grad_after6_first", "vsa_grad_elsewhere_after6_first",
        "vsa_enroll_after6_first", "vsa_enroll_elsewhere_after6_first",
        "vsa_grad_after4_transfer", "vsa_grad_elsewhere_after4_transfer",
        "vsa_enroll_after4_transfer", "vsa_enroll_elsewhere_after4_transfer",
        "vsa_grad_after6_transfer", "vsa_grad_elsewhere_after6_transfer",
        "vsa_enroll_after6_transfer", "vsa_enroll_elsewhere_after6_transfer",
        "counted_pct"
]
college = college.drop(
    columns=[c for c in drop if c in college.columns]
)
college.dtypes.value_counts()

# %%
# Handle Missing Values
# college.isna().sum() # there are quite a few

numeric_cols = college.select_dtypes(include=[np.number]).columns
categorical_cols = college.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    if col != "grad_ontime_above_median":
        college[col] = college[col].fillna(college[col].median())

for col in categorical_cols:
    college[col] = college[col].fillna(college[col].mode()[0])

college.isna().sum() # none left!
# %%
# Convert data types
# strings --> categorical columns
cat_cols = ['level','control','basic']
college[cat_cols] = college[cat_cols].astype('category')
college.dtypes.value_counts()

# take a look at the 'level' category
print(college.level.value_counts())
# only two categories, leave as is 

# collapse 'control' category
# print(college.control.value_counts())
controls = ['Public','Private']
college.control = (college.control.apply(lambda x: x if x in controls
                               else "Private")).astype('category')
print(college.control.value_counts())

# collapse 'basic' category
if "basic" in college.columns:
    def simplify_basic(x):
        x = str(x).lower()
        if "research" in x:
            return "Research"
        if "masters" in x:
            return "Masters"
        if "baccalaureate" in x:
            return "Baccalaureate"
        if "associate" in x:
            return "Associate"
        return "Other"

    college["basic"] = college["basic"].apply(simplify_basic).astype("category")
print(college.basic.value_counts())
# %%
# Standardize numeric cols
numeric_cols = list(college.select_dtypes('number'))
college[numeric_cols] = MinMaxScaler().fit_transform(college[numeric_cols])

# %%
# One Hot Encoding
category_list = list(college.select_dtypes('category'))
college_encoded = pd.get_dummies(college, columns=category_list)
college_encoded.info()

# %%
# Train, Tune, & Test Split
# First drop the column used to make target variable
college_clean = college_encoded.drop(columns=["grad_100_percentile","grad_100_value"])
college_clean.info() #successfully dropped. 

# First Split

train, test = train_test_split(
    college_clean,
    train_size= 0.7,      # 70% of total entries
    stratify=college_clean.grad_ontime_above_median
)

# Second Split 
tune, test = train_test_split(
    test,
    train_size=.5,
    stratify=test.grad_ontime_above_median
)

print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
print(f"Test set shape: {tune.shape}")
# %% [markdown]
# # Step 3: Concerns 
# I worry that some of the dropped columns may have been important. There was a lot
# of missing data as well, so dropping those columns and the rows with missing values
# could skew results. Graduation rates may also be due to other unseen factors
# that could not be represented in the data. There is also a pretty low 
# prevalence so I wonder if our model will be any good at predicting the rate.