# %% [markdown]
# Import Packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from io import StringIO 
import requests

# %% [markdown]
# # College Dataset Pipeline

# %%
def load_college_data(csv):
    college = pd.read_csv(csv)
    return college

# %%
def create_target_college(college):
    # create target variable
    college['grad_ontime_above_median'] = (
        college.grad_100_percentile > 50
    ).astype(int)

    # find prevalence
    prevalence = college.grad_ontime_above_median.mean()

    return college
# %%
def clean_college_data(college):
    # drop unnecessary features
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

    # Handle Missing Values
    numeric_cols = college.select_dtypes(include=[np.number]).columns
    categorical_cols = college.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if col != "grad_ontime_above_median":
            college[col] = college[col].fillna(college[col].median())

    for col in categorical_cols:
        college[col] = college[col].fillna(college[col].mode()[0])

    return college

# %%
def standardize_college_data(college):
    # Convert data types and clean up categories
    cat_cols = ['level','control','basic']
    college[cat_cols] = college[cat_cols].astype('category')
    controls = ['Public','Private']
    college.control = (college.control.apply(lambda x: x if x in controls
                               else "Private")).astype('category')
    
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

    # standardize the numeric columns
    numeric_cols = list(college.select_dtypes('number'))
    college[numeric_cols] = MinMaxScaler().fit_transform(college[numeric_cols])

    # one-hot encoding
    category_list = list(college.select_dtypes('category'))
    college_encoded = pd.get_dummies(college, columns=category_list)

    return college_encoded


# %%
def split_college(college_encoded):
    # drop variables used to make target
    college_clean = college_encoded.drop(columns=["grad_100_percentile","grad_100_value"])

    # First Split

    train, test = train_test_split(
        college_clean,
        train_size= 0.7,    
        stratify=college_clean.grad_ontime_above_median
    )

    # Second Split 
    tune, test = train_test_split(
        test,
        train_size=.5,
        stratify=test.grad_ontime_above_median
    )   

    return train, tune, test

# %% [markdown]
# # Jobs Dataset Pipeline
# %%
def load_jobs_data(url):
    jobs = pd.read_csv(url)
    return jobs
# %%
def clean_jobs_data(jobs):
    # fill missing values
    jobs['salary'] = jobs['salary'].fillna(0)
    return jobs
# %%
def standardize_jobs_data(jobs):
    # convert strings to categorical columns
    cat_cols = ['gender','ssc_b','hsc_b','hsc_s','degree_t','specialisation']
    jobs[cat_cols] = jobs[cat_cols].astype('category')

    # convert other strings to booleans
    jobs["workex"] = (
        jobs["workex"]
        .str.strip()
        .str.lower()
        .map({"yes": True, "no": False})
        .astype(bool)
    )

    jobs["status"] = (
        jobs["status"]
        .str.strip()
        .str.lower()
        .map({
            "placed": True,
            "not placed": False
        })
        .astype(bool)
    )

    # standardize numeric columns
    numeric_cols = list(jobs.select_dtypes('number'))
    jobs[numeric_cols] = MinMaxScaler().fit_transform(jobs[numeric_cols])

    # one-hot encoding
    category_list = list(jobs.select_dtypes('category'))
    jobs_encoded = pd.get_dummies(jobs, columns=category_list)

    return jobs_encoded
# %%
def create_target_jobs(jobs_encoded):
    # Establish the target variable:
    jobs_encoded['placement'] = (
        jobs_encoded.status == True
    ).astype(bool)

    # Find the prevalence
    prevalence = jobs_encoded.placement.mean()

    return jobs_encoded
# %%
def split_jobs(jobs_encoded):
    # drop the column(s) used to make target variable
    jobs_clean = jobs_encoded.drop(columns=['status','salary'])

    # First Split
    train, test = train_test_split(
        jobs_clean,
        train_size= 0.7,      
        stratify=jobs_clean.placement
    )

    # Second Split
    tune, test = train_test_split(
        test,
        train_size=.5,
        stratify=test.placement
    )

    return train, test, tune
# %%
