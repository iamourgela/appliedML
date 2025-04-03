# %% [markdown]
# # House Prices Regression Analysis
# This notebook follows a complete pipeline: EDA, preprocessing, modeling, and evaluation using linear models.

# %%
# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Plot aesthetics
sns.set_context("paper", rc={"font.size":15, "axes.titlesize":15, "axes.labelsize":15})  
plt.rcParams['axes.labelsize']  = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Scikit-learn utilities
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score

# Dimensionality reduction (optional)
from sklearn.decomposition import PCA

# Inline plotting for Jupyter
%matplotlib inline


# %% [markdown]
# ##  Load and Inspect the Data

# %% [markdown]
# This data is interesting because there is a combination of oridinal, numerical and categorical data. It is important to handle it correctly from the beginning by inspecting the description of the data and features. 

# %%
#load the datasets and don't convert "NA" to NaN because it is an entry for 14 features 
#meaning that the house doesn't have that specific feature (eg. Alley, )
import pandas as pd

# Load normally with "NA" values as NaN
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Features where "NA" or "None" is a valid category 
na_is_valid = [
    'Alley', 'Fence', 'MiscFeature', 'PoolQC', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]

# Restore 'NA' string in valid-category columns only (if it got converted to NaN)
for col in na_is_valid:
    if col in train.columns:
        train[col] = train[col].fillna('NA')
    if col in test.columns:
        test[col] = test[col].fillna('NA')

print("Train Shape:",train.shape)
print(train.head())

# %%
print("Test Shape:",test.shape)
print(test.head())

# %%
#check for data leakage between the test and train dataset using ID feature 
overlap = set(train['Id']).intersection(set(test['Id']))
print("Number of Overlapping values:", len(overlap))

#we are good! no data leakage according to common ID numbers 

# %%
#set ID to index for the train datasets
train.set_index('Id', inplace=True) 
train.head(2)

# %%
#set ID to index for the test datasets
test.set_index('Id', inplace=True) 
test.head(2)

# %% [markdown]
# ## Check for and Handle Missing Values (Imputing)

# %% [markdown]
# We can impute missing values here because the train and test data is already split and there is no leakage

# %%
# Check nulls in train set
nulls = train.isnull().sum().to_frame(name='MissingValues_Train')
#sorting by descending bc there's too many features
nulls.sort_values(by='MissingValues_Train', ascending=False, inplace=True)
print(nulls.head(10))
#LotFrontage, MasVnrArea and Electrical all have missing values we will impute

# %%
#visualize null values
nulls_train_viz=train[["LotFrontage", "MasVnrArea", "Electrical","Street","Alley", "LotArea","SalePrice"]]

sns.heatmap(nulls_train_viz.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# %%
# Find datatypes of columns with null values
#Making sure target feature doesn't have nulls

train[["LotFrontage", "MasVnrArea", "Electrical","SalePrice"]].info()


# %%
# Check nulls in test set
nulls_test = test.isnull().sum().to_frame(name='MissingValues_test')
#sorting by descending bc there's too many features
nulls_test.sort_values(by='MissingValues_test', ascending=False, inplace=True)
print(nulls_test.head(20))

#'LotFrontage', 'MasVnrArea', 'MSZoning', 'BsmtHalfBath', 'Functional', 'BsmtFullBath', 'Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 
# 'SaleType', 'KitchenQual', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2',  'GarageYrBlt' all have missing values we will impute

# %%
#visualize null values
nulls_test_viz=test[['LotFrontage', 'MasVnrArea', 'MSZoning', 'BsmtHalfBath', 'Functional', 'BsmtFullBath', 'Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'SaleType', 'KitchenQual', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'GarageYrBlt']]

sns.heatmap(nulls_test_viz.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# %%
# Find datatypes of columns with null values
#Making sure target feature doesn't have nulls

test[['LotFrontage', 'MasVnrArea', 'MSZoning', 'BsmtHalfBath', 'Functional', 'BsmtFullBath', 'Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'SaleType', 'KitchenQual', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2']].info()


# %%
#long code so will define a new one
num_col = ['LotFrontage', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea',  'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2']
cat_col = ['MSZoning','Functional','Utilities','Exterior1st','Exterior2nd','SaleType','KitchenQual']

#instantiate the imputers 
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

#fit the imputers
train[num_col] = num_imputer.fit_transform(train[num_col])
test[num_col]=num_imputer.transform(test[num_col])
train[cat_col] = cat_imputer.fit_transform(train[cat_col])
test[cat_col] = cat_imputer.transform(test[cat_col])


# %%
#Sanity check on the null values 
print("Train shape:", train.shape)
print("Train nulls:", train.isnull().sum())
print("-"*30)
print("Test shape:", test.shape)
print("Test nulls:", test.isnull().sum())

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
# Histogram to check the distribution of our target variable
plt.figure(figsize=(15, 5))
sns.histplot(data=train, x="SalePrice", kde=True)
plt.show()

# %%
# 1) Print the skewness
# 2) print the kurtosis

print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

# a lot of kurtosis because it has a really long tail

# %% [markdown]
# ##### Independent variables: check the statistical properties of the numerical features 

# %%
# Get the statistical properties of the numerical features

train.describe()

# %% [markdown]
# ##### Correlation heatmap 

# %%
# Get correlation matrix
corr_matrix = train.corr(numeric_only=True)

# Correlation with SalePrice
corr_target = corr_matrix['SalePrice'].drop('SalePrice')

# Get top 10 positively and 10 negatively correlated features
pos_corr = corr_target.sort_values(ascending=False).head(10)
neg_corr = corr_target.sort_values(ascending=True).head(10)
combined_corr = pd.concat([pos_corr, neg_corr])

# Plot as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_corr.to_frame(), annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Top Positively and Negatively Correlated Features with SalePrice")
plt.tight_layout()
plt.show()


# %%
#sns box plot of the features against one another instead of our target variable
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot_kws={'size': 8}, cmap="coolwarm", yticklabels=True);

# %%
# Select some of the continuous numerical features and HouseStyle as the Hue
features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'OverallQual', 'YearBuilt','HouseStyle']

# make a dataframe of those features
con_features = train[features]

# Plot pairplot
sns.pairplot(data=con_features, hue='HouseStyle', palette='colorblind')
plt.show()

# %%
# Plot the relationship between GrLivArea and SalePrice using a 2D scatterplot. Use hue='cut'

plt.figure(figsize=(15, 10))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue="HouseStyle", data=train, palette='colorblind', s=15)
plt.show()

# %%
#group by to see average sale price by housestyle and street type

grouped_col = train.groupby(['HouseStyle', 'Street'])['SalePrice'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_col, x='SalePrice', y='HouseStyle', hue='Street', palette='rocket')
plt.title("Average Sale Price by House Style and Street Type")


# %% [markdown]
# ## Feature Engineering and Creation

# %% [markdown]
# Due to the pleathora of features, many are repetitive and can be combined and perhaps will become more useful. The following new features will be created based on highly correlated features:
# 
# 1. TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF (numerical, continuous)
# 2. TotalBathrooms = Combination of FullBath, HalfBAth, BsmtFullBath and BsmtHalfBath (numerical, float)
# 3. Remodel = When YearBuilt does not equal YearRemodAdd (binary)
# 4. GSpaceperCar = GarageArea / GarageCars to determine actual space per car (numerical, continuous)
# 5. RemodelAge = YearRemodAdd - YearBuilt
# 
# Because many of these features are highly correlated to SalePrice on their own, we won't drop all of them initially. 

# %%
# 1. TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# 2. TotalBathrooms = FullBath + (0.5 * HalfBath) + BsmtFullBath + (0.5 * BsmtHalfBath)
train['TotalBathrooms'] = (
    train['FullBath'] + (0.5 * train['HalfBath']) + 
    train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath'])
)
test['TotalBathrooms'] = (
    test['FullBath'] + (0.5 * test['HalfBath']) + 
    test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath'])
)

# 3. Remodel = When YearBuilt != YearRemodAdd
train['Remodeled'] = (train['YearBuilt'] != train['YearRemodAdd']).astype(int)
test['Remodeled'] = (test['YearBuilt'] != test['YearRemodAdd']).astype(int)

# 4. GSpaceperCar = GarageArea / GarageCars (with handling for zero cars)
train['GSpaceperCar'] = (train['GarageArea'] / train['GarageCars'] +1)
test['GSpaceperCar'] = (test['GarageArea'] / test['GarageCars'] +1 )

#5. RemodelAge = YearRemodAdd-YearBuilt (added later because remodel binary was not highly correlated)
train['RemodelAge'] = (train['YearRemodAdd'] - train['YearBuilt'])
test['RemodelAge'] = (test['YearRemodAdd'] / test['YearBuilt'])

# %% [markdown]
# Now that the new features are created, I want to visualize their correlation and plot them to see if they are useful

# %%
#create a DF using the new features and their components to compare correlation w target feature
newfeatures = train[['SalePrice','TotalSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','TotalBathrooms','FullBath', 'HalfBath',
                    'BsmtFullBath','BsmtHalfBath','Remodeled', 'RemodelAge','YearBuilt','YearRemodAdd','GSpaceperCar','GarageArea','GarageCars']]

# Get correlation matrix
new_corr_matrix = newfeatures.corr(numeric_only=True)

# Correlation with SalePrice
new_corr_target = new_corr_matrix['SalePrice'].drop('SalePrice').sort_values(ascending=False)

# Plot as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(new_corr_target.to_frame(), annot=True, cmap='coolwarm',vmin=-1, vmax=1)
plt.title("New and Original Correlated Features with SalePrice")
plt.tight_layout()
plt.show()



# %% [markdown]
# There are too many features to encode. Let's drop the features that are between -0.29 and 0.29. At least the numerical ones before encoding. 

# %%
#Drop unecessary features

# Get correlation matrix
final_corr_matrix = train.corr(numeric_only=True)

# Correlation with SalePrice
final_corr_target = final_corr_matrix['SalePrice'].drop('SalePrice')

drop_col = final_corr_target[(final_corr_target > -0.29) & (final_corr_target < 0.29)].index.tolist()

print("Columns to drop:", drop_col)

# %%
#drop the columns 
train.drop(columns=drop_col, inplace=True)
test.drop(columns=drop_col, inplace=True)

#sanity check
print(train.shape)
print(test.shape)

# %% [markdown]
# ## Encoding and Scaling

# %% [markdown]
# Due to the combination of categorical and ordinal features, we will use a combination of OHE and ordinal encoding

# %%
# Define ordinal mappings
qual_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
exposure_map = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
finish_map = {'Unf': 1, 'RFn': 2, 'Fin': 3}
fence_map = {'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
slope_map = {'Sev': 1, 'Mod': 2, 'Gtl': 3}
lotshape_map = {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
paved_map = {'N': 1, 'P': 2, 'Y': 3}
util_map = {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}
func_map = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}

# Dictionary to map the column names to their variables
ordinal_features = {
    'ExterQual': qual_map,
    'ExterCond': qual_map,
    'BsmtQual': qual_map,
    'BsmtCond': qual_map,
    'HeatingQC': qual_map,
    'KitchenQual': qual_map,
    'FireplaceQu': qual_map,
    'GarageQual': qual_map,
    'GarageCond': qual_map,
    'PoolQC': qual_map,
    'BsmtExposure': exposure_map,
    'GarageFinish': finish_map,
    'Fence': fence_map,
    'LandSlope': slope_map,
    'LotShape': lotshape_map,
    'PavedDrive': paved_map,
    'Utilities': util_map,
    'Functional': func_map
}

# Apply mappings
for col, mapkey in ordinal_features.items():
    if col in train.columns:
        train[col] = train[col].map(mapkey)

# ------------------------------
# ONE-HOT ENCODING
# ------------------------------

# Identify object-type (nominal) columns not in ordinal list
nominal_cols = train.select_dtypes(include='object').columns.difference(ordinal_features.keys())

# One-hot encode those columns (drop_first avoids multicollinearity)
train = pd.get_dummies(train, columns=nominal_cols, drop_first=True)

# Sanity check
print("Train shape:", train.shape)
train.head()


# %%
# One-hot encoding
train_encoded = pd.get_dummies(train, drop_first=True)
test_encoded = pd.get_dummies(test, drop_first=True)

# Align columns
X = train_encoded.drop(columns=['SalePrice', 'Id'])
y = train_encoded['SalePrice']
X_test = test_encoded.reindex(columns=X.columns, fill_value=0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# %%
#check on features scale against one another 

plt.figure(figsize=(15,5))
ax = sns.boxplot(data=train)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.show();

# %% [markdown]
# ## Step 6: Feature Importance

# %%
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_scaled_df, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title('Top 20 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7: Modeling and Evaluation

# %%
# Define models
from sklearn.pipeline import Pipeline
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Polynomial Regression (deg 2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linreg', LinearRegression())])
}

# Use top features for poly regression
X_poly = X_scaled_df[top_features.index]
results = []

for name, model in models.items():
    if 'Polynomial' in name:
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
    else:
        model.fit(X_scaled_df, y)
        y_pred = model.predict(X_scaled_df)
    
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    y_bin = pd.qcut(y, q=4, labels=False)
    y_pred_bin = pd.qcut(pd.Series(y_pred).rank(method='first'), q=4, labels=False)
    f1 = f1_score(y_bin, y_pred_bin, average='macro')
    results.append({'Model': name, 'Train RMSE': round(rmse, 2), 'Train R²': round(r2, 3), 'F1 Score': round(f1, 3)})

pd.DataFrame(results).sort_values(by='Train RMSE')


