#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


import numpy as np
import pandas as pd


# In[5]:


np.random.seed(42)


# In[7]:


import matplotlib as mpl
import matplotlib.pyplot as plt


# In[9]:


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[10]:


import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# In[22]:


get_ipython().system('head/cxldata/datasets/project/housing/housing.csv')


# In[24]:


HOUSING_PATH = '/cxldata/datasets/project/housing/housing.csv'


# In[25]:


housing = pd.read_csv(HOUSING_PATH)


# In[26]:


housing.head()


# In[30]:


housing.info()


# In[31]:


housing.describe()


# In[32]:


housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[34]:


housing["median_income"].hist(bins=50, figsize=(20,15))
plt.show()


# In[35]:


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()


# In[39]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[40]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[41]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[45]:


housing = strat_train_set.copy()


# In[46]:


import matplotlib.image as mpimg
california_img=mpimg.imread('/cxldata/datasets/project/housing/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.show()


# In[50]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[51]:


corr_matrix = housing.corr()


# In[52]:


corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[56]:


housing = strat_train_set.drop("median_house_value", axis=1)


# In[57]:


housing_labels= strat_train_set["median_house_value"].copy()


# In[58]:


from sklearn.impute import SimpleImputer


# In[59]:


imputer = SimpleImputer(strategy="median")


# In[60]:


housing_num = housing.drop("ocean_proximity", axis=1)


# In[61]:


imputer.fit(housing_num)


# In[62]:


X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                      index=housing.index)


# In[65]:


housing_cat= housing[["ocean_proximity"]]


# In[66]:


housing_cat.head(10)


# In[67]:


from sklearn.preprocessing import OneHotEncoder


# In[68]:


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[70]:


housing_cat_1hot[:10].toarray()


# In[74]:


from sklearn.base import BaseEstimator, TransformerMixin


# In[75]:


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[79]:


col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names]

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])


# In[80]:


housing_prepared = full_pipeline.fit_transform(housing)


# In[83]:


from sklearn.tree import DecisionTreeRegressor


# In[84]:


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[85]:


from sklearn.metrics import mean_squared_error


# In[86]:


housing_predictions = tree_reg.predict(housing_prepared)


# In[87]:


tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[90]:


from sklearn.ensemble import RandomForestRegressor


# In[100]:


forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[92]:


housing_predictions = forest_reg.predict(housing_prepared)


# In[93]:


forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[101]:


def display_scores (scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[102]:


from sklearn.model_selection import cross_val_score


# In[103]:


scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)


# In[104]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[108]:


from sklearn.model_selection import GridSearchCV


# In[109]:


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]


# In[110]:


forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# In[111]:


grid_search.best_params_


# In[112]:


grid_search.best_estimator_


# In[113]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[120]:


final_model= grid_search.best_estimator_


# In[121]:


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test= strat_test_set["median_house_value"].copy()


# In[122]:


X_test_prepared = full_pipeline.transform(X_test)


# In[123]:


final_predictions = final_model.predict(X_test_prepared)


# In[124]:


final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[125]:


final_rmse


# In[ ]:




