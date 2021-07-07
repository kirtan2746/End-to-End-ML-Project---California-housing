# End-to-End-ML-Project---California-housing
This is an end-to-end Machine Learning project. we would start by learning how to load a dataset, visualize it, fill in the missing values, create pipelines, handle categorical variables, train models based on that data, and finally predict using that model.  This will not only help you understand how to train a machine learning model, but will also give you a detailed idea of how to clean and prepare data for machine learning, train the model, and fine tune it in real life projects.  

# Skills we will develop:  

1. scikit-learn 

2. Data visualization

3. Treating missing values in dataset 

4. Handling categorical variables 

5. Creating transformation pipelines 

6. Training Machine Learning models  

here we work on California housing dataset and train a model for their district  prediction and rent prediction . 

# End to End ML Project - Import the libraries

First, let's import a few common modules, set a random seed so that the code can produce the same results every time it is executed. We will also ignore non-essential warning.

We will also set the rc params to change the label size for the plots' axes, x- and y-axis ticks using rc method.

matplotlib.rc(group, kwargs)
group is the grouping for the rc, e.g., for lines.linewidth the group is lines, for axes.facecolor, the group is axes, and so on. Group may also be a list or tuple of group names, e.g., (xtick, ytick).

INSTRUCTIONS
Import sklearn, Numpy as np, and Pandas as pd

import <<your code goes here>>
import numpy as <<your code goes here>>
import pandas as <<your code goes here>>
Set the random.seed to 42

np.random.seed(<<your code goes here>>)
Import Matplotlib as mpl, and Pyplot as plt

%matplotlib inline
import matplotlib as <<your code goes here>>
import matplotlib.pyplot as <<your code goes here>>
Set the rc params for axes, xtick, and ytick

mpl.rc('axes', labelsize=14)
mpl.<<your code goes here>>('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
Finally, we will import warnings and ensure that the non-essential warnings are ignored

import <<your code goes here>>
warnings.filterwarnings(action="ignore", message="^internal gelsd")
  
Now let's load the dataset. The dataset is located in the following path:
  
Load the dataset
  
/cxldata/datasets/project/housing/housing.csv

INSTRUCTIONS
Set HOUSING_PATH variable to the path of the dataset as given above

HOUSING_PATH = <<your code goes here>>
Read the dataset using the read_csv function of Pandas

housing = pd.<<your code goes here>>(HOUSING_PATH)
Display the first few rows of the dataset using the head method

housing.<<your code goes here>>()
  
**Explore the dataset**
  
Now we will explore the dataset. Here, we will be using the hist method to plot a histogram to view the data. A histogram is used to visually represent the distribution of the data instead of the actual data itself, simply put, it is used to summarize discrete or continuous data.

The hist methods here has the bins parameter. These are also sometimes referred to as classes, intervals, or `buckets, are groups of equal widths into which the data is separated. Each bin is plotted as a bar whose height corresponds to how many data points are in that bin.

We will also be using the cut method from Pandas. This is used to segment and sort data values into bins. cut is also helpful for converting from a continuous variable to a categorical variable. For example, cut could convert ages to groups of age ranges. Supports binning into an equal number of bins, or a pre-specified array of bins. The labels parameters here specifies the labels for the returned bins. It has to be of the same length as the resulting bins. Also, if you notice, we have mentioned a np.inf here for the bins. That is a form of floating point representation of infinity.

INSTRUCTIONS
Use the info method to get more information on the dataset

housing.<<your code goes here>>
Get a better understand of the mean, standard deviation, maximum value and other such information from the dataset by using the describe method

housing.<<your code goes here>>
Plot histograms of all the features using hist method

housing.<<your code goes here>>(bins=50, figsize=(20,15))
plt.show()
Plot a histogram of the median income attribute of the dataset

housing["median_income"].<<your code goes here>>
Divide the median income attribute into bins and labels using the cut mthod, and then plot another histogram of the same

housing["income_cat"] = pd.<<your code goes here>>(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
  
  ** Split the dataset**
  
In this step, we will split the dataset into train and test sets. We will be using the StratifiedShuffleSplit method from the sklearn library which is a cross-validator that provides train/test indices to split data in train/test sets.

INSTRUCTIONS
Import StratifiedShuffleSplit from sklearn

from sklearn.model_selection import <<your code goes here>>
Now let's divide the dataset in a 80-20 split, for this you need to set the test_size as 0.2

split = StratifiedShuffleSplit(n_splits=1, test_size=<<your code goes here>>, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
Finally, we will drop the income_cat column from both the train and test set since it is the attribute that our model will predict. For this we will use the drop method

for set_ in (strat_train_set, strat_test_set):
    set_.<<your code goes here>>("income_cat", axis=1, inplace=True)
  
  ** Visualize the geographic distribution of the data**
  
In this step we will visualize how the income categories are distributed geographically. This will give us a better understanding of how the housing prices are very much related to the location (e.g., close to the ocean) and to the population density. We will do this by creating a scatter plot.

INSTRUCTIONS
First, create a copy of the strat_train_set dataset and save it in the housing variable using the copy method

housing = strat_train_set.<<your code goes here>>()
Now let's plot the scatter plot using Matplotlib as shown below. Please copy the code as is.

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
Here we are using the imread method to load the PNG image of California that is set as a background to the scatter plot. The xlabel and ylabel methods sets the labels for x- and y-axis. We show the scatter plot using imshow method where we have used the cmap parameter to fix the color map, this is used to map scalar data to colors. The linspace method returns evenly spaced numbers over a specified interval.
  
  **Create a correlation matrix**
  
Now, we will create a correlation matrix to see the correlation coefficients between different variables. The correlation coefficient is a statistical measure of the strength of the relationship between the relative movements of two variables.

If 2 variables are positively correlated then if one of the variables increase, the other one increases along with it. If they are negatively correlated then if one of the variables increase, the other one decreases along with it. However, we must note that even if 2 variables are positively/negatively correlated, it does not always mean that one of them is increasing/decreasing because of the other one which is defined by the phrase "correlation does not imply causation".

We will also create 3 new features from the existing features in the dataset.

INSTRUCTIONS
  
First we will create 3 new features from the existing features as shown below

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
Now let's calculate the correlation coefficient of all the variables using the corr method

corr_matrix = housing.corr()
Now, let's plot the correlation matrix of all the features. First, we will sort the values using the sort_values method, then we will plot a scatter plot using the scatter_matrix method from Pandas

corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
Finally, let's get more information on the updated dataset with the new added features using the describe method

housing.describe()
  
  ** Fill in the missing data**
  
When you were exploring the dataset, you must have noticed that some of the features had missing data.

INSTRUCTIONS
  
We will revert to a clean training set that we got after we used StratifiedShuffleSplit and drop the median_house_value since it is the label that we will predict

housing = strat_train_set.<<your code goes here>>("median_house_value", axis=1)
Now we will store the labels in housing_labels variable

<<your code goes here>> = strat_train_set["median_house_value"].copy()
Now we will impute the missing values using the SimpleImputer class. First, import the SimpleImputer class from sklearn

from sklearn.impute import <<your code goes here>>
Now, for the missing values we will consider the median value for that feature. We are not considering mean since median is a better measure of central tendency as it takes into account the outliers. We will set the strategy parameter to "median" in the SimpleImputer class

imputer = SimpleImputer(<<your code goes here>>="median")
Now let's drop the categorical column ocean_proximity because median can only be calculated on numerical attributes

housing_num = housing.drop("ocean_proximity", axis=1)
We will use fit on the housing_num dataset

imputer.<<your code goes here>>(housing_num)
Now we will use transform the training set

X = imputer.<<your code goes here>>(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                      index=housing.index)
  
  ** Handling categorical attributes**
  
So far we have only dealt with numerical attributes, but now let’s look at text attributes. In this dataset, there is just one: the ocean_proximity attribute. A Machine Learning model does not understand categorical values, so we will turn this into a numerical value using onehot encoding.

Onehot encoding creates one binary attribute per category: one attribute equal to 1 when the category is <1H OCEAN (and 0 otherwise), another attribute equal to 1 when the category is INLAND (and 0 otherwise), and so on.

Notice that the output is a SciPy sparse matrix, instead of a NumPy array. This is very useful when you have categorical attributes with thousands of categories. After onehot encoding, we get a matrix with thousands of columns, and the matrix is full of 0s except for a single 1 per row. Using up tons of memory mostly to store zeros would be very wasteful, so instead a sparse matrix only stores the location of the nonzero elements.

Let's see how it is done.

INSTRUCTIONS
First, we will store the categorical feature in a new variable called housing_cat

<<your code goes here>> = housing[["ocean_proximity"]]
Let's see what it looks like using the head method

housing_cat.<<your code goes here>>(10)
Now let's import OneHotEncoder from sklearn

from sklearn.preprocessing import <<your code goes here>>
Now we will fit_transform our categorical data

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.<<your code goes here>>(housing_cat)
housing_cat_1hot
Finally, we will convert it to a dense Numpy array using toarray method

housing_cat_1hot.<<your code goes here>>()
  
  ** Creating custom transformer**
  
Now we will create a custom transformer to combine the attributes that we created earlier.

INSTRUCTIONS
First we will import BaseEstimator, and TransformerMixin classes from sklearn

from sklearn.base import BaseEstimator, TransformerMixin
Now, copy paste the code given below as is

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
In this example the transformer has one hyperparameter, add_bedrooms_per_room, set to True by default (it is often helpful to provide sensible defaults). This hyperparameter will allow you to easily find out whether adding this attribute helps the Machine Learning algorithms or not
  
  **Creating transformation pipelines**
  
As you have seen, there are many data transformation steps that need to be executed in the right order. Fortunately, Scikit-Learn provides the Pipeline class to help with such sequences of transformations.

INSTRUCTIONS
Copy paste the code below as is. Here we are using a pipeline to process the data by first imputing it using SimpleImputer, then using the custom transformer created earlier to merge the columns, and finally, use the StandardScaler class to scale the entire training data

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
Finally, we will fit_transform the entire training data

housing_prepared = full_pipeline.<<your code goes here>>(housing)
  
  **Train a Decision Tree model**
  
Now that we have prepared the data, we will train a Decision Tree model on that data and see how it performs. Since this is a regression problem, we will use the DecisionTreeRegressor class from Scikit-learn.

INSTRUCTIONS
Import the DecisionTreeRegressor class from Scikit-learn

from sklearn.tree import <<your code goes here>>
Now let's train the DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
To evaluate the performance of our model, we will import the mean_squared_error class from Scikit-learn

from sklearn.metrics import <<your code goes here>>
Now let's predict using our model using the predict method

housing_predictions = tree_reg.<<your code goes here>>(housing_prepared)
Finally, let's evaluate our model

tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
If your trained your model correctly, the rmse would come to 0.0. This means that our model is overfitting. How to resolve this issue? We will come to that in a bit, but before that we will train a Random Forest model.
  
  **Train a Random Forest model**
  
Now let's train a Random Forest model the same way we trained the Decision Tree model and see how it performs.

INSTRUCTIONS
Import the RandomForestRegressor class from Scikit-learn

from sklearn.ensemble import <<your code goes here>>
Now let's train the model with our training data

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.<<your code goes here>>(housing_prepared, housing_labels)
Now we will predict using out model

housing_predictions = forest_reg.<<your code goes here>>(housing_prepared)
Finally, we will evaluate our model

forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
  
  **Fine tune your model with Cross Validation**
  
In this step we will fine tune our models using cross validation. It is a resampling technique that is used to evaluate machine learning models on a limited data sample.

A test set should still be kept aside for final evaluation. We would no longer need a validation set (which is sometimes called the dev set) while doing cross validation. The training set is split into k smaller sets (there are other approaches too, but they generally follow the same principles). The following procedure is followed for each of the k folds:

A model is trained using k-1 of the folds as training data
The resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy)
The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.
  
Now let's work on fine tuning our models using cross validation.

INSTRUCTIONS
First, let's define a function called display_scores that would display the scores, mean, and standard deviation of all the models after applying cross validation

def <<your code goes here>>(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
Now let's import cross_val_score from Scikit-learn

from sklearn.model_selection import <<your code goes here>>
Now let's calculate the cross validation scores for our Decision Tree model

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
Finally, let's calculate the cross validation scores for our Random Forest model

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
  
  ** Fine tune your model with Grid Search**
  
We will further fine tune our models using hyper parameter tuning through GridSeachCV. It loops through predefined hyperparameters and fit your estimator (model) on your training set. After this you can select the best set of parameters from the listed hyperparameters to use with your model.

INSTRUCTIONS
First we will import GridSearchCV from Scikit-learn

from sklearn.model_selection import <<your code goes here>>
  
Then we will define a set of various n_estimators and max_features in your model. First it will try a set of 3 n_estimators and 4 max_features giving a total of 12 combination of parameters. Next it will set the bootstrap hyperparameter as False and try a combination of 6 different hyperparameters as shown below

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
Now we will use these combination of hyperparameters on our Random Forest model

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
Now let's see the best combination of parameters

grid_search.best_params_
And the best combination of estimator

grid_search.best_estimator_
Finally, let's computer the results and print the scores

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
  
  ** Analyze and evaluate best model**
  
Finally, we will make predictions using our model. We will also evaluate those predictions, which is very important so that we can determine how good our model is in predicting attributes it has not seen based on the training it got so far.

INSTRUCTIONS
  
Set the best_estimator_ that we got from the GridSearchCV results to a variable named final_model

<<your code goes here>> = grid_search.best_estimator_
  
Now let's drop the labels from the test set that our model will be predicting, save the attributes in a variable called X_test and save the labels in another variable called y_test

X_test = strat_test_set.drop("median_house_value", axis=1)
<<your code goes here>> = strat_test_set["median_house_value"].copy()
Now let's pass the attributes through the pipeline we created for our model earlier

X_test_prepared = full_pipeline.transform(X_test)
Now, let's make predictions based on the set of data we got from the pipeline and save it in a variable called final_predictions

final_predictions = final_model.predict(X_test_prepared)
Finally, let's compare the predictions of our model against the actual data and see how it performs

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
Now let's print the final result

final_rmse

 
 
