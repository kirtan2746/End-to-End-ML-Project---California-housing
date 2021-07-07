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

# Load the dataset
  
/cxldata/datasets/project/housing/housing.csv

INSTRUCTIONS
Set HOUSING_PATH variable to the path of the dataset as given above

# Explore the dataset
  
Now we will explore the dataset. Here, we will be using the hist method to plot a histogram to view the data. A histogram is used to visually represent the distribution of the data instead of the actual data itself, simply put, it is used to summarize discrete or continuous data.

The hist methods here has the bins parameter. These are also sometimes referred to as classes, intervals, or `buckets, are groups of equal widths into which the data is separated. Each bin is plotted as a bar whose height corresponds to how many data points are in that bin.

We will also be using the cut method from Pandas. This is used to segment and sort data values into bins. cut is also helpful for converting from a continuous variable to a categorical variable. For example, cut could convert ages to groups of age ranges. Supports binning into an equal number of bins, or a pre-specified array of bins. The labels parameters here specifies the labels for the returned bins. It has to be of the same length as the resulting bins. Also, if you notice, we have mentioned a np.inf here for the bins. That is a form of floating point representation of infinity.

# Split the dataset
  
In this step, we will split the dataset into train and test sets. We will be using the StratifiedShuffleSplit method from the sklearn library which is a cross-validator that provides train/test indices to split data in train/test sets.

# Visualize the geographic distribution of the data
  
In this step we will visualize how the income categories are distributed geographically. This will give us a better understanding of how the housing prices are very much related to the location (e.g., close to the ocean) and to the population density. We will do this by creating a scatter plot.

Here we are using the imread method to load the PNG image of California that is set as a background to the scatter plot. The xlabel and ylabel methods sets the labels for x- and y-axis. We show the scatter plot using imshow method where we have used the cmap parameter to fix the color map, this is used to map scalar data to colors. The linspace method returns evenly spaced numbers over a specified interval.

# Create a correlation matrix
  
Now, we will create a correlation matrix to see the correlation coefficients between different variables. The correlation coefficient is a statistical measure of the strength of the relationship between the relative movements of two variables.

If 2 variables are positively correlated then if one of the variables increase, the other one increases along with it. If they are negatively correlated then if one of the variables increase, the other one decreases along with it. However, we must note that even if 2 variables are positively/negatively correlated, it does not always mean that one of them is increasing/decreasing because of the other one which is defined by the phrase "correlation does not imply causation".

We will also create 3 new features from the existing features in the dataset.

INSTRUCTIONS
  
First we will create 3 new features from the existing features as shown below

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
Now let's calculate the correlation coefficient of all the variables using the corr method

# Fill in the missing data
  
When you were exploring the dataset, you must have noticed that some of the features had missing data.

INSTRUCTIONS
  
We will revert to a clean training set that we got after we used StratifiedShuffleSplit and drop the median_house_value since it is the label that we will predict

# Handling categorical attributes
  
So far we have only dealt with numerical attributes, but now let’s look at text attributes. In this dataset, there is just one: the ocean_proximity attribute. A Machine Learning model does not understand categorical values, so we will turn this into a numerical value using onehot encoding.

Onehot encoding creates one binary attribute per category: one attribute equal to 1 when the category is <1H OCEAN (and 0 otherwise), another attribute equal to 1 when the category is INLAND (and 0 otherwise), and so on.

Notice that the output is a SciPy sparse matrix, instead of a NumPy array. This is very useful when you have categorical attributes with thousands of categories. After onehot encoding, we get a matrix with thousands of columns, and the matrix is full of 0s except for a single 1 per row. Using up tons of memory mostly to store zeros would be very wasteful, so instead a sparse matrix only stores the location of the nonzero elements.

# Creating custom transformer
  
Now we will create a custom transformer to combine the attributes that we created earlier.

INSTRUCTIONS
First we will import BaseEstimator, and TransformerMixin classes from sklearn

# Creating transformation pipelines
  
As you have seen, there are many data transformation steps that need to be executed in the right order. Fortunately, Scikit-Learn provides the Pipeline class to help with such sequences of transformations.

INSTRUCTIONS
Copy paste the code below as is. Here we are using a pipeline to process the data by first imputing it using SimpleImputer, then using the custom transformer created earlier to merge the columns, and finally, use the StandardScaler class to scale the entire training data

# Train a Decision Tree model
  
Now that we have prepared the data, we will train a Decision Tree model on that data and see how it performs. Since this is a regression problem, we will use the DecisionTreeRegressor class from Scikit-learn.

INSTRUCTIONS
Import the DecisionTreeRegressor class from Scikit-learn

# Train a Random Forest model
  
Now let's train a Random Forest model the same way we trained the Decision Tree model and see how it performs.

INSTRUCTIONS
Import the RandomForestRegressor class from Scikit-learn

# Fine tune your model with Cross Validation
  
In this step we will fine tune our models using cross validation. It is a resampling technique that is used to evaluate machine learning models on a limited data sample.

A test set should still be kept aside for final evaluation. We would no longer need a validation set (which is sometimes called the dev set) while doing cross validation. The training set is split into k smaller sets (there are other approaches too, but they generally follow the same principles). The following procedure is followed for each of the k folds:

A model is trained using k-1 of the folds as training data
The resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy)
The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.
  
# Fine tune your model with Grid Search
  
We will further fine tune our models using hyper parameter tuning through GridSeachCV. It loops through predefined hyperparameters and fit your estimator (model) on your training set. After this you can select the best set of parameters from the listed hyperparameters to use with your model.


# Analyze and evaluate best model
  
Finally, we will make predictions using our model. We will also evaluate those predictions, which is very important so that we can determine how good our model is in predicting attributes it has not seen based on the training it got so far.



 
 
