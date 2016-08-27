#Note on some ML basics
ML is a research subject.  This note only records some important points. ML algorithms are divided into supervised learning and unsupervised learning.
##Mathematics pre-requisite
The mathematics knowledge is linear algebra (matrix product) and calculus (differentiation).
##Supervised Learning
There is a set of **actual data** providing some kind of expected values.  These data is used to train the algorithm to predict a value.  These kinds of algorithms are classified as supervised learning.  
The training steps are:
* Splitting the actual data into training data and validation data, the ratio is ~ 3:1.
* Extracting the feature(s) from the data.  E.g. if the algorithm is predicting the house price, the features will be suburb, house area, house age, etc.
* Identifying the label(s) from the data.  E.g. house price.
* Feeding the training data to the algorithm to tune the best parameters for the algorithm.
* Validating the trained algorithm with the validation data.

### Linear regression
Supervised learning.  It's usually used for prediction.  For the house price example, the features will be suburb, house area, house age, etc.  These will be represented by ```x1```, ```x2```, ```x3```, ....  The weightings of these features will be represented by ```w0```, ```w1```, ```w2```, ```w3```, ...  The label will be represented by ```y```.
Therefore the equation will be ```w0 + w1*x1 + w2*x2 + w3*x3 = y```.  
Suppose there are ```n``` row of training data.  
The features can be represented by the following matrix annotated by **X**:
1 x1(1) x2(1) x3(1)
...
1 x1(n) x2(n) x3(n)
The parameters can be represented by the following vector annotated by **W**:
w0 w1 w2
The label can be represented by the following vector annotated by **Y**:
Therefore by matrix dot product ```X * W = Y```
###Cost function
From the previous, there is difference between the actual ```Y``` values from the calculated ```X * W```.  Cost function is the mean of the square of the difference across the whole training data set.  To minimize the error, **W** should be selected appropriately.
###Gradient descent
It is an iterative algorithm to find the optimum **W**.  It involves taking the differentiation of the cost function against W (gradient).  In each iteration, subtracts the gradient from the W for a new W' to calculate the cost function value.  Repeat the step until the minimum cost function value is found.
### Linear classification
Supervised learning.  The algorithm is used to decide the classified type.  Logistic regression is a particular type of linear classification in which there are 2 types to be classified.  E.g. determing whether an email is spam or not.
###Underfitting, Overfitting and regularization
Underfitting means the algorithm's determined values is largely deviated from the actual data.  Overfitting is the opposite.  But the problem is the algorithm only fits to the training data.  A large error is found when it is validated against the validation set.  Therefore regularization is used to penalize the equation such that it is more general to fit in both training and validation.
##Unsupervised Learning 
Unlike supervised learning, there is no expected values from the data.  The algorithm is used to find the trend in the given data.  Examples of these algorithms are clustering, dimensionality reducation and PCA.
##BigData
To train the algorithm to be more accurate, a huge data set is needed.  Therefore ML drives the need of technologies that handle BigData.
##Spark
Spark has a MLlib for training the algorithm such as ```LinearRegressionWithSGD```, ```LogisticRegressionWithSGD```.
