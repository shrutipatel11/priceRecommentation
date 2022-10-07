Link to the [github.io page](https://shrutipatel11.github.io/priceRecommentation/)

# Introduction/Background
Housing price instability was an important cause of the 2007-2008 US recession[1]. Predicting property’s value to forecast such depressions was consequential thereafter. Some of the past works include regression techniques[2][3], and dimensionality reduction to improve house prediction model’s accuracy[4]. We are using the Ames Housing Dataset[5] which describes sales of individual housing properties in Ames, Iowa in 2006-2010. This dataset contains 2930 observations and 80 explanatory features such as property’s sale price, finish quality, construction, and remodeling year. It provides a comprehensive understanding of factors that influence housing price negotiations.


# Problem
The economy of a country is sensitive to the residential market. When the property prices increase, the owners spend more which in turn encourages construction and hence employment. However, during a recession, the prices decline which depress construction spending leading to a decline in employment. Thus, housing prices not only affect the buying/selling trends, but also can reflect current economic conditions. Using ML techniques, we aim to provide a predictive model to forecast the housing prices of a property and in turn predict the economic situation.

# Methods
Our strategy is to utilize a variety of supervised and unsupervised models to forecast home prices, and evaluate each one's performance using a set of measures (R-Square, Root Mean Square Error (RMSE), etc). We intend to progressively implement algorithms, evaluate the metrics, select the next model to employ and finally arrive at an optimized suitable model for house-price prediction.


## **Unsupervised Learning**
House price prediction is normally done using supervised learning. But, we aim to use unsupervised learning to form clusters of similar homes and then use the actual prices to find the mean of a cluster and price the new houses that belong to the cluster.
We will employ different methods (K-Means, Gaussian Mixture Model, DBSCAN) to form clusters and compare their efficacies. Since we form clusters of houses that share similar features, we will use the same to recommend houses using collaborative filtering / content-based filtering (if time permits).


## **Supervised Learning**
We will also perform supervised learning on models to predict the price of homes. There are several model options that we can use to perform regression on this dataset. We will start from some basic models (Linear Regression, Support Vector Regression, Random Forests, Simple Neural Network) and evaluate the results. We do not expect the simple models to give us the best predictions for this dataset given the large number of features. Thus, we can then try more complex techniques like Gradient Boosting (XGBoost, LightGBM), Lasso/Ridge Regression and Stacking/Ensembling multiple models.

# Potential Results and Discussion
Through extensive data pre-processing and feature extraction, we seek to select the most relevant features from among the 80 to reduce the dimensionality for our models. Using these selected features, we employ supervised and unsupervised learning to perform prediction of house prices. If time permits, we also seek to evaluate the generalizability of our model on other house price datasets with similar features. 


# References
1.  Congressional Research Service: Introduction to U.S. Economy: Housing Market- https://sgp.fas.org/crs/misc/IF11327.pdf. 2021 May 3
2.  P. Durganjali and M. V. Pujitha, "House Resale Price Prediction Using Classification Algorithms," 2019 International Conference on Smart Structures and Systems (ICSSS), 2019, pp. 1-4
3.  V. S. Rana, J. Mondal, A. Sharma and I. Kashyap, "House Price Prediction Using Optimal Regression Techniques," 2020 2nd International Conference on Advances in Computing, Communication Control and Networking (ICACCCN), 2020, pp. 203-208
4.  S. B. Sakri and Z. Ali, "Analysis of the Dimensionality Issues in House Price Forecasting Modeling," 2022 Fifth International Conference of Women in Data Science at Prince Sultan University (WiDS PSU), 2022, pp. 13-19
5.  De Cock D. Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project. Journal of Statistics Education. 2011 Nov 1;19(3)
6.  Lee, Anthony J.T.; Lin, Ming-Chih; Kao, Rung-Tai; and Chen, Kuo-Tay, "An Effective Clustering Approach to Stock Market Prediction" (2010). PACIS 2010 Proceedings. 54.

# Timeline
The detailed timeline can be found in [Gantt Chart](https://docs.google.com/spreadsheets/d/1az16wonnse66ozJ7zGFMCuXEZcn8mAw1/edit?usp=sharing&ouid=105426710738045720116&rtpof=true&sd=true)

| Process                                                      | Date Range        |
|:-------------------------------------------------------------|:------------------|
| Data Sourcing and Cleaning                                   | 10/7 - 10/12      |
| Model Selections and Data Pre-processing (supervised)        | 10/12 - 10/20     |
| Implement Supervised Learning Models and Analysis            | 10/20 -  11/11    |
| Model Selections and Data Pre-processing (unsupervised)      | 11/11 - 11/18     |
| Implement Unsupervised Learning Models and Analysis          | 11/18 - 11/30     | 
| Model Comparisons and Analysis                               | 11/30 - 12/4      |
| Final Proposal & Presentation                                | 12/4 - 12/6       |

# Contribution
| Process                           | Name      |
|:----------------------------------|:----------|
| Problem Selection                 | All       |
|Introduction & Background          | Navdha    |
|Problem Definition                 | Navdha    |
|Methods (Supervised)               | Dhruv     |
|Methods (Unupervised)              | Anirudh   |
|Potential Results & Discussion     | Aayushi   |
|Presentation                       | All       |
|Video Recording                    | Shruti    |
|GitHub Page                        | Shruti    |   
