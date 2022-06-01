# Challenge 10 Machine Learning - Unsupervised Learning - Kmeans - Elbow method - Principal Contribution Analysis (PCA) - HVPLOT - Scatter/Hover/Line
The purpose of this effort is to take large data sets with multiple variables and utilize state of the art machine learning techniques to normalize (place each element on the same scale), cluster the data, sythesize the clusters by combining the common behaviors and reducing the data sets to generalized clusters that can be used to distinguish particular features of the data.  In our example, we will take 41 different cryptocurrencies and data from those currencies taken over sequentially longer timeframes showing the percentage change in price and compare the currencies along with the timeframes in a simplified data set that can be split into smaller reasonably sized data sets and separated to find the elements that would make an optimal combined portfolio of currencies based upon their performances over these varied timeframes.  The dynamic feature of this Artificial Intelligence application is the forward looking predictive abilities of synthesizing the backward looking data.    

I will be the first to say that this all sounds a bit like voodoo but that large multinational megastocks like the FAANG stocks: Facebook, Amazon, Applie Netflix and Google use these techniques daily and are experiencing dramatic and profitable results.  The Module begins with some interesting disclaimers that I will utilize in my writeup.  In the first paragraph, it states,  


    *"Several theoretical aspects behind machine learning go beyond the scope of this module."* 

Additional research was necessary to complete the overall theory summary relayed in the first paragraph.  I certainly know how to code the features with success.  But, it will take further study to grasp the full essence of the theoretical foundation of machine learning. I have found a reasonable set of resources in, of all places, YouTube.  The University of YouTube was instrumental in demostrating some of the practical applications of machine learning and more specifically the KMeans, PCA, elbow and clustering functionality.  One particular channel that I found very useful was Edureka on YouTube.  

Hvplot was used to graph the results along the way.  Visual interpretation was vital in all of the steps.  As I researched the various methods available through Hvplot I recognized that here I've also only scratched the surface.  

There were numerous steps to be undertaken in this process in this Challenge.  These are:

1. *Import the cryptocurrency dataset with periodic performance metrics.*

2. *Identify which periodic performance metrics relate the most to clustering.*

3. *Find a reasonable cluster size.*

4. *Cluster the data by using the K-means algorithm on the most important performance metrics.*

5. *Graph and interpret the results.*

Within Machine Learning, there are three broad categories of models that can be employed:

1. *Supervised Learning - these models take labeled datasets where each example in the dataset is tagged with the answer.  This provides an answer key that can be used to evaluate accuracy of the training data.*

2. *Unsupervised Learning - The algorithm tries to make sense of an unlabeled dataset by extracting features and patterns on its own.*

3. *Reinforcement Learning - The model attempts to find the optimal way to accomplish a goal or complete a task.  As the algorithm improves in achieving that goal, it receives a reward*

###Source: Bootcampspot.com

In our model, we will employ Unsupervised Learning.  

The graphic below shows the elements of the different types of learning and some of the methods and applications of each:

![Graphic - Learning methods](https://github.com/Cryptopher2022/Challenge_10a/blob/main/images/Graphic%20with%20Supervised%2C%20Unsupervised%20and%20Reinforcement%20Learning.png)

---

## Technologies

This project was completed almost entirely in Jupyter Notebook.  The README.md was modified in VS Code.  

There were numerous libraries used in this project:
import pandas as pd (pandas)
import hvplot.pandas (hvplot)
from pathlib import Path (Path)
from sklearn.cluster import KMeans (KMeans)
from sklearn.decomposition import PCA (PCA)
from sklearn.preprocessing import StandardScaler (StandardScaler)

More information can be found regarding each of these libraries at the following links:

pandas - https://pandas.pydata.org/

hvplot - https://hvplot.holoviz.org/

Path from pathlib - https://docs.python.org/3/library/pathlib.html

KMeans - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

PCA - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

StandardScaler - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html


This program was written and will run on Windows 10.  

---

## Installation Guide

In this section, you should include detailed installation notes containing code blocks and screenshots.

From the Command Line in Git Bash, navigate your directory to the location of the file package.  Then, type "Jupyter Notebook" to launch the application used to write and run this program.  It's outside of the scope of this README.md file to explain the installation of Jupyter Notebook.  

From Jupyter Notebook, navigate to the directory where the program is found and click on the program: "crypto_investments.ipynb".  
---

## Usage



As it was explained in the preamble, the goal of this kind of analysis is to:
1. Import the cryptocurrency dataset with periodic performance metrics.  *This was accomplished using the read_csv function and imported a CSV file from the Resources folder.  The index was set as the coin_id or the name or the symbol for each currency.*

2. Identify which periodic performance metrics relate the most to clustering.  *A line plot was drawn using hvplot to show the raw data.  !![Plot](https://github.com/Cryptopher2022/Challenge_10a/blob/main/images/Line%20Plot%20-%20raw%20data.png)
Prepare the Data - This section prepares the data before running the K-Means algorithm. It follows these steps:

        a. Use the StandardScaler module from scikit-learn to normalize the CSV file data. This will require you to utilize the fit_transform function.

        b. Create a DataFrame that contains the scaled data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.

Find the Best Value for k Using the Original Data
In this section, you will use the elbow method to find the best value for k.

        a. Code the elbow method algorithm to find the best value for k. Use a range from 1 to 11.

        b. Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.

Cluster Cryptocurrencies with K-means Using the Original Data
In this section, you will use the K-Means algorithm with the best value for k found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.

        a. Initialize the K-Means model with four clusters using the best value for k.

        b. Fit the K-Means model using the original data.

        c. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.

        d. Create a copy of the original data and add a new column with the predicted clusters.

        e. Create a scatter plot using hvPlot by setting x="price_change_percentage_24h" and y="price_change_percentage_7d". Color the graph points with the labels found using K-Means and add the crypto name in the hover_cols parameter to identify the cryptocurrency represented by each data point.

![Scatter1](https://github.com/Cryptopher2022/Challenge_10a/blob/main/images/Scatter1.png)
Optimize Clusters with Principal Component Analysis
In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.

        a. Create a PCA model instance and set n_components=3.

        b. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame.

        c. Retrieve the explained variance to determine how much information can be attributed to each principal component.

        d. Answer the following question: What is the total explained variance of the three principal components?

        e. Create a new DataFrame with the PCA data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.

        f. Answer the following question: What is the best value for k?
            #The answer is 4. 

![PCA1](https://github.com/Cryptopher2022/Challenge_10a/blob/main/images/New%20PCA1.png)
Find the Best Value for k Using the PCA Data
In this section, you will use the elbow method to find the best value for k using the PCA data.

        a. Code the elbow method algorithm and use the PCA data to find the best value for k. Use a range from 1 to 11.

        b. Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.

        c. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
                Answer: It does not differ

![K2](https://github.com/Cryptopher2022/Challenge_10a/blob/main/images/K2.png)
### Cluster Cryptocurrencies with K-means Using the PCA Data

In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.

        1. Initialize the K-Means model with four clusters using the best value for `k`. 

        2. Fit the K-Means model using the PCA data.

        3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.

        4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.

        5. Create a scatter plot using hvPlot by setting `x="PC1"` and `y="PC2"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

![Scatter2](https://github.com/Cryptopher2022/Challenge_10a/blob/main/images/New%20Scatter%202.png)
Visualize and Compare the Results
In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

        a. Create a composite plot using hvPlot and the plus (+) operator to contrast the Elbow Curve that you created to find the best value for k with the original and the PCA data.

        b. Create a composite plot using hvPlot and the plus (+) operator to contrast the cryptocurrencies clusters using the original and the PCA data.

        c. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
![Combined Elbow]()

![Combined Scatter](https://github.com/Cryptopher2022/Challenge_10a/blob/main/New%20Combined%20But%20separate%20scatters.png)


![Final Scatter](https://github.com/Cryptopher2022/Challenge_10a/blob/main/New%20Combined%20Scatter%20Final%20all.png)

![Final Scatter scaled](https://github.com/Cryptopher2022/Challenge_10a/blob/main/New%20Combined%20Scatter%20Final%20scaled.png)
                Answer: When looking at the columns of data in the prior dataframe from which we started, there is no way to graph a dataframe with more than 3 columns. By reducing the number of columns but not the data contained within, we can use visualization tools to pick out these coins below from one of the quadrants in the scatter plot to make our selections for the crypto currencies to recommend to the board. I understand now how these methods can bring in visualization as a critical part of the overall analysis. Thank you again for giving me the opportunity to repair this work and to learn its value.  I did recognize some interesting data results from the analysis.  The graph below shows the two scatters combined.  The second scatter is zoomed in to the positive quandrants for PC1 and PC2.  These resulted in the following list of cryptocurrencies that should be recommended as the portfolio of choice to the board.  I say this because the results mirror actual results in real life.  The list is as follows:

                     1. Monero*
                     2. bitcoin-cash*
                     3. binancecoin*
                     4. wrapped bitcoin*
                     5. cosmos*
                     6. cardano*
                     7. zcash*
                     8. omisego*
                     9. celsius-degree-token*
                     10. theta-token*
                     11. ethland*
                     12. nem*


    



---

## Contributors

This was done solely by Christopher Todd Garner

---

## License

Feel free to use this program and happy hunting for arbitrage profits.  Add some for loops or the like and optimal profits can be achieved.  
