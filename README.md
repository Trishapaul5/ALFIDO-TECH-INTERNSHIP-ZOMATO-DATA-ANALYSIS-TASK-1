Zomato Data Analysis

This repository contains a Data Science project focused on analyzing the Zomato dataset to enhance practical Data Science skills. The project involves data cleaning, visualization, and applying machine learning techniques to predict restaurant ratings.

Objective

Enhance Data Science skills by working on practical tasks using real-world datasets. Learn data cleaning, visualization, and machine learning basics.

Features





Hands-on Data Analysis and Visualization: Perform exploratory data analysis and create visualizations to understand the Zomato dataset.



Working with Real-World Datasets: Analyze a real-world dataset from Zomato to extract meaningful insights.



Exploring Machine Learning Models: Build and evaluate a Random Forest Regressor to predict restaurant ratings.

Technologies





Python: Core programming language for the project.



Pandas, NumPy: Libraries for data manipulation and numerical operations.



Matplotlib, Seaborn: Tools for data visualization.



Scikit-Learn: Framework for implementing machine learning models.

Files





zomato_data_analysis.py: Main Python script containing the data analysis, visualization, and machine learning code (originally a Kaggle Notebook converted to .py).



cleaned_zomato.csv: Cleaned version of the Zomato dataset after preprocessing.



ratings_distribution.png: Histogram showing the distribution of restaurant ratings.



cost_vs_rating.png: Scatter plot of average cost for two people vs. restaurant rating.



feature_importance.png: Bar plot showing feature importance for the Random Forest model.



online_order_vs_rating.png: Box plot comparing ratings by online order availability.



top_cuisines.png: Bar plot of the top 5 cuisines on Zomato.



top_rest_types.png: Bar plot of the top 5 restaurant types on Zomato.

How to Run





Prerequisites:





Python 3.x installed on your system.



Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn



Dataset:





The script expects the Zomato dataset at the path /kaggle/input/zomato/zomato.csv.



If running locally, download the dataset from Kaggle Zomato Dataset and place it in the appropriate directory, or update the data_path variable in the script to point to the dataset’s location.



Run the Script:





Execute the Python script in your environment:

python zomato_data_analysis.py



Alternatively, if you have the original .ipynb notebook, you can run it in Jupyter Notebook or Kaggle:

jupyter notebook zomato-analysis.ipynb



Outputs:





The script generates visualizations saved as PNG files in the working directory.



A cleaned dataset (cleaned_zomato.csv) is saved after preprocessing.

Project Structure





Data Loading and Exploration:





Loads the Zomato dataset and explores its structure.



Identifies missing values and unique entries in key columns.



Data Cleaning:





Cleans the rate column by handling invalid entries (e.g., 'NEW', review text).



Converts numerical columns (votes, approx_cost(for two people)) to appropriate formats.



Fills missing values using medians for numerical columns and modes for categorical columns.



Data Visualization:





Creates visualizations to understand the dataset:





Distribution of ratings.



Top 5 cuisines and restaurant types.



Relationship between cost and rating.



Impact of online ordering on ratings.



Machine Learning:





Encodes categorical variables using LabelEncoder.



Trains a Random Forest Regressor to predict restaurant ratings.



Evaluates the model using Mean Squared Error (MSE) and R² Score.



Visualizes feature importance.

Results





Model Performance:





Mean Squared Error (MSE): ~0.1–0.3 (varies based on dataset).



R² Score: ~0.6–0.8, indicating a decent fit for predicting ratings.



Key Insights:





Most restaurant ratings are clustered around 3.5–4.0.



Popular cuisines include North Indian and Chinese.



Votes and location are significant predictors of ratings.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments





The Zomato dataset is sourced from Kaggle.



Built as part of a Data Science learning task to enhance practical skills.
