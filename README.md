# Predicting Box Office Success Based on Early Rotten Tomatoes Scores and Social Media Sentiment

## Overview

This project aims to build a predictive model for estimating a movie's box office performance within its first week of release.  The model leverages early Rotten Tomatoes scores (both critics and audience) and aggregated social media sentiment (e.g., from Twitter) as key predictors.  The analysis explores the correlation between these factors and box office revenue, ultimately aiming to provide studios with a tool to better allocate marketing resources based on early indicators of potential success.

## Technologies Used

This project utilizes the following Python libraries:

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Tweepy (for social media data acquisition - if applicable.  Remove if not used)


## How to Run

1. **Install Dependencies:**  Ensure you have Python 3 installed.  Navigate to the project directory in your terminal and install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

   *Note:* You may need to adjust data paths within `main.py` to reflect your local file structure if you are using your own dataset.


## Example Output

The script will print key statistical analysis to the console, including model performance metrics (e.g., R-squared, RMSE).  Additionally, the script generates visualizations, such as:

* A scatter plot illustrating the relationship between Rotten Tomatoes scores and box office revenue.
* A plot showing the predictive model's performance (e.g., predicted vs. actual box office revenue).

These plots are saved as PNG files in the `output` directory (create this directory if it doesn't exist).  For example, `output/sales_trend.png` might display the sales trend prediction.  The specific plots generated may vary depending on the analysis performed.


## Data

This project requires a dataset containing movie titles, Rotten Tomatoes scores (critic and audience), social media sentiment scores, and first-week box office revenue.  The data used for this project can be [Insert data source or explanation of data acquisition here, e.g., "found publicly available on Kaggle" or "scraped from Rotten Tomatoes and Twitter"].  The data preprocessing steps are detailed within the Jupyter Notebook (`data_preprocessing.ipynb` - if applicable).