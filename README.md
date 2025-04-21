# Overview
This project aims to perform an in-depth data analysis on a professional-grade movie dataset to uncover meaningful insights into movie ratings, revenue trends, genre popularity, and audience preferences. Leveraging The Movies Dataset from Kaggle, which aggregates data from The Movie Database (TMDb), this analysis integrates multiple data sources, including metadata, user ratings, and financial metrics like budget and revenue.

The primary goal is to explore relationships between various movie features such as genre, runtime, popularity, budget, and release date also their influence on audience ratings and box office performance. Additionally, the project examines long-term trends across the film industry and highlights key characteristics of top-performing movies.

Data Source: [Kaggle: Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

# Questions

Below are the questions to be answer in this project:
1. What genres tend to receive the highest average ratings?
2. How does budget relate to box office revenue?
3. Are there any notable trends in genre popularity or movie ratings over time?
4. Which movies are outliers in terms of profitability or audience reception?
5. What are the most common characteristics of successful movies?

# Skill and Tools Used
To analyze the datasets I used several key tools:

- **Python:** The backbone of my analysis, allowing me to analyze the data and find critical insights.I also used the following Python libraries:
    - **Pandas Library:** This was used to analyze the data. 
    - **Matplotlib Library:** I visualized the data.
    - **Seaborn Library:** Helped me create more advanced visuals. 
- **Jupyter Notebooks:** The tool I used to run my Python scripts which let me easily include my notes and analysis.
- **Visual Studio Code:** My go-to for executing my Python scripts.
- **Git & GitHub:** Essential for version control and sharing my Python code and analysis, ensuring collaboration and project tracking.

# Data Preperation and Cleanup

This section contains all the steps performed to prepare the data for analysis ensuring accuracy and usability.

## Import and Clean Up Data

Download [Kaggle: Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Refer to Jupyter Notebook: [1_Data_Importing_Cleaning](./1_Analysis/1_Data_Importing_Cleaning.ipynb)

### Importing Data and Libraries

Import the libraries needed for the project then turn the dataset to a dataframe using `pandas`.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

movies_df = pd.read_csv('../movies_metadata.csv', low_memory= False)
```

### Cleaining Data

 When inspecting the dataset using `movies_df.info()`, I found that the `release_date` column is in type `object`, and `budget` is a `string`. To convert these columns to their appropriate data types, I used `pandas.to_datetime()` for `release_date` and `pandas.to_numeric()` for `budget`.


```python
# Convert data type of release_date to datetime.
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'],errors='coerce') 

# Convert data type of budget to numeric.
movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce') 
```

The columns `genres`, `production_companies`, and `production_countries` are stored as strings, but they actually represent lists of dictionaries. Each dictionary contains keys like `id`, `name`, and possibly others.
To extract just the names (e.g. genre names, country names, or company names), I wrote a function that first converts the string into an actual list using `ast.literal_eval()`. Then, it loops through each dictionary in the list and collects the value associated with the `name` key into a new list.


```python
# Convert the string list of dictionary into list of name.
def list_name(string_list_dict): 
    list_dict = ast.literal_eval(string_list_dict) if pd.notna(string_list_dict) else string_list_dict
    return [dict['name'] for dict in list_dict] if isinstance(list_dict,list) else []

movies_df['genres'] = movies_df['genres'].apply(lambda x: list_name(x))
movies_df['production_companies'] = movies_df['production_companies'].apply(lambda x: list_name(x))
movies_df['production_countries'] = movies_df['production_countries'].apply(lambda x: list_name(x))
```

Since our focus is on movies produced in the United States, we filtered out films made in other countries. Additionally, we narrowed the dataset to include only those released between the 1990s and 2020.

```python
# Filtering all the movies that was produced in United States.
movies_df = movies_df.explode('production_countries')
movies_df_US = movies_df[movies_df['production_countries'] == "United States of America"].copy() 
```

### Output: 

```html
<class 'pandas.core.frame.DataFrame'>
Index: 21153 entries, 0 to 45463
Data columns (total 24 columns):
 #   Column                 Non-Null Count  Dtype         
---  ------                 --------------  -----         
 0   adult                  21153 non-null  object        
 1   belongs_to_collection  2573 non-null   object        
 2   budget                 21153 non-null  float64       
 3   genres                 21153 non-null  object        
 4   homepage               4107 non-null   object        
 5   id                     21153 non-null  object        
 6   imdb_id                21150 non-null  object        
 7   original_language      21149 non-null  object        
 8   original_title         21153 non-null  object        
 9   overview               21116 non-null  object        
 10  popularity             21153 non-null  object        
 11  poster_path            21075 non-null  object        
 12  production_companies   21153 non-null  object        
 13  production_countries   21153 non-null  object        
 14  release_date           21147 non-null  datetime64[ns]
 15  revenue                21153 non-null  float64       
 16  runtime                21148 non-null  float64       
 17  spoken_languages       21153 non-null  object        
 18  status                 21139 non-null  object        
 19  tagline                14003 non-null  object        
 20  title                  21153 non-null  object        
 21  video                  21153 non-null  object        
 22  vote_average           21153 non-null  float64       
 23  vote_count             21153 non-null  float64       
dtypes: datetime64[ns](1), float64(5), object(18)
memory usage: 4.0+ MB
```
