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

# Data Preperation and Clean-up

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

 When inspecting the dataset using `movies_df.info()`, I found that the `release_date` column is in type `object`, and [`budget`,`id`,`imdb_id`] are `string`. To convert these columns to their appropriate data types, I used `pandas.to_datetime()` for `release_date` and `pandas.to_numeric()` for [`budget`,`id`,`imdb_id`] .


```python
# Convert data type of release_date to datetime.
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'],errors='coerce') 

# Convert data type to numeric.
movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce') 
movies_df['imdb_id'] = pd.to_numeric(movies_df['imdb_id'], errors='coerce')
movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce') 
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
## Exploratory Data Analysis
In this part of the project, I will discuss how to answer each question and provide insights. Each question will include data preparation, data visualization, and key takeaways based on the results.

### What genres tend to receive the highest average ratings?
After cleaning the dataset, I focused on a column containing a list of dictionaries, each holding a genre `name` and its corresponding `id`. I extracted the genre names and transformed the column into a list of genre names. Then, I used the `explode()` function to split the list into separate rows for each genre. Finally, I filtered the dataset to include only movies with a `vote_average` of 8.0 or higher and a `vote_count` greater than 1000—defining these as the highest-rated movies.

```python
hights_rating_movies = movies_df[(movies_df['vote_count'] > 1000) & (movies_df['vote_average'] >= 8.0)]

```

#### Data Visualized

```python
sns.set_theme(style= 'ticks')
sns.barplot(
    data= genres_count
    ,x= 'perc'
    ,y= 'genres'
    ,hue= 'count'
    ,palette= 'dark:b_r'
    ,legend= False
    )

ax= plt.gca()
ax.set_title('Likelihood of a Genre in Top-Rated Movies', fontsize= 15)
ax.set_xlabel('Percentage (%)')
ax.set_ylabel('Genre')
ax.set_xlim(0,100)
ax.spines[['top','right']].set_visible(False)

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,i: f'{x:.0f}%'))
for i, value in enumerate(genres_count['perc']):
    ax.text(
        x= value + 1
        ,y= i
        ,s= f'{value:.1f}%'
        ,va= 'center'
    )
plt.show()
```
#### Result

!['Barchart'](./2_Images/bar_Genres%20in%20Top-rated%20Movies.png)

#### Analysis

- `Drama` dominates top-rated films at 67.8%, highlighting its emotional impact and broad appeal, while mid-tier genres like `Thriller`, `Crime`, and `Adventure` also stand out for their intensity and intrigue; in contrast, niche genres such as `Horror`, `History`, and `Western` are far less common, appearing in only 3.4% of top movies.

### How does budget relate to box office revenue?
First thing to do is filter the dataset and include only movies that have `budget` and `revenue`. Then, using scatterplot show how `revenue` changes as `budget` increases. Add regression line to show the trend, this will help to see if there's a correlation.

```python
movies_budget_revenue = movies_df[(movies_df['budget'] > 0) & (movies_df['revenue'] > 0)][['budget','revenue']].copy()
```
#### Data Visualized

```python
sns.set_theme(style= 'ticks')
sns.scatterplot(
    data= movies_budget_revenue
    ,x= 'budget'
    ,y= 'revenue'
    )

sns.regplot(
    data= movies_budget_revenue
    ,x= 'budget'
    ,y= 'revenue'
    ,scatter= False
    ,color= 'red'
    ,
    )

ax= plt.gca()
ax.set_title('Movie Budget vs Box Office Revenue')
ax.set_xlabel('Budget')
ax.set_ylabel('Revenue')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, i: f'${y/1_000_000_000}B'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, i: f'${x/1_000_000_000}B'))
```
#### Result
!['scatter_Budget vs Revenue'](./2_Images/scatter_Movie%20Budget%20vs%20Box%20Office%20Revenue.png)
*Scatterplot shows the relationship of Movie Budget and Box Office Revenue*

#### Analysis
- Higher movie budgets generally lead to higher box office revenues, as shown by the upward trend, but the wide scatter reveals that spending more doesn't always guarantee blockbuster success.

### Are there any notable trends in genre popularity or movie ratings over time?
Start by extracting the year I want to analyze, its ideal if its not that old and have enough numbers of movies. Upon checking the 2018 to 2020 contains less than 10 movies, this will affect our conclusion if we add this years. I decided to use the data from 1971 to 2017.

```python
movies_df['release_year'] = movies_df['release_date'].dt.year
movies_1971_2017 = movies_df[(movies_df['release_year'] > 1970) & (movies_df['release_year'] < 2018)]
```

### *Genre Popularity Over Time*

To analyze the genre popularity over time I need to explode `genres` column, so each movie can show up multiple times if it belongs to multiple genres. Then let's calculate what percentage of all movies each genre took up every year. Let's focus on the top 5 genres.


Jupyter Notebook: ['5_Genres_Ratings_Over_Time'](./1_Analysis/5_Genres_Ratings_Over_Time.ipynb)
```python
movies_count_yearly = movies_1971_2017['release_year'].value_counts().sort_index()

genres_explode = movies_1971_2017.explode('genres')
top5_genres = genres_explode['genres'].value_counts().sort_values(ascending= False).head(5).index.to_list()

genres_pivot = genres_explode.pivot_table(columns= 'genres', index= 'release_year', aggfunc= 'size').fillna(0)
genres_perc = genres_pivot.div(movies_count_yearly / 100, axis= 0)
```

#### Data Visualized
```python
ticks_year = list(range(1971,2019,2))

plt.figure(figsize= (15,7))
sns.set_theme(style= 'ticks')
sns.lineplot(
    data= genres_perc[top5_genres]
    ,dashes= False
    ,legend= False
)
ax= plt.gca()
ax.set_title('Genre Popularity Over Time (1971 to 2017)', fontsize= 15)
ax.set_xlabel('Release Year')
ax.set_ylabel('Percentage (%)')
ax.set_xticks(ticks_year)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, i: f'{x:.0f}%'))
ax.spines[['top','right']].set_visible(False)
for genre in top5_genres:
    ax.text(
        x= 2017.5
        ,y= genres_perc.loc[[2017],[genre]].values
        ,s= genre
        ,va= 'center'
    )
```
#### Result

!['Genre Popularity Over Time (1971 to 2017)'](./2_Images/line_Genre%20Popularity%20Over%20Time%20(1971%20to%202017).png)
*Line chart showing the genres popularity trend over time*

#### Analysis
- `Drama` has consistently remained the most popular genre from 1971 to 2017, while `Comedy` maintained strong second-place popularity; meanwhile, `Thriller` and `Action` genres showed gradual growth over time, and `Romance` steadily declined in popularity.

### *Movie Ratings Over Time*
Filter out movies that have no `vote_average` then group the data by `release_year` and calculate the median rating each year.

Jupyter Notebook: ['5_Genres_Ratings_Over_Time'](./1_Analysis/5_Genres_Ratings_Over_Time.ipynb)

```python
movie_ratings = movies_1971_2017[movies_1971_2017['vote_average'] > 0]
yearly_ratings = movie_ratings.groupby('release_year')['vote_average'].median()
```

#### Data Visualized
```python
ticks_year = list(range(1971,2019,2))

plt.figure(figsize= (15,7))
sns.lineplot(yearly_ratings)

plt.xticks(ticks_year)
plt.xlabel('Release Year')
plt.ylabel('Vote Average')
plt.title('Movie Rating Over Time from 1971 to 2017', fontsize= 15)
plt.ylim(5,7)
plt.show()
```

#### Result
!['Movie Rating Over Time from 1971 to 2017'](./2_Images/line_Movie%20Rating%20Over%20Time%20from%201971%20to%202017.png)

#### Analysis
- Movie ratings slightly declined from the early 1970s to the 2000s, stabilizing around an average score of 6, but showed a small upward trend again after 2015, hinting at a recent rise in audience satisfaction.

### Which movies are outliers in terms of profitability or audience reception?

Jupyter Notebook: ['6_Outlier_Movies'](./1_Analysis/6_Outlier_Movies.ipynb)

### *Profitability Outliers*
Filter only the movies that have budget and revenue then create a column containing the movie profit.

```python
movies_df = movies_df[(movies_df['budget'] > 0) & (movies_df['revenue'] > 0)].copy()
movies_df['profit'] = movies_df['revenue'] - movies_df['budget']
```
Set the limit to identify if the profit is outlier, I will use `df.quantile(x)` to set the upper and lower limit

```python
Q1 = movies_df['profit'].quantile(0.25)
Q3 = movies_df['profit'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR 
upper_bound = Q3 + 1.5 * IQR
```
Using the limit (`lower_bound` and `upper_bound`), create dataset `hue_gain_profit` and `huge_loss_profit`.

```python
huge_gain_profit = movies_df[movies_df['profit'] > upper_bound].sort_values(by= 'profit', ascending= False).head(5)
huge_loss_profit = movies_df[movies_df['profit'] < lower_bound].sort_values(by= 'profit', ascending= True).head(5)
```

#### Data Visualized
```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

# First column, top and bottom rows
ax1 = fig.add_subplot(gs[0, 0])  # Row 0, Col 0
ax2 = fig.add_subplot(gs[1, 0])  # Row 1, Col 0

# Second column, spans both rows
ax3 = fig.add_subplot(gs[:, 1])  # All rows, Col 1


sns.barplot(
    data= huge_gain_profit
    ,x= 'profit'
    ,y= 'title'
    ,hue= 'profit'
    ,palette= 'dark:g_r'
    ,legend= False
    ,ax= ax1
)
ax1.set_title('Huge Profit Gains Movies', fontsize= 20)
ax1.set_ylabel('')
ax1.set_xlabel('')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,i: f'{x/1_000_000_000}B($)'))
ax1.set_xlim(0,3_000_000_000)
for i,profit in enumerate(huge_gain_profit['profit']):
    ax1.text(
        x= profit + 25_000_000
        ,y= i
        ,s= f'{profit/1_000_000_000:.1f}B($)'
        ,va= 'center'
    )


sns.barplot(
    data= huge_loss_profit
    ,x= 'profit'
    ,y= 'title'
    ,hue= 'profit'
    ,palette= 'dark:r'
    ,legend= False
    ,ax= ax2
)
ax2.set_title('Huge Loss Profit Movies', fontsize= 20)
ax2.set_ylabel('')
ax2.set_xlabel('Profit')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,i: f'{x/1_000_000:.0f}M($)'))
ax2.set_xlim(-200_000_000,0)
for i,profit in enumerate(huge_loss_profit['profit']):
    ax2.text(
        x= profit - 17_000_000
        ,y= i
        ,s= f'{profit/1_000_000:.0f}M($)'
        ,va= 'center'
    )

sns.boxenplot(
    data= movies_df['profit']
    ,ax=ax3
    )
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,i: f'{x/1_000_000_000:.1f}B($)'))
ax3.set_ylabel('')
ax3.set_title('Profit Distribution of All Movies', fontsize= 20)

plt.tight_layout()
plt.show()
```
#### Result
!['Profit Outlier Visualization'](./2_Images/distplot_Profit%20Outlier%20Visualization.png)

#### Analysis
- The movie industry's profitability is extremely skewed, and tiny fraction of blockbuster films generate outsized profits, while the vast majority of movies cluster around break-even or incur losses, highlighting a high-risk, high-reward economic model.

- A few iconic movies like *Avatar*, *Star Wars: The Force Awakens*, and *Titanic* dominate industry profits by earning billions, while major titles like *The Lone Ranger* and *Mars Needs Moms* suffered huge losses, proving that even big productions aren't immune to financial failure.

### *Audience Reception Outliers*
Filter the movies that have at least 500 `vote_counts`.
```pthon
rated_movies = movies_df[movies_df['vote_count'] >= 500]
```
Find the upper and lower limit using IQR method
```python
low_ratings = Q1 - 1.5 * IQR
high_ratings = Q3 + 1.5 * IQR
```
Using the `low_ratings` and `high_ratings` filter the dataset for outliers

```python
lowest_ratings = rated_movies[rated_movies['vote_average'] < low_ratings].sort_values(by= 'vote_average', ascending= True).head(5)
highest_ratings = rated_movies[rated_movies['vote_average'] > high_ratings].sort_values(by= 'vote_average', ascending= False).head(5)
```

#### Data Visualized
```python
fig = plt.figure(figsize= (20, 8))
gs = gridspec.GridSpec(2,2,width_ratios=[1,1])

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[:,1])

sns.barplot(
    data= highest_ratings
    ,x= 'vote_average'
    ,y= 'title'
    ,hue= 'vote_average'
    ,palette= 'dark:g_r'
    ,legend= False
    ,ax= ax1
    )
ax1.set_title('Highest Rated Movies', fontsize= 15)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_xlim(0,11)

for i,value in enumerate(highest_ratings['vote_average']):
    ax1.text(value+.1,i,value)

sns.barplot(
    data= lowest_ratings
    ,x= 'vote_average'
    ,y= 'title'
    ,hue= 'vote_average'
    ,palette= 'dark:r'
    ,legend= False
    ,ax= ax2
    )
ax2.set_title('Lowest Rated Movies', fontsize= 15)
ax2.set_xlabel('Rating')
ax2.set_ylabel('')
ax2.set_xlim(0,11)

for i,value in enumerate(lowest_ratings['vote_average']):
    ax2.text(value+.1,i,value)

sns.boxplot(rated_movies['vote_average'], ax=ax3)
ax3.set_ylabel('')
ax3.set_title('Ratings Distribution of All Movies', fontsize= 15)

plt.tight_layout()
plt.show()
```
#### Result
!['Ratings Outlier Visualization'](./2_Images/distplot_Ratings%20Outlier%20Visualization.png)

#### Analysis
- The ratings data shows that *Dilwale Dulhania Le Jayenge* is a clear outlier with an exceptionally high score of 9.1, while the lowest-rated movies like *Jack and Jill*, *Catwoman*, and *Batman & Robin* fall between 4.0 and 4.4, reflecting very poor audience reception.

- Most movies tend to cluster around 6 to 7 rating, suggesting that average quality films are far more common than exceptional hits or major failures. Although extreme outliers exist on both ends, they are relatively rare compared to the large concentration of moderately rated movies.


### What are the most common characteristics of successful movies?
A successful movie can be defined based on high audience ratings (vote average ≥ 8.0 with at least 1,000 votes), high profitability (top 10% in profit), or a combination of both strong ratings and profits. To better understand the traits of these successful films, I will analyze key characteristics including their genres, runtime, language, budget, release year and production companies. This analysis will help reveal common patterns among the movies that perform best both critically and financially.

Jupyter Notebook: [7_Succesful_Movies_Characterestics](./1_Analysis/7_Succesful_Movies_Characterestics.ipynb)

```python
movies = movies_df[(movies_df['budget'] > 0) & (movies_df['revenue'] > 0)].copy()
movies['profit'] = movies['revenue'] - movies['budget']
profit_threshold = movies['profit'].quantile(0.90)

successful_movies = movies[(movies['profit'] > profit_threshold) & (movies['vote_average'] >= 8.0) & (movies['vote_count'] >= 1000)]
```

### *Genres*
examine the genre distribution of successful movies by first expanding the `genres` column separating movies with multiple genres into individual rows, giving each genre its own entry.

```python
genres_explode = successful_movies.explode('genres')
genres_count = genres_explode['genres'].value_counts(normalize= True).reset_index(name= 'perc')
```

#### Data Visualized
```python
sns.set_theme(style= 'ticks')
sns.barplot(
    data= genres_count
    ,x= 'perc'
    ,y= 'genres'
    ,hue= 'perc'
    ,palette= 'dark:b_r'
    ,legend= False
    )
ax= plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,i: f'{x*100:.0f}%'))
for i,value in enumerate(genres_count['perc']):
    ax.text(
        x= value+0.001
        ,y= i
        ,s= f'{value*100:.0f}%'
        ,va= 'center'
    )
ax.spines[['top','right']].set_visible(False)
ax.set_title('Genre Distribution for Successful Movies', fontsize= 15)
ax.set_xlabel('Distribution (%)')
ax.set_ylabel('')
plt.show()
```

#### Result
![Genre Distribution for Successful Movies](./2_Images/bar_Genre%20Distribution%20for%20Successful%20Movies.png)

#### Analysis
- *Drama* and *Adventure* dominate successful movie genres, together making up 31% of the total distribution, suggesting audiences are heavily drawn to emotionally engaging and exciting narratives.

- *Action*, *Thriller*, *Fantasy*, and *Crime* are strong mid-tier genres, each capturing 9 - 10% of the share, highlighting the popularity of suspenseful and imaginative storytelling.

- Genres like *Romance*, *War*, and *Mystery* are the least represented, each with 3% or less, indicating that these genres are less frequently associated with successful movies.

### *Runtime*
You are analyzing the runtime distribution by first filtering out movies with a runtime of zero, then plotting two Kernel Density Estimation (KDE) curves one for all movies and one for successful movies. This allows you to visually compare how runtimes are distributed across both groups and see whether successful movies tend to be longer, shorter, or similar in length compared to the overall movie set.

```python
movies_runtime = movies_df[movies_df['runtime'] > 0 ]
```
#### Data visualized
```python
sns.kdeplot(movies_runtime['runtime'], label='All Movies', fill=True)
sns.kdeplot(successful_movies['runtime'], label='Successful Movies', fill=True)
plt.title('Runtime Distribution')
plt.xlabel('Runtime (mins.)')
plt.xlim(0,600)
plt.legend()
plt.show()
```

#### Result
![Runtime Distribution](./2_Images/kde_Runtime%20Distribution.png)

#### Analysis
- Successful movies tend to have longer runtimes compared to the overall movie population, with a peak around 120–150 minutes, whereas most movies cluster closer to 90–100 minutes.
- Successful movies have a wider spread in runtime, suggesting that slightly longer, more developed storytelling may contribute to better outcomes.

### *Language*
To analyze the language distribution among successful movies first calculating the proportion of each original language and converting it to percentages. You then map language codes (like 'en' to English, 'fr' to French) for easier reading.

```python
successfull_languages = successful_movies['original_language'].value_counts(normalize= True).reset_index(name= 'proportion')
successfull_languages['proportion'] = successfull_languages['proportion'] * 100

language = {'en':'English', 'ja':'Japanese', 'it':'Italiano', 'fr':'French'}
successfull_languages['original_language'] = successfull_languages['original_language'].map(language)
```

#### Data Visualized

```python
sns.set_theme(style= 'ticks')                     
sns.barplot(
    data= successfull_languages
    ,x= 'original_language'
    ,y='proportion'
    ,hue= 'proportion'
    ,palette= 'dark:b_r'
    ,legend= False
)
ax= plt.gca()

ax.set_xlabel('')
ax.set_ylabel('Proportion(%)')
ax.set_title('Language Distribution in Successful Movies')
ax.set_ylim(0,100)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,i : f'{x:.0f}%'))
for i,value in enumerate(successfull_languages['proportion']):
    ax.text(
        x= i
        ,y= value + 3
        ,s= f'{value:.0f}%'
        ,ha= 'center'
    )
plt.show()
```
#### Result
![Language Distribution in Successful Movies](./2_Images/bar_Language%20Distribution%20in%20Successful%20Movies.png)

#### Analysis
- English dominates successful movies with 83% of the share, highlighting the global reach and dominance of English-language films.

### *Budget*
Categorizing successful movies into budget groups based on their budget values.

```python
def budget_bucket(x):
    if x < 1e6:
        return 'Low <M'
    elif x < 1e7:
        return 'Mid (1M-10M)'
    elif x < 5e7:
        return 'High (10M-50M)'
    else:
        return 'Blockbuster (50M+)'

bucket = pd.DataFrame({'budget_group':['Low <M','Mid (1M-10M)','High (10M-50M)','Blockbuster (50M+)']})   
successful_movies['budget_group'] = successful_movies['budget'].apply(budget_bucket)
successful_budget_group = successful_movies['budget_group'].value_counts(normalize= True).reset_index(name='proportion')
budget_group_count = bucket.merge(right=successful_budget_group, how= 'left', on= 'budget_group')
```

#### Data Visualized
```python
sns.set_theme(style= 'ticks')
sns.barplot(
    data= budget_group_count
    ,x= 'proportion'
    ,y= 'budget_group'
)
ax = plt.gca()
ax.invert_yaxis()
ax.set_xlabel('Proportion (%)')
ax.set_ylabel('')
ax.set_title('Budget Group Distribution of Successful Movies')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,i: f'{x*100:.0f}%'))
ax.set_xlim(0,0.70)
for i,value in enumerate(budget_group_count['proportion']):
    ax.text(
        x= value + 0.05
        ,y= i
        ,s= f'{value*100:.0f}%'
    )
plt.show()
```
#### Result
![Budget Group Distribution of Successful Movies](./2_Images/bar_Budget%20Group%20Distribution%20of%20Successful%20Movies.png)

#### Analysis
- Successful movies are predominantly those with high budgets ranging from $10 million to $50 million, making up 57% of the total. Blockbusters with budgets exceeding $50 million follow, accounting for 35%. 

- Mid-budget films ($1 million to $10 million) contribute only 9% to the success pool, while low-budget films (below $1 million) are notably absent. 

- This suggests a strong correlation between higher production budgets and movie success, with the $10M–$50M range appearing to offer the most favorable balance between investment and outcome.

### *Release Date*
To identify what month produce a successful movies, create a column that contains the month number and month name. month number will be used to arrange the month name in graph.

```python
movies_df['release_month_no'] = movies_df['release_date'].dt.month
movies_df['release_month'] = movies_df['release_date'].dt.strftime('%b')
```

#### Data Visualized

```python
month_bucket = movies_df[['release_month','release_month_no']].drop_duplicates().dropna().sort_values('release_month_no')

successful_month_count = successful_movies['release_month_no'].value_counts().reset_index()
succesful_movies_month = month_bucket.merge(successful_month_count, how= 'left', left_on= 'release_month_no', right_on= 'release_month_no').drop('release_month_no',axis= 1)
sns.barplot(
    data= succesful_movies_month
    ,x= 'release_month'
    ,y= 'count'
)
ax = plt.gca()
ax.set_xlim(-1,12)
ax.set_ylim(0,6)
ax.set_ylabel('No. of Movies')
ax.set_xlabel('Release Month')
ax.set_title('Total Numbers of Successful Movies Release by Month')
plt.tight_layout()
plt.show()
```

#### Result
![Total Numbers of Successful Movies Release by Month](./2_Images/bar_Total%20Numbers%20of%20Successful%20Movies%20Release%20by%20Month.png)

#### Analysis
- Successful movie releases peak in July, November, and December, suggesting that summer and holiday seasons are prime times for blockbuster launches, while early spring months like March and April see very few hits.

### *Company*
Explode the column `production_companies` then use `value_counts()` to identify which company produced morethan one successful movies.

#### Data Visualized
```python
company_successful_movies = successful_movies['production_companies'].explode().value_counts().reset_index(name= 'successful_movies')
top_company = company_successful_movies[company_successful_movies['successful_movies'] > 1]
sns.barplot(
    data= top_company
    ,x= 'successful_movies'
    ,y= 'production_companies'
)
plt.xlim(0,5)
plt.title('Numbers of Successful Movies Produced by Companies')
plt.xlabel('Count of Successful Movies')
plt.ylabel('')
```

#### Result
![Numbers of Successful Movies Produced by Companies](./2_Images/bar_Numbers%20of%20Successful%20Movies%20Produced%20by%20Companies.png)

#### Analysis
- New Line Cinema and Warner Bros. lead in producing successful movies, each with four hits, while several other major studios like Paramount, WingNut Films, and Syncopy closely follow with three, highlighting a competitive but slightly top-heavy industry landscape.

