# NASA Nearest Earth Objects Analysis Project by Alyssa Soderlund

This is a simple exploration, analysis, and model building project I created in Python 3. 

See a project I created in R here: ([Wine Analysis Project](https://asoderlund.github.io/WineAnalysis/)).

## The Dataset
The dataset comes from the NASA Open API and NEO Earth Close Approaches. It can be found on Kaggle here: ([Data](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)). I am using version 2 of this dataset.

This dataset contains information about asteroids orbiting earth. It is important to understand objects close to earth, as they can impact the earth in many ways and distrupt the earths natural phenomena. Information about the size, velocity, distance from earths orbit, and the magnitude of the luminosity of the asteroid can help experts identify whether an asteroid poses a threat or not. This project will analyze information about these asteroids, and attempt to create a model to predict whether or not an asteroid is potentially hazardous.

The attributes of this dataset are: 
- *id*: identifier (the same object can have several rows in the dataset, as it has been observed multiple times)
- *name*: name given by NASA (including the year the asteroid was discovered)
- *est_diameter_min*: minimum estimated diameter in kilometers
- *est_diameter_max*: maximum estimated diameter in kilometers
- *relative_velocity*: velocity relative to earth
- *miss_distance*: distance in kilometers it misses Earth
- *orbiting_body*: planet that the asteroid orbits
- *sentry_object*: whether it is included in sentry - an automated collision monitoring system
- *absolute_magnitude*: intrinsic luminosity
- *hazardous*: whether the asteriod is potentially harmful or not

## Exploratory Data Analysis and Pre-processing
This dataset has 10 columns and 90,836 rows. It has no missing values. Peeking at the first 10 rows of data reveals what the data looks like:

![](./images/table1.png)
_Table 1_


A cursory examination of the dataset shows that *orbiting_body* and *sentry_object* each only have 1 unique value, so they are dropped from the table.

We also see that id and name each only have 27423 unique values. This means that the same asteroid is measured multiple times. Let's take a look at one of these asteroids to see what changes with each record:

![](./images/table2.png)
_Table 2_

Looking at Table 2, it appears that *relative_velocity* and *miss_distance* change with each observation of the same asteroid. A large majority of the time, the classification of *hazardous* does not change with each observation.

We would assume intuitively that most of these asteroids are not hazardous, because if most asteroids were hazardous we would probably have a lot more collisions with them! The imbalance is not too extreme though- about 9.7% of objects are classified as hazardous. This can be handled with a stratified train-test split later on.

![](./images/fig1.png)
_Figure 1_

Next, a correlation heatmap in Figure 2 shows that *est_diameter_min* and *est_diameter_max* are perfectly correlated. This means we only need to keep one of these variables, so we will drop *est_diameter_min*.

![](./images/fig2.png)
_Figure 2_

Next, I was curious about the year that is included in the names of each asteroid. I decided to extract the year from the *name* variable to see if there is any pattern with the year the asteroid was discovered.

<details><summary markdown="span">Click HERE to see my code for extracting the year from the name</summary>
```python
df[['drop','work']]=df.name.str.split('(',expand=True)
df.drop(columns='drop',inplace=True)

def year_extract(x):
    return x.strip()[0:x.strip().index(' ')]
df['year']=df['work'].apply(year_extract)

df.drop(columns='work', inplace=True)

df.loc[df.year=='A911','year']='1911' 
df.loc[df.year=='6743','year']='1960'
df.loc[df.year=='A898','year']='1898'
df.loc[df.year=='6344','year']='1960'
df.loc[df.year=='A924','year']='1924'
df.loc[df.year=='A/2019','year']='2019'
df.loc[df.year=='4788','year']='1960'
  
df.year=df.year.astype(int)
```
</details>
<br/>


## Univariate and Bivariate Analysis


## Model Building

### More Pre-Processing


### Basic Decision Tree Classification


### K Nearest Neighbors Classification


### Random Forest Decision Tree Classification


### Gradient Boosted Decision Tree Classification

# Final Results and Final Remarks
