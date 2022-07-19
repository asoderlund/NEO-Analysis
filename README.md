# NASA Nearest Earth Objects Analysis Project by Alyssa Soderlund

This is a simple exploration, analysis, and model building project I created in Python 3. 

See a project I created in R here: ([Wine Analysis Project](https://asoderlund.github.io/WineAnalysis/)).

## The Dataset
The dataset comes from the NASA Open API and NEO Earth Close Approaches. It can be found on Kaggle here: ([Data](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)). I am using version 2 of this dataset.

This dataset contains information about asteroids orbiting earth. It is important to understand objects close to earth, as they can impact the earth in many ways and distrupt the earths natural phenomena. Information about the size, velocity, distance from earths orbit, and the magnitude of the luminosity of the asteroid can help experts identify whether an asteroid poses a threat or not. This project will analyze information about these asteroids, and attempt to create a model to predict whether or not an asteroid is potentially hazardous.

The attributes of this dataset are: 
- id: identifier (the same object can have several rows in the dataset, as it has been observed multiple times)
- name: name given by NASA (including the year the asteroid was discovered)
- est_diameter_min: minimum estimated diameter in kilometers
- est_diameter_max: maximum estimated diameter in kilometers
- relative_velocity: velocity relative to earth
- miss_distance: distance in kilometers it misses Earth
- orbiting_body: planet that the asteroid orbits
- sentry_object: whether it is included in sentry - an automated collision monitoring system
- absolute_magnitude: intrinsic luminosity
- hazardous: whether the asteriod is potentially harmful or not

## Exploratory Data Analysis and Pre-processing
This dataset has 10 columns and 90836 rows. Peaking at the first 10 rows of data reveals what the data looks like:

![](./images/table1.png)
_Table 1_


A cursory examination of the dataset shows that orbiting_body and sentry_object each only have 1 unique value, so they are dropped from the table.

We also see that id and name each only have 27423 unique values. This means that the same asteroid is measured multiple times. Lets take a look at one of these asteroids to see what changes with each record:

![](./images/table2.png)
_Table 2_


<details><summary markdown="span">Let's see some code!</summary>
```python
print('Hello World!')
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
