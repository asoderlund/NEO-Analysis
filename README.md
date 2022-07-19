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

<details><summary>CLICK ME</summary>
<p>

#### We can hide anything, even code!

```python
   len(df)
```

</p>
</details>


## Univariate and Bivariate Analysis


## Model Building

### More Pre-Processing


### Basic Decision Tree Classification


### K Nearest Neighbors Classification


### Random Forest Decision Tree Classification


### Gradient Boosted Decision Tree Classification

# Final Results and Final Remarks
