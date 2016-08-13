##Capstone Proposal##

###Facebook V: Predicting Check Ins###

#####The Problem#####
Predicting check-ins in an artificial 10km x 10km region. The goal of this competition is to predict which place a person would like to check in to. For the purposes of this competition, Facebook created an artificial world consisting of more than 100,000 places located in a 10 km by 10 km square. For a given set of coordinates, your task is to return a ranked list of the most likely places. Data was fabricated to resemble location signals coming from mobile devices, giving you a flavor of what it takes to work with real data complicated by inaccurate and noisy values. Inconsistent and erroneous location data can disrupt experience for services like Facebook Check In.

#####The Data#####
- row_id: id of the check-in event
- x y: coordinates
- accuracy: location accuracy 
- time: timestamp
- place_id: id of the business, this is the target you are predicting

*Summary*
- Distinct Locations:  108390  
- Total Data Points:  29118021

#####Approach#####
This is a multi-class Classification problem with 108390 distinct known classes. This makes the problem hard to be modelled by a single classifier. The approach we will take is to train multiple classifier for smaller windows of regions inside the world. 
We will begin by using a simple Logistic Regression classifier and follow it up with Naive Bayes, Decision Trees, Neural Nets and Boosting, evaluate performance and decide on our final model.

https://www.kaggle.com/c/facebook-v-predicting-check-ins/data
