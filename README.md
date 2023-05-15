# Ventilator Pressure Prediction for Covid-19 Patients

## To simulate a ventilator connected to a sedated patient's lung

## Overview:

A ventilator is a machine that provides mechanical ventilation by moving breathable air into and out of the lungs, to deliver breaths to a patient who is physically unable to breathe, or breathing insufficiently. A ventilator is needed when a patient suffers respiratory failure. Mechanical ventilators are mainly used in hospitals and in transport systems such as ambulances and MEDEVAC air transport etc. Ventilator has been a key component for Covid-19 treatment. I took part in a competetion during Covid-19 that aims to simulate a ventilator connected to a sedated patient's lung. Currently ventilators are simulated using PID controllers and it is belived that a better performance can be obtained by Machine Learning. This approach has the potential to overcome the cost barrier of developing new methods for controlling mechanical ventilators. This will pave the way for algorithms that adapt to patients and reduce the burden on clinicians during these novel times and beyond. As a result, ventilator treatments may become more widely available to help patients breathe.

## Motivation:

The cost of devising novel techniques to regulate mechanical ventilators is excessively high, even prior to starting clinical experimentation. The implementation of advanced simulators has the potential to mitigate this obstacle.


## Objectives:

- The objective of this project is to develop AI system to simulate a ventilator connected to a sedated patient's lung.

## My Approch:

### Analysis & Baseline

In order to develop a successful AI model, it is important to understand the data at hand. For this reason, an exploratory data analysis was performed to gain insights from the data. Additionally, a baseline model was developed using the LGBM algorithm to establish a starting point for comparison with future models.

### Feature Engineering

Feature engineering is a critical step in developing machine learning models, as it involves creating new features from the existing data to improve model performance. In this project, extensive feature engineering was performed to extract meaningful information from the raw data. This step ensured that the machine learning models had access to the most relevant information for making accurate predictions.

### Tranning & Ensamble

To train the machine learning models, a bi-directional LSTM model with K-fold cross-validation was used. This model was chosen due to its effectiveness in handling sequential data. By using Bidirectional LSTM, the model can learn from the past values in both directions, which helps in capturing more complex patterns in the data and improving the accuracy of the model. To further improve the performance of the models, ensemble learning was employed. Ensemble learning involves combining the predictions of multiple models to improve the overall accuracy and reduce overfitting.

## Conclusion

The development of an AI system to simulate a ventilator connected to a sedated patient's lung is a crucial step in improving the accessibility and effectiveness of mechanical ventilator treatments. The implementation of advanced simulators and machine learning techniques can potentially overcome the cost barriers of developing new methods for controlling mechanical ventilators. This project utilized extensive feature engineering and Bi-directional LSTM with K-fold to train and ensemble the model. The approach taken in this project has the potential to reduce the burden on clinicians during these novel times and beyond, thus making ventilator treatments more widely available to help patients breathe.
