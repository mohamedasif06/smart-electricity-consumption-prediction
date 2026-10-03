# Smart Electricity Consumption Prediction

A machine learning project that predicts **next-hour household electricity consumption** using an XGBoost regression model and provides predictions through a Flask web application.

## Project Overview

Electricity consumption changes based on current electrical usage, previous consumption, and time-related patterns. This project uses historical household electricity data to build a machine learning model capable of estimating the next hour's **Global Active Power**.

The trained XGBoost model is integrated into a Flask web application where users can enter or upload electricity data and receive a next-hour consumption prediction.

## Problem Statement

Household electricity consumption varies throughout the day depending on appliance usage and time-based patterns. The objective of this project is to use historical electricity measurements and time-related features to predict the household's electricity consumption for the following hour.

## Objectives

- Analyze household electricity consumption data.
- Perform data preprocessing and feature engineering.
- Create time-based and lag features.
- Train machine learning regression models.
- Compare different regression models.
- Select an appropriate model for prediction.
- Deploy the trained model using Flask.
- Provide a simple web interface for next-hour prediction.
- Provide energy-awareness remarks based on predicted consumption.

## Dataset

The project uses the **Individual Household Electric Power Consumption** dataset.

The original raw dataset is not included in this repository because of its large file size.

The processed feature dataset used by the project is included in:

```text
data/processed/electricity_features.csv
