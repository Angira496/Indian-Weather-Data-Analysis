# Indian Weather Data Analysis
A Streamlit web application for exploring and visualizing 10 years (2009–2019) of daily weather data across eight major Indian cities. The app offers interactive pages for dataset preview, exploratory analysis, and key outcomes, all wrapped in a sleek dark theme. Built with Python and its libraries (NumPy, Pandas, Matplotlib, Seaborn, Plotly).


## Preview
[▶️ Webapp Link ](https://indian-weather-data-analysis-reekparna.streamlit.app/)


## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Usage](#usage)
- [Technologies & Libraries](#technologies--libraries)
- [Authors](#authors)

## Project Overview
This project analyzes historical weather data for eight major Indian cities: Pune, Mumbai (Bombay), Delhi, Hyderabad, Jaipur, Kanpur, Nagpur, and Bengaluru. It provides insights into seasonal trends, regional climatic variations, and key weather parameters such as temperature, humidity, wind, precipitation, UV index, cloud cover, and air quality.

## Features
- **Home Page**: Project overview, importance, context, methodology, and impact.
- **Dataset Page**: Preview dataset details, download full CSV, view EDA results and pseudocode for data preparation.
- **Analysis Page**: Eight interactive sub-pages for temperature trends, humidity patterns, wind/gust analysis, precipitation overview, UV index variations, cloud cover & solar potential, air quality insights, and correlation heatmap.
- **Outcomes Page**: Key findings, limitations, future scope, and conclusion.
- **Dark Theme**: Sleek and modern dark UI using Streamlit’s theming capabilities.

## Dataset
- **Source**: “Historical Weather Data for Indian Cities” on Kaggle.
- **Records**: 771,264 daily observations.
- **Columns**: 28 parameters including date, time, city, temperature, humidity, windspeed, precipitation, UV index, and more.
- **Time Span**: January 2009 to December 2019.

## Usage
1. Place `modified_data.csv` in the project root.
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Use the sidebar to navigate between pages and explore interactive visualizations.

## Technologies & Libraries
- **Python** 3.8+
- **Streamlit** for web UI
- **NumPy** for numerical operations
- **Pandas** for data manipulation
- **Matplotlib** & **Seaborn** for static plotting
- **Plotly** for interactive charts
- **Statsmodels** for any statistical analysis

## Authors
- Angira Bannerjee
- Reekparna Sen
- Pratyush Chakraborty
