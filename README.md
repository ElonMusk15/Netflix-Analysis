# Netflix Data Analysis

This project is an exploratory data analysis (EDA) of Netflix's dataset to understand the content available on the platform. The dataset includes details about movies and TV shows such as title, director, cast, country, date added, release year, rating, duration, and genre.

## Features

- Data cleaning and preprocessing
- Analysis of Netflix content over the years
- Insights on most common genres, countries, and ratings
- Visualizations for better understanding of patterns and trends
- Handling of missing data and date formatting issues

## Dataset

The dataset used in this project is the **Netflix Movies and TV Shows** dataset, which includes the following columns:

- `show_id`
- `type`
- `title`
- `director`
- `cast`
- `country`
- `date_added`
- `release_year`
- `rating`
- `duration`
- `listed_in`
- `description`

## Installation

1. Clone the repository:
```bash
https://github.com/yourusername/netflix-data-analysis.git
cd netflix-data-analysis
```

2. Install dependencies:
```bash
pip install pandas matplotlib seaborn jupyterlab
```

## Usage

1. Run the Jupyter Notebook or Python script to perform the analysis:
```bash
jupyter lab
```
2. Open `netflix_analysis.ipynb` or run `week1.py` to see the outputs and visualizations.

## Key Learnings

- Cleaning messy data (e.g., handling spaces in dates)
- Converting string columns to datetime format
- Visualizing data using Matplotlib and Seaborn
- Extracting insights from categorical data

## Future Improvements

- Build an interactive dashboard using Streamlit or Plotly Dash
- Predict trends in content releases using time series forecasting
- Perform sentiment analysis on descriptions or reviews

## License

This project is open-source and available under the [MIT License](LICENSE).

---





