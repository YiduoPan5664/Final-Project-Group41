# Final-Project-Group41

**Growth-Driven or Redistributive? An Analysis of Chicago's TIF Spending**

Yiduo Pan | GitHub: YiduoPan5664

## Streamlit Dashboard

[https://final-project-group41.streamlit.app/](https://final-project-group41.streamlit.app/)

> **Note:** Streamlit apps need to be "woken up" if they have not been accessed in the last 24 hours. If the app shows a sleep screen, click "Yes, get this app back up!" and wait ~30 seconds.

## Data Sources

All datasets are from the [Chicago Data Portal](https://data.cityofchicago.org/) and placed in `data/raw-data/`.

| File | Description |
|------|-------------|
| `Tax_Increment_Financing_(TIF)_Annual_Report_-_Analysis_of_Special_Tax_Allocation_Fund_20260301.csv` | Annual TIF expenditure and increment data by district, 2017–2024 |
| `Boundaries_-_Tax_Increment_Financing_Districts_20260301.csv` | Current TIF district boundaries (geometry) |
| `Boundaries_-_Tax_Increment_Financing_Districts_(Deprecated_March_2018)_20260301.csv` | Deprecated TIF boundaries for districts dissolved before March 2018 |
| `Boundaries_-_Community_Areas_20260301.csv` | Chicago's 77 community area boundaries |
| `ACS_5_Year_Data_by_Community_Area_20260301.csv` | ACS 5-year household income estimates by community area (2023) |

## How the Data is Processed

All data processing is handled in `final_project.qmd`. The key steps are:

1. **TIF annual data** — cleaned and joined to district boundaries via a normalised TIF reference number (`TIF Number` / `REF` column), since the two files use different naming conventions.
2. **Boundary files** — the current and deprecated boundary files are combined to ensure full geographic coverage (154 unique districts).
3. **Income data** — weighted mean household income is computed from ACS bracket counts using bracket midpoints.
4. **Spatial join** — TIF district centroids are matched to community areas to assign an income value to each district.

## Repository Structure
```
├── data/
│   └── raw-data/
│       ├── Tax_Increment_Financing_(TIF)_Annual_Report_-_Analysis_of_Special_Tax_Allocation_Fund_20260301.csv
│       ├── Boundaries_-_Tax_Increment_Financing_Districts_20260301.csv
│       ├── Boundaries_-_Tax_Increment_Financing_Districts_(Deprecated_March_2018)_20260301.csv
│       ├── Boundaries_-_Community_Areas_20260301.csv
│       └── ACS_5_Year_Data_by_Community_Area_20260301.csv
├── streamlit-app/
│   └── app.py             # Streamlit dashboard code
├── final_project.qmd      # Full writeup and analysis code
├── final_project.html     # Knitted HTML version
├── final_project.pdf      # Knitted PDF version
├── requirements.txt       # Python dependencies
└── .gitignore
```