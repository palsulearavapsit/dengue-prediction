# Dengue Prediction Project — Dataset Preparation

## Step 1: Identify the Historical Dengue Dataset

We started with the historical dataset named `Final_data(3)(1).csv`. It contained 8,985 rows and 15 columns, including disease name, reported cases, deaths, date, district, state, latitude, longitude, precipitation, LAI, and temperature.

**Dataset file:** `Final_data(3)(1).csv`

**Important:** The exact original download webpage for this CSV was not recorded in our project files, so we cannot confidently identify its original download link.

## Step 2: Identify the Required Data Sources

We identified the following sources for the project:

1. **Historical epidemiological dataset:** Used for historical disease observations, reported cases, deaths, dates, and locations.
2. **Open-Meteo Historical Weather API:** https://open-meteo.com/en/docs/historical-weather-api
   Used to obtain historical weather information, such as temperature, humidity, and precipitation.
3. **National Centre for Vector Borne Diseases Control (NCVBDC):** https://ncvbdc.mohfw.gov.in/
   Considered for dengue surveillance information, including reported cases and deaths.
4. **OpenStreetMap Nominatim:** https://nominatim.openstreetmap.org/
   Used by the application to resolve geographic locations.

**Note:** These sources have different roles. The historical CSV is the documented input to our cleaning script. Open-Meteo, NCVBDC, and Nominatim are also used or referenced by the application pipeline; we should not claim that all their data was merged into the original CSV.

## Step 3: Inspect the Original Dataset

We inspected the CSV to understand its structure, column names, data types, missing values, and disease categories.

The original dataset contained multiple diseases rather than dengue alone.

**Original dataset size:** 8,985 rows × 15 columns.

## Step 4: Filter Dengue Records

We filtered the dataset using the `Disease` column and retained only records where the disease was Dengue.

This removed records belonging to other diseases and produced a dengue-only subset of **1,619 rows**.

## Step 5: Standardize and Convert the Data

We cleaned the column names by removing extra spaces and converted relevant fields into numeric values where necessary.

These fields included reported cases, deaths, day, month, year, latitude, longitude, precipitation, LAI, and temperature.

Invalid numeric values were converted to missing values so they could be handled during cleaning.

## Step 6: Create a Proper Date Column

The source stored the date in separate `day`, `mon`, and `year` columns.

We combined these fields to create a proper date column. Records with invalid dates were treated as invalid.

This made it possible to sort observations chronologically and build time-based features later.

## Step 7: Remove Invalid Records

We removed records that were missing required information, including state, district, reported cases, date components, coordinates, precipitation, or LAI.

**Invalid records removed: 422.**

This step improved data consistency for the prediction pipeline.

## Step 8: Handle Missing Values

We standardized state and district names by trimming extra spaces.

For missing death counts, the preparation script preserved a `Deaths_missing` indicator and used the median to fill missing values for machine-learning preparation.

Temperature was checked for missing values. The script was designed to remove the temperature column if every value was missing in the cleaned dataset rather than inventing temperature measurements.

## Step 9: Remove Duplicate Records

We sorted the records by district and date and removed duplicate observations based on state, district, disease, and date.

**Duplicate records removed: 149.**

This helped avoid counting the same district-level observation multiple times.

## Step 10: Save the Cleaned Dataset

After filtering and cleaning, the dataset contained:

- **1,048 rows**
- **307 districts**
- **32 states/UTs**
- Date range: **31 August 2009 to 8 June 2022**

The cleaning script saves its cleaned output as `prepared_data/dengue_clean.csv`.

## Step 11: Prepare Weather and Geographic Context

The application uses geographic coordinates to identify the requested location and can retrieve historical weather information through Open-Meteo.

Relevant environmental information includes temperature, humidity, and precipitation. Geographic information includes latitude, longitude, district, and state.

These features provide context for dengue prediction; their inclusion does not prove that any individual environmental factor caused a change in dengue cases.

## Step 12: Build the Forecasting Features

We prepared time-based and historical case features for machine learning, including:

- Current and previous reported cases.
- Lagged case counts.
- Rolling means, medians, minimums, maximums, and standard deviations.
- Changes in previous case counts.
- Time between reports.
- Year, month, week, and seasonal features.
- Geographic coordinates.
- Environmental features such as temperature, precipitation, and LAI where available.

These features help the model learn patterns in historical observations.

## Step 13: Create the Model-Ready Dataset

The forecasting dataset was documented as approximately **1,002 rows and 39 columns** before the final model-input preparation.

The model pipeline then selects and prepares the required features for the Random Forest regression model.

## Step 14: Create and Use `Final_data(3).csv`

The project folder contains `Final_data(3).csv`, a 15-column CSV with the same column structure as `Final_data(3)(1).csv`.

The exact file-generation history for `Final_data(3).csv` was not recorded in the preparation script we inspected. Therefore, we should not claim that it was directly generated by the cleaning steps above without checking the file's creation history.

In the n8n workflow, the historical data file is read, filtered to the relevant district and date, and used to build lag and temporal features for prediction.

**Final purpose:** Prepare historical dengue observations and related features so the Random Forest model can estimate dengue cases at the next observed district report.

---

## Summary

The dataset preparation process began with 8,985 historical records. We filtered the records to Dengue, removed 422 invalid rows and 149 duplicate observations, and obtained a cleaned dataset of 1,048 rows covering 307 districts across 32 states/UTs. We then prepared historical, temporal, geographic, and environmental features for the forecasting pipeline.

**Important limitation:** The documented cleaning counts describe the cleaned dengue dataset. The exact source-download URL and the precise steps used to create `Final_data(3).csv` still need to be verified from the original data source and file-generation history.
