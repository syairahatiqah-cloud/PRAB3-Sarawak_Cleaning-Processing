# FSM WL/SF Data Cleaning and Imputation App

This Streamlit application cleans, visualises, summarises and imputes water-level or streamflow time-series data.

## Files to upload to GitHub

- `app.py` — main Streamlit application
- `requirements.txt` — Python packages installed by Streamlit Community Cloud
- `.gitignore` — prevents temporary Python files from being uploaded

`README.md` is optional but recommended because it documents the repository.

## Supported input files

- Excel: `.xlsx` and `.xls`
- CSV: `.csv`

The uploaded data must contain at least:

1. A date/time column
2. A water-level or streamflow value column

## Missing values recognised

The app automatically recognises blank cells and text markers such as `NA`, `N/A`, `NaN`, `NULL`, `None`, `Missing`, `No Data`, `-`, and `--`.

The default numeric missing-value markers are:

```text
-99999, -9999, -999.99, -999, -99.99, 9999, 99999
```

These markers can be changed inside the app before processing.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files from this folder to the repository root.
3. Commit the files.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/).
5. Select **Create app**.
6. Choose your GitHub repository and branch.
7. Set the main file path to `app.py`.
8. Select **Deploy**.

If the app was already deployed, commit these updated files and then reboot the app from the Streamlit dashboard.

## Recommended date setting

For Malaysian data in a format such as `01/09/2020 00:00`, select:

```text
DMY (DD/MM/YYYY) — recommended for Malaysia
```

This prevents the date from being incorrectly interpreted as `MM/DD/YYYY`.
