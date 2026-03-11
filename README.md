# Predicting Car Prices

This project performs exploratory data analysis (EDA) on a used-car pricing dataset in `price_prediction.ipynb`.
The notebook focuses on understanding data quality, distribution patterns, and relationships that matter for a future price prediction model.

## Project Files

- `price_prediction.ipynb`: Main analysis notebook
- `Price_prediction.csv`: Dataset used in the notebook
- `README.md`: Documentation and walkthrough

## Notebook Objective

The notebook is structured as a coursework-style EDA flow:

1. Dataset exploration
2. Data preprocessing and cleaning checks
3. Basic categorical analysis
4. Visual analysis for distribution, relationships, and outliers
5. Final observations and modeling recommendations

## Dataset Snapshot

Based on the notebook workflow and reproduced metrics:

- Shape: `19,237 rows x 18 columns`
- Target column used for analysis: `Price`
- Main numeric fields referenced in visual analysis: `Price`, `Prod. year`, `Airbags`, `Cylinders`, transformed `Mileage`, transformed `Engine volume`
- Main categorical fields referenced: `Fuel type`, `Manufacturer`

Columns in the dataset:

`ID, Price, Levy, Manufacturer, Model, Prod. year, Category, Leather interior, Fuel type, Engine volume, Mileage, Cylinders, Gear box type, Drive wheels, Doors, Wheel, Color, Airbags`

## Step-by-Step Notebook Walkthrough

### 1) Dataset Exploration

The notebook imports `pandas`, `matplotlib`, and `seaborn`, then loads:

```python
df = pd.read_csv('Price_prediction.csv')
```

It then checks:

- First 5 rows (`df.head()`)
- Dimensionality (`df.shape`)
- Data types (`df.dtypes`)
- Descriptive statistics (`df.describe()`)

Why this matters:
- Confirms the data loaded correctly.
- Identifies which fields are still text and need conversion (`Engine volume`, `Mileage`, etc.).
- Gives a first look at scale and spread for numeric variables.

### 2) Data Preprocessing Checks

The notebook evaluates missing data with:

- Per-column null counts: `df.isnull().sum()`
- Total null count: `df.isnull().sum().sum()`
- Missing columns list: `df.columns[df.isnull().any()]`

Observed missingness:

- Total missing values: `20`
- Affected column: `Cylinders` (20 missing)

Cleaning approach used:

```python
df_cleaned = df.dropna()
```

Dataset size comparison:

- Before cleaning: `(19237, 18)`
- After cleaning: `(19217, 18)`

Interpretation:
- Only 20 rows are removed, so dropna has a small impact on sample size.
- For modeling, imputation can still be preferable to preserve all records.

### 3) Categorical Analysis

The notebook computes category frequencies using the cleaned dataset:

```python
df_cleaned['Fuel type'].value_counts()
df_cleaned['Manufacturer'].value_counts()
```

Fuel type counts:

- Petrol: `10137`
- Diesel: `4031`
- Hybrid: `3577`
- LPG: `891`
- CNG: `494`
- Plug-in Hybrid: `86`
- Hydrogen: `1`

Top manufacturers by count:

- HYUNDAI: `3769`
- TOYOTA: `3660`
- MERCEDES-BENZ: `2072`
- FORD: `1108`
- CHEVROLET: `1069`
- BMW: `1049`
- LEXUS: `981`
- HONDA: `977`

Interpretation:
- The dataset is class-imbalanced in several categorical dimensions.
- Rare categories (e.g., Hydrogen) may need grouping or careful encoding in ML.

### 4) Visual Analysis

The notebook includes five major visual tasks:

1. Countplot of `Fuel type`
2. Scatterplot of transformed `Engine volume` vs `Price`
3. Histogram of `Price`
4. Pairplot among selected numeric features
5. Boxplots for outlier detection

#### 4.1 Countplot: Cars by Fuel Type

Shows category imbalance directly and confirms Petrol dominance.

#### 4.2 Scatterplot: Engine Volume vs Price

Preprocessing in notebook:

```python
df['Engine volume'] = df['Engine volume'].astype(str).str.replace(' Turbo', '', case=False).astype(float)
df_filtered = df[df['Price'] < 500000]
```

Outlier filter effect:

- Records kept under 500k: `19,234`
- Records removed as extreme outliers: `3`

Interpretation:
- Removing extreme prices makes the central relationship easier to inspect.
- A weak positive visual trend can exist, but price variation remains high.

#### 4.3 Histogram: Price Distribution

Preprocessing in notebook:

```python
df_normal_prices = df[df['Price'] < 100000]
```

Filter effect:

- Records kept under 100k: `19,124`
- High-end records excluded: `113`

Distribution characteristics:

- Strong right skew in original prices
- Median price: `13,172`
- 75th percentile: `22,075`
- 99th percentile: `84,675`
- Maximum observed price: `26,307,500`

Interpretation:
- Most cars sit in a moderate price range.
- A few very high prices heavily distort mean and variance.

#### 4.4 Pairplot of Selected Features

Notebook transformation:

```python
df['Mileage'] = df['Mileage'].astype(str).str.replace(' km', '', case=False).str.replace(',', '').astype(int)
features = ['Price', 'Mileage', 'Engine volume', 'Cylinders']
sns.pairplot(df_filtered[features])
```

Key numeric pattern:

- `Engine volume` and `Cylinders` are strongly related (`corr ~ 0.78`)

Notes:
- In the raw data, `Mileage` contains extreme values (max near integer limit), so robust outlier handling is important before modeling.

#### 4.5 Boxplots for Outlier Detection

Features plotted:

- `Price`
- `Prod. year`
- `Cylinders`
- `Airbags`

Interpretation:
- `Price` has clear high-end outliers.
- `Prod. year`, `Cylinders`, and `Airbags` show narrower central ranges but still contain edge cases.

## Key Findings

- Price is heavily right-skewed with extreme outliers.
- Missing data is limited and concentrated in one column (`Cylinders`).
- Dataset has strong category imbalance (fuel type and manufacturer).
- Engine and cylinder configuration are strongly linked.
- Raw string-formatted numeric columns (`Engine volume`, `Mileage`) require cleaning before ML.

## Recommendations for Price Prediction Modeling

1. Clean and type-cast fields before training:
    - Strip text suffixes from `Engine volume` and `Mileage`
    - Convert `Doors` into a consistent numeric/category format
    - Handle non-numeric placeholders in `Levy`
2. Handle outliers deliberately:
    - Use capping/winsorization, robust losses, or segmented models
3. Encode categoricals properly:
    - One-hot encoding or target encoding depending on model choice
4. Drop non-predictive identifiers:
    - Remove `ID`
5. Validate with robust metrics:
    - Prefer MAE/RMSE with cross-validation, and inspect residuals by price band

## Environment Setup

From the project folder `Predicting Car Prices`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas matplotlib seaborn jupyter
```

Then open and run the notebook:

```bash
jupyter notebook
```


