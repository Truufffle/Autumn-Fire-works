# Northeast China Fire-Emission Dissertation Code

This repository contains the Python code used for the MSc dissertation:

**The Post-2018 Shift in Autumn Fire Emissions in Northeast China:
Evidence from Multiple Inventories, Pollutants and Explanatory Checks**

## Study area

Heilongjiang, Jilin and Liaoning, Northeast China.

## Main datasets

- GFED5.1
- FINNv2.5 MODIS-only
- CAMS GFAS
- ERA5
- Provincial agricultural statistics

## Analysis

The code includes:

- seasonal and long-term GFED analysis
- segmented breakpoint regression
- multi-pollutant Pearson and Spearman correlations
- baseline and extreme-year sensitivity tests
- agricultural activity-emission associations
- Benjamini-Hochberg FDR correction
- soybean-proxy sensitivity analysis
- meteorological adjustment
- cross-inventory comparison

## Important methodological note

The fire inventories were analysed as total regional fire emissions.
No source-specific agricultural-fire filter was applied because an
equivalent agricultural-fire classification was not available across
GFED, FINN and GFAS.

## Data availability

The original fire-emission and meteorological datasets are not included
in this repository. They are available from their respective data
providers. Local file paths in the Python script should be updated
before execution.

## Software

The analysis was conducted in Python.
