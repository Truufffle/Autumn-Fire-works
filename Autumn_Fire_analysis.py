
# IMPORTS

import os
import glob
import calendar

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely

from scipy.stats import (
    linregress,
    spearmanr
)

import statsmodels.api as sm

from statsmodels.stats.multitest import (
    multipletests
)

# 1. PATHS

# Processed GFED regional data used for the main statistical analysis.
GFED_FILE = (
    r"path\to\GFED_NEChina_annual_spring_autumn_1997_2022.csv"
)

GFED_MONTHLY_FILE = (
    r"path\to\GFED_NEChina_monthly_1997_2022.csv"
)


# FINN raw MODIS-only files.
FINN_FOLDER = (
    r"path\to\FINN_monthly"
)


# Final processed GFAS data in Gg.
# Change the filename only if your final GFAS file has another name.
GFAS_FILE = (
    r"path\to\GFAS_NEChina_annual_spring_autumn_2003_2024.csv"
)


# Agricultural data.
CROP_FILE = (
    r"path\to\crop_data_NEChina.xlsx"
)


# Northeast China dissolved boundary.
SHAPEFILE = (
    r"path\to\NE_China_3provinces_dissolved.shp"
)


# ERA5 files.
ERA5_PRECIP_FILE = (
    r"path\to\data_stream-moda_stepType-avgad.nc"
)

ERA5_MET_FILE = (
    r"path\to\data_stream-moda_stepType-avgua.nc"
)


# 2. COMMON SETTINGS

POLLUTANTS = [
    "CO",
    "PM2.5",
    "OC",
    "BC",
    "NOx"
]

COMMON_INVENTORY_POLLUTANTS = [
    "BC",
    "CO",
    "OC",
    "PM2.5"
]

PRE_START = 2012
PRE_END = 2017

POST_START = 2018
POST_END = 2022

# 3. READ GFED

gfed = pd.read_csv(
    GFED_FILE
)

gfed = gfed.sort_values(
    [
        "pollutant",
        "season",
        "year"
    ]
).copy()


print("\n--- GFED DATA CHECK ---")

print(
    "Years:",
    gfed["year"].min(),
    "-",
    gfed["year"].max()
)

print(
    "Pollutants:",
    gfed["pollutant"]
    .unique()
)

print(
    "Seasons:",
    gfed["season"]
    .unique()
)


# 4. GFED MONTHLY CLIMATOLOGY

gfed_monthly = pd.read_csv(
    GFED_MONTHLY_FILE
)


monthly_climatology = (
    gfed_monthly
    .groupby(
        [
            "month",
            "pollutant"
        ],
        as_index=False
    )
    ["emission_Gg"]
    .mean()
)


# Normalize each pollutant to its own maximum monthly climatological value.
monthly_climatology[
    "normalized_emission"
] = (
    monthly_climatology
    .groupby(
        "pollutant"
    )
    ["emission_Gg"]
    .transform(
        lambda x:
            x / x.max()
    )
)


print(
    "\n--- GFED MONTHLY CLIMATOLOGY ---"
)

print(
    monthly_climatology
    .round(3)
    .to_string(index=False)
)



# 5. GFED NORMALIZED TEMPORAL SERIES
#    1997-2011 mean = 100

reference_1997_2011 = (
    gfed[
        gfed["year"].between(
            1997,
            2011
        )
    ]
    .groupby(
        [
            "pollutant",
            "season"
        ]
    )
    ["emission_Gg"]
    .mean()
    .rename(
        "reference_1997_2011"
    )
)


gfed_index = gfed.join(
    reference_1997_2011,
    on=[
        "pollutant",
        "season"
    ]
)


gfed_index["emission_index"] = (
    gfed_index["emission_Gg"]
    /
    gfed_index["reference_1997_2011"]
    * 100
)


# 6. LONG-TERM LINEAR TREND ANALYSIS

trend_results = []


for pollutant in POLLUTANTS:

    for season in [
        "Annual",
        "Spring",
        "Autumn"
    ]:

        data = gfed[
            (gfed["pollutant"] == pollutant)
            &
            (gfed["season"] == season)
            &
            (gfed["year"].between(
                1997,
                2022
            ))
        ].copy()


        # Simple OLS-equivalent linear regression
        reg = linregress(
            data["year"],
            data["emission_Gg"]
        )


        trend_results.append({

            "pollutant":
                pollutant,

            "season":
                season,

            "slope_Gg_per_year":
                reg.slope,

            "p_value":
                reg.pvalue,

            "R_squared":
                reg.rvalue ** 2

        })


trend_results = pd.DataFrame(
    trend_results
)


print(
    "\n--- GFED LONG-TERM TRENDS: 1997-2022 ---"
)

print(
    trend_results
    .round(4)
    .to_string(index=False)
)

# 7. PERIOD COMPARISON
#    2012-2017 vs 2018-2022

period_results = []


for pollutant in POLLUTANTS:

    for season in [
        "Annual",
        "Spring",
        "Autumn"
    ]:

        data = gfed[
            (gfed["pollutant"] == pollutant)
            &
            (gfed["season"] == season)
        ]


        pre = data[
            data["year"].between(
                PRE_START,
                PRE_END
            )
        ]["emission_Gg"]


        post = data[
            data["year"].between(
                POST_START,
                POST_END
            )
        ]["emission_Gg"]


        pre_mean = pre.mean()

        post_mean = post.mean()


        change_percent = (
            (post_mean - pre_mean)
            /
            pre_mean
            * 100
        )


        period_results.append({

            "pollutant":
                pollutant,

            "season":
                season,

            "2012_2017_mean_Gg":
                pre_mean,

            "2018_2022_mean_Gg":
                post_mean,

            "change_percent":
                change_percent

        })


period_results = pd.DataFrame(
    period_results
)


print(
    "\n--- GFED PERIOD COMPARISON ---"
)

print(
    period_results
    .round(2)
    .to_string(index=False)
)


# 8. SEGMENTED REGRESSION FUNCTION

def run_segmented_regression(
    data,
    breakpoint,
    start_year=None,
    end_year=None
):

    """
    Segmented OLS regression with HAC standard errors.

    Model:
        emission =
        constant
        + beta1 * time
        + beta2 * post
        + beta3 * time_after

    beta2 = level change at the breakpoint
    beta3 = change in slope after the breakpoint

    Percentage level change:
        beta2 / expected emission at breakpoint
        under the pre-break trend * 100
    """

    d = data.copy()


    if start_year is not None:

        d = d[
            d["year"] >= start_year
        ]


    if end_year is not None:

        d = d[
            d["year"] <= end_year
        ]


    d = (
        d
        .sort_values(
            "year"
        )
        .copy()
    )


    first_year = (
        d["year"].min()
    )


    d["time"] = (
        d["year"]
        -
        first_year
    )


    d["post"] = (
        d["year"]
        >=
        breakpoint
    ).astype(int)


    d["time_after"] = np.where(
        d["year"]
        >=
        breakpoint,

        d["year"]
        -
        breakpoint,

        0
    )


    X = sm.add_constant(
        d[
            [
                "time",
                "post",
                "time_after"
            ]
        ]
    )


    y = d[
        "emission_Gg"
    ]


    model = sm.OLS(
        y,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": 1
        }
    )


    pre_slope = (
        model.params[
            "time"
        ]
    )


    level_change = (
        model.params[
            "post"
        ]
    )


    slope_change = (
        model.params[
            "time_after"
        ]
    )


    post_slope = (
        pre_slope
        +
        slope_change
    )


    # Predicted emission at the breakpoint.
    # if the pre-break trend had continued.
    expected_at_breakpoint = (
        model.params[
            "const"
        ]
        +
        pre_slope
        *
        (
            breakpoint
            -
            first_year
        )
    )


    level_change_percent = (
        level_change
        /
        expected_at_breakpoint
        * 100
    )


    return {

        "pre_slope_Gg_per_year":
            pre_slope,

        "level_change_Gg":
            level_change,

        "level_change_percent":
            level_change_percent,

        "level_change_p":
            model.pvalues[
                "post"
            ],

        "slope_change_Gg_per_year":
            slope_change,

        "slope_change_p":
            model.pvalues[
                "time_after"
            ],

        "post_slope_Gg_per_year":
            post_slope,

        "R_squared":
            model.rsquared

    }


# 9. FULL-RECORD BREAKPOINT ANALYSIS
#    3 seasons x 5 pollutants x 2 breakpoints = 30 models

breakpoint_results = []


for season in [
    "Annual",
    "Spring",
    "Autumn"
]:

    for pollutant in POLLUTANTS:

        data = gfed[
            (gfed["pollutant"] == pollutant)
            &
            (gfed["season"] == season)
        ].copy()


        for breakpoint in [
            2012,
            2018
        ]:

            result = (
                run_segmented_regression(
                    data,
                    breakpoint
                )
            )


            result[
                "pollutant"
            ] = pollutant


            result[
                "season"
            ] = season


            result[
                "breakpoint"
            ] = breakpoint


            breakpoint_results.append(
                result
            )


breakpoint_results = pd.DataFrame(
    breakpoint_results
)


print(
    "\n--- FULL-RECORD SEGMENTED REGRESSIONS ---"
)

print(
    breakpoint_results[
        [
            "pollutant",
            "season",
            "breakpoint",
            "level_change_percent",
            "level_change_p",
            "slope_change_p",
            "R_squared"
        ]
    ]
    .round(4)
    .to_string(index=False)
)


print(
    "\n--- AUTUMN BREAKPOINT RESULTS USED IN MAIN FIGURE ---"
)

print(
    breakpoint_results[
        breakpoint_results[
            "season"
        ]
        ==
        "Autumn"
    ][
        [
            "pollutant",
            "breakpoint",
            "level_change_percent",
            "level_change_p"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# 10. RESTRICTED 2012-2022 BREAKPOINT SENSITIVITY

restricted_results = []


for pollutant in POLLUTANTS:

    data = gfed[
        (gfed["pollutant"] == pollutant)
        &
        (gfed["season"] == "Autumn")
    ].copy()


    result = run_segmented_regression(
        data,
        breakpoint=2018,
        start_year=2012,
        end_year=2022
    )


    result[
        "pollutant"
    ] = pollutant


    restricted_results.append(
        result
    )


restricted_results = pd.DataFrame(
    restricted_results
)


print(
    "\n--- RESTRICTED 2012-2022 BREAKPOINT SENSITIVITY ---"
)

print(
    restricted_results[
        [
            "pollutant",
            "level_change_percent",
            "level_change_p",
            "slope_change_p",
            "R_squared"
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# 11. PEARSON + SPEARMAN MULTI-POLLUTANT CONSISTENCY

pearson_matrices = {}

spearman_matrices = {}


for season in [
    "Annual",
    "Spring",
    "Autumn"
]:

    wide = (
        gfed[
            gfed["season"]
            ==
            season
        ]
        .pivot(
            index="year",
            columns="pollutant",
            values="emission_Gg"
        )
        [POLLUTANTS]
    )


    pearson_matrices[
        season
    ] = wide.corr(
        method="pearson"
    )


    spearman_matrices[
        season
    ] = wide.corr(
        method="spearman"
    )


    print(
        f"\n--- {season.upper()} PEARSON ---"
    )

    print(
        pearson_matrices[
            season
        ]
        .round(3)
    )


    print(
        f"\n--- {season.upper()} SPEARMAN ---"
    )

    print(
        spearman_matrices[
            season
        ]
        .round(3)
    )


# 12. MEAN VS MEDIAN SENSITIVITY

mean_median_results = []


for pollutant in POLLUTANTS:

    for season in [
        "Annual",
        "Spring",
        "Autumn"
    ]:

        data = gfed[
            (gfed["pollutant"] == pollutant)
            &
            (gfed["season"] == season)
        ]


        pre = data[
            data["year"].between(
                2012,
                2017
            )
        ]["emission_Gg"]


        post = data[
            data["year"].between(
                2018,
                2022
            )
        ]["emission_Gg"]


        mean_change = (
            (
                post.mean()
                -
                pre.mean()
            )
            /
            pre.mean()
            * 100
        )


        median_change = (
            (
                post.median()
                -
                pre.median()
            )
            /
            pre.median()
            * 100
        )


        mean_median_results.append({

            "pollutant":
                pollutant,

            "season":
                season,

            "mean_change_percent":
                mean_change,

            "median_change_percent":
                median_change

        })


mean_median_results = pd.DataFrame(
    mean_median_results
)


print(
    "\n--- MEAN VS MEDIAN SENSITIVITY ---"
)

print(
    mean_median_results
    .round(2)
    .to_string(index=False)
)

# 13. IQR EXTREME-YEAR SENSITIVITY

# Full pre-2018 autumn record: 1997-2017.
autumn_pre2018 = (
    gfed[
        (gfed["season"] == "Autumn")
        &
        (gfed["year"] <= 2017)
    ]
    .pivot(
        index="year",
        columns="pollutant",
        values="emission_Gg"
    )
    [POLLUTANTS]
)


# Normalize each pollutant by its 1997-2017 pre-2018 mean.
normalized_pre2018 = (
    autumn_pre2018
    /
    autumn_pre2018.mean()
)


# Composite multi-pollutant autumn index.
composite_pre2018 = (
    normalized_pre2018
    .mean(
        axis=1
    )
)


Q1 = composite_pre2018.quantile(
    0.25
)

Q3 = composite_pre2018.quantile(
    0.75
)

IQR = (
    Q3
    -
    Q1
)

upper_threshold = (
    Q3
    +
    1.5
    *
    IQR
)


extreme_years = (
    composite_pre2018[
        composite_pre2018
        >
        upper_threshold
    ]
    .index
    .tolist()
)


print(
    "\n--- IQR EXTREME AUTUMN YEARS ---"
)

print(
    "Q1:",
    round(
        Q1,
        3
    )
)

print(
    "Q3:",
    round(
        Q3,
        3
    )
)

print(
    "IQR:",
    round(
        IQR,
        3
    )
)

print(
    "Upper threshold:",
    round(
        upper_threshold,
        3
    )
)

print(
    "Extreme years:",
    extreme_years
)


# Only extreme years that fall inside the 2012-2017 baseline
# affect the main period sensitivity.

baseline_extreme_years = [
    year
    for year
    in extreme_years
    if 2012 <= year <= 2017
]


print(
    "Extreme years within 2012-2017 baseline:",
    baseline_extreme_years
)


gfed_without_extremes = gfed[
    ~gfed["year"].isin(
        baseline_extreme_years
    )
].copy()


iqr_sensitivity = []


for pollutant in POLLUTANTS:

    data = gfed_without_extremes[
        (gfed_without_extremes["pollutant"] == pollutant)
        &
        (
            gfed_without_extremes["season"]
            ==
            "Autumn"
        )
    ]


    pre = data[
        data["year"].between(
            2012,
            2017
        )
    ]["emission_Gg"]


    post = data[
        data["year"].between(
            2018,
            2022
        )
    ]["emission_Gg"]


    change_percent = (
        (
            post.mean()
            -
            pre.mean()
        )
        /
        pre.mean()
        * 100
    )


    iqr_sensitivity.append({

        "pollutant":
            pollutant,

        "change_without_baseline_extremes_percent":
            change_percent

    })


iqr_sensitivity = pd.DataFrame(
    iqr_sensitivity
)


print(
    "\n--- IQR SENSITIVITY RESULTS ---"
)

print(
    iqr_sensitivity
    .round(2)
    .to_string(index=False)
)

# 14. LEAVE-ONE-YEAR-OUT SENSITIVITY

leave_one_out = []


for omitted_year in range(
    2012,
    2018
):

    for pollutant in POLLUTANTS:

        data = gfed[
            (gfed["pollutant"] == pollutant)
            &
            (gfed["season"] == "Autumn")
        ]


        pre = data[
            data["year"].between(
                2012,
                2017
            )
            &
            (
                data["year"]
                !=
                omitted_year
            )
        ]["emission_Gg"]


        post = data[
            data["year"].between(
                2018,
                2022
            )
        ]["emission_Gg"]


        change_percent = (
            (
                post.mean()
                -
                pre.mean()
            )
            /
            pre.mean()
            * 100
        )


        leave_one_out.append({

            "omitted_year":
                omitted_year,

            "pollutant":
                pollutant,

            "change_percent":
                change_percent

        })


leave_one_out = pd.DataFrame(
    leave_one_out
)


print(
    "\n--- LEAVE-ONE-YEAR-OUT: AUTUMN ---"
)

print(
    leave_one_out
    .pivot(
        index="omitted_year",
        columns="pollutant",
        values="change_percent"
    )
    .round(2)
)


leave_one_out_range = (
    leave_one_out
    .groupby(
        "pollutant"
    )
    ["change_percent"]
    .agg(
        [
            "min",
            "max"
        ]
    )
)


print(
    "\n--- LEAVE-ONE-OUT RANGES ---"
)

print(
    leave_one_out_range
    .round(2)
)

# 15. AGRICULTURAL DATA

crop = pd.read_excel(
    CROP_FILE,
    sheet_name="Clean_Output"
)


# Standardise year-column name.
if "data_year" in crop.columns:

    crop = crop.rename(
        columns={
            "data_year":
                "year"
        }
    )


crop["crop_code"] = (
    crop["crop_code"]
    .str.lower()
)


print(
    "\n--- AGRICULTURAL DATA CHECK ---"
)

print(
    "Years:",
    crop["year"].min(),
    "-",
    crop["year"].max()
)

print(
    "Provinces:",
    crop["province"].unique()
)

print(
    "Crops:",
    crop["crop_code"].unique()
)

# 16. COMBINED MAJOR-CROP ACTIVITY

selected_crop_data = crop[
    crop["crop_code"].isin(
        [
            "maize",
            "rice",
            "soybean"
        ]
    )
].copy()


combined_agriculture = (
    selected_crop_data
    .groupby(
        "year",
        as_index=False
    )
    .agg(

        combined_production_tonnes=(
            "production_tonnes",
            "sum"
        ),

        combined_sown_area_ha=(
            "sown_area_ha",
            "sum"
        )

    )
)


# Area-weighted aggregate yield:
# combined production / combined area.

combined_agriculture[
    "aggregate_yield_t_ha"
] = (
    combined_agriculture[
        "combined_production_tonnes"
    ]
    /
    combined_agriculture[
        "combined_sown_area_ha"
    ]
)


def calculate_period_change(
    data,
    variable
):

    pre = data[
        data["year"].between(
            2012,
            2017
        )
    ][variable]


    post = data[
        data["year"].between(
            2018,
            2022
        )
    ][variable]


    return (
        (
            post.mean()
            -
            pre.mean()
        )
        /
        pre.mean()
        * 100
    )


combined_changes = pd.DataFrame({

    "variable": [
        "Combined major-crop production",
        "Combined major-crop sown area",
        "Area-weighted aggregate yield"
    ],

    "change_percent": [

        calculate_period_change(
            combined_agriculture,
            "combined_production_tonnes"
        ),

        calculate_period_change(
            combined_agriculture,
            "combined_sown_area_ha"
        ),

        calculate_period_change(
            combined_agriculture,
            "aggregate_yield_t_ha"
        )

    ]

})


print(
    "\n--- COMBINED MAJOR-CROP CHANGES ---"
)

print(
    combined_changes
    .round(2)
    .to_string(index=False)
)


# 17. CROP-SPECIFIC ACTIVITY AND COMPOSITION

crop_year = (
    selected_crop_data
    .groupby(
        [
            "year",
            "crop_code"
        ],
        as_index=False
    )
    .agg(

        production_tonnes=(
            "production_tonnes",
            "sum"
        ),

        sown_area_ha=(
            "sown_area_ha",
            "sum"
        )

    )
)


crop_specific_changes = []


for crop_name in [
    "maize",
    "rice",
    "soybean"
]:

    data = crop_year[
        crop_year["crop_code"]
        ==
        crop_name
    ]


    crop_specific_changes.append({

        "crop":
            crop_name,

        "production_change_percent":
            calculate_period_change(
                data,
                "production_tonnes"
            ),

        "area_change_percent":
            calculate_period_change(
                data,
                "sown_area_ha"
            )

    })


crop_specific_changes = pd.DataFrame(
    crop_specific_changes
)


print(
    "\n--- CROP-SPECIFIC CHANGES ---"
)

print(
    crop_specific_changes
    .round(2)
    .to_string(index=False)
)


# Composition shares.
crop_year[
    "combined_year_production"
] = (
    crop_year
    .groupby(
        "year"
    )
    ["production_tonnes"]
    .transform(
        "sum"
    )
)


crop_year[
    "combined_year_area"
] = (
    crop_year
    .groupby(
        "year"
    )
    ["sown_area_ha"]
    .transform(
        "sum"
    )
)


crop_year[
    "production_share_percent"
] = (
    crop_year[
        "production_tonnes"
    ]
    /
    crop_year[
        "combined_year_production"
    ]
    * 100
)


crop_year[
    "area_share_percent"
] = (
    crop_year[
        "sown_area_ha"
    ]
    /
    crop_year[
        "combined_year_area"
    ]
    * 100
)


# 18. GFED AUTUMN COMPOSITE INDEX
#    2012-2017 mean = 100 for each pollutant

gfed_autumn = gfed[
    (gfed["season"] == "Autumn")
    &
    (
        gfed["pollutant"]
        .isin(
            POLLUTANTS
        )
    )
].copy()


autumn_baseline = (
    gfed_autumn[
        gfed_autumn["year"]
        .between(
            2012,
            2017
        )
    ]
    .groupby(
        "pollutant"
    )
    ["emission_Gg"]
    .mean()
    .rename(
        "baseline_2012_2017"
    )
)


gfed_autumn = gfed_autumn.join(
    autumn_baseline,
    on="pollutant"
)


gfed_autumn[
    "emission_index"
] = (
    gfed_autumn[
        "emission_Gg"
    ]
    /
    gfed_autumn[
        "baseline_2012_2017"
    ]
    * 100
)


gfed_autumn_composite = (
    gfed_autumn
    .groupby(
        "year",
        as_index=False
    )
    ["emission_index"]
    .mean()
    .rename(
        columns={
            "emission_index":
                "GFED_autumn_index"
        }
    )
)


# 19. AGRICULTURAL CORRELATION DATA
#    Common period = 2002-2022

crop_wide = (
    crop_year
    .pivot(
        index="year",
        columns="crop_code",
        values=[
            "production_tonnes",
            "sown_area_ha"
        ]
    )
)


crop_wide.columns = [

    (
        f"{crop_name}_production"
        if measure
        ==
        "production_tonnes"

        else
        f"{crop_name}_area"
    )

    for measure, crop_name
    in crop_wide.columns

]


crop_wide = (
    crop_wide
    .reset_index()
)


agri_correlation_data = (
    combined_agriculture
    .merge(
        crop_wide,
        on="year",
        how="inner"
    )
    .merge(
        gfed_autumn_composite,
        on="year",
        how="inner"
    )
)


agri_correlation_data = (
    agri_correlation_data[
        agri_correlation_data[
            "year"
        ].between(
            2002,
            2022
        )
    ]
    .sort_values(
        "year"
    )
    .copy()
)


# 20. AGRICULTURAL SPEARMAN CORRELATIONS
#    8 variables x 2 test types = 16 tests

AGRICULTURAL_VARIABLES = [

    "combined_production_tonnes",

    "combined_sown_area_ha",

    "maize_production",

    "maize_area",

    "rice_production",

    "rice_area",

    "soybean_production",

    "soybean_area"

]


agricultural_tests = []


for variable in AGRICULTURAL_VARIABLES:

    data = (
        agri_correlation_data[
            [
                "year",
                variable,
                "GFED_autumn_index"
            ]
        ]
        .dropna()
        .copy()
    )


    # Level correlation

    rho_level, p_level = spearmanr(

        data[
            variable
        ],

        data[
            "GFED_autumn_index"
        ]

    )


    agricultural_tests.append({

        "variable":
            variable,

        "test_type":
            "level",

        "rho":
            rho_level,

        "p_nominal":
            p_level

    })


    # Year-to-year percentage changes

    data = (
        data
        .sort_values(
            "year"
        )
        .copy()
    )


    data[
        "variable_yoy"
    ] = (
        data[
            variable
        ]
        .pct_change()
        * 100
    )


    data[
        "emission_yoy"
    ] = (
        data[
            "GFED_autumn_index"
        ]
        .pct_change()
        * 100
    )


    yoy = data[
        [
            "variable_yoy",
            "emission_yoy"
        ]
    ].dropna()


    rho_yoy, p_yoy = spearmanr(

        yoy[
            "variable_yoy"
        ],

        yoy[
            "emission_yoy"
        ]

    )


    agricultural_tests.append({

        "variable":
            variable,

        "test_type":
            "year_to_year",

        "rho":
            rho_yoy,

        "p_nominal":
            p_yoy

    })


agricultural_tests = pd.DataFrame(
    agricultural_tests
)


print(
    "\n--- 16 AGRICULTURAL ASSOCIATION TESTS ---"
)

print(
    agricultural_tests
    .round(3)
    .to_string(index=False)
)


# 21. BENJAMINI-HOCHBERG FDR CORRECTION
#    Applied jointly across all 16 tests

reject_fdr, p_fdr, _, _ = multipletests(

    agricultural_tests[
        "p_nominal"
    ].values,

    alpha=0.05,

    method="fdr_bh"

)


agricultural_tests[
    "p_FDR"
] = p_fdr


agricultural_tests[
    "significant_FDR_0.05"
] = reject_fdr


print(
    "\n--- AGRICULTURAL TESTS WITH BH-FDR ---"
)

print(
    agricultural_tests
    .sort_values(
        "p_nominal"
    )
    .round(3)
    .to_string(index=False)
)


# 22. JILIN SOYBEAN-PROXY SENSITIVITY

# Historical Jilin soybean-production observations used a broader
# beans-total proxy where soybean-specific production was unavailable.
# This sensitivity excludes Jilin completely from soybean production.


soybean_no_jilin = selected_crop_data[
    (selected_crop_data["crop_code"] == "soybean")
    &
    (
        selected_crop_data["province"]
        !=
        "Jilin"
    )
].copy()


soybean_no_jilin_year = (
    soybean_no_jilin
    .groupby(
        "year",
        as_index=False
    )
    .agg(
        soybean_production_tonnes=(
            "production_tonnes",
            "sum"
        )
    )
)


soybean_change_no_jilin = (
    calculate_period_change(
        soybean_no_jilin_year,
        "soybean_production_tonnes"
    )
)


soybean_sensitivity_data = (
    soybean_no_jilin_year
    .merge(
        gfed_autumn_composite,
        on="year",
        how="inner"
    )
)


soybean_sensitivity_data = (
    soybean_sensitivity_data[
        soybean_sensitivity_data[
            "year"
        ].between(
            2002,
            2022
        )
    ]
    .sort_values(
        "year"
    )
    .copy()
)


rho_soy_level, p_soy_level = spearmanr(

    soybean_sensitivity_data[
        "soybean_production_tonnes"
    ],

    soybean_sensitivity_data[
        "GFED_autumn_index"
    ]

)


soybean_sensitivity_data[
    "soybean_yoy"
] = (
    soybean_sensitivity_data[
        "soybean_production_tonnes"
    ]
    .pct_change()
    * 100
)


soybean_sensitivity_data[
    "GFED_yoy"
] = (
    soybean_sensitivity_data[
        "GFED_autumn_index"
    ]
    .pct_change()
    * 100
)


soybean_yoy_data = (
    soybean_sensitivity_data[
        [
            "soybean_yoy",
            "GFED_yoy"
        ]
    ]
    .dropna()
)


rho_soy_yoy, p_soy_yoy = spearmanr(

    soybean_yoy_data[
        "soybean_yoy"
    ],

    soybean_yoy_data[
        "GFED_yoy"
    ]

)


print(
    "\n--- SOYBEAN SENSITIVITY: EXCLUDING JILIN ---"
)

print(
    "2012-2017 vs 2018-2022 production change:",
    round(
        soybean_change_no_jilin,
        2
    ),
    "%"
)

print(
    "Level rho:",
    round(
        rho_soy_level,
        3
    )
)

print(
    "Level nominal p:",
    round(
        p_soy_level,
        3
    )
)

print(
    "YoY rho:",
    round(
        rho_soy_yoy,
        3
    )
)

print(
    "YoY nominal p:",
    round(
        p_soy_yoy,
        3
    )
)

# 23. READ STUDY-AREA SHAPEFILE FOR ERA5 AND FINN

region = gpd.read_file(
    SHAPEFILE
)

region = region.to_crs(
    "EPSG:4326"
)


geometry = (
    region.geometry
    .union_all()
)


min_lon, min_lat, max_lon, max_lat = (
    region.total_bounds
)

# 24. ERA5 PROCESSING


ds_p = xr.open_dataset(
    ERA5_PRECIP_FILE
)

ds_m = xr.open_dataset(
    ERA5_MET_FILE
)


# Select regional bounding box.
lat = ds_p[
    "latitude"
]

lon = ds_p[
    "longitude"
]


ds_p = ds_p.sel(

    latitude=lat[
        (lat >= min_lat)
        &
        (lat <= max_lat)
    ],

    longitude=lon[
        (lon >= min_lon)
        &
        (lon <= max_lon)
    ]

)


ds_m = ds_m.sel(

    latitude=ds_m.latitude[
        (ds_m.latitude >= min_lat)
        &
        (ds_m.latitude <= max_lat)
    ],

    longitude=ds_m.longitude[
        (ds_m.longitude >= min_lon)
        &
        (ds_m.longitude <= max_lon)
    ]

)


lon2d, lat2d = np.meshgrid(

    ds_p.longitude.values,

    ds_p.latitude.values

)


inside_ne = shapely.contains_xy(

    geometry,

    lon2d,

    lat2d

)


# Cosine-latitude weights.

weights = np.cos(

    np.deg2rad(
        ds_p.latitude.values
    )

)


weights_2d = np.broadcast_to(

    weights[
        :,
        None
    ],

    inside_ne.shape

)


weights_2d = np.where(

    inside_ne,

    weights_2d,

    np.nan

)


def spatial_mean(
    data
):

    values = data.values


    valid_weights = np.where(

        np.isfinite(
            values
        ),

        weights_2d,

        np.nan

    )


    return (

        np.nansum(
            values
            *
            valid_weights
        )

        /

        np.nansum(
            valid_weights
        )

    )


# 25. ERA5 RELATIVE HUMIDITY

temperature_c = (
    ds_m["t2m"]
    -
    273.15
)


dewpoint_c = (
    ds_m["d2m"]
    -
    273.15
)


# Magnus formulation.

rh = (

    100

    *

    np.exp(

        (
            17.625
            *
            dewpoint_c
        )

        /

        (
            243.04
            +
            dewpoint_c
        )

    )

    /

    np.exp(

        (
            17.625
            *
            temperature_c
        )

        /

        (
            243.04
            +
            temperature_c
        )

    )

)


rh = rh.clip(
    min=0,
    max=100
)


# 26. ERA5 WIND SPEED

wind_speed = np.sqrt(

    ds_m["u10"] ** 2

    +

    ds_m["v10"] ** 2

)


# 27. AUTUMN ERA5 REGIONAL SERIES

meteorology_rows = []


for year in range(
    2001,
    2025
):


    # September-November precipitation
    
    precip_total_mm = 0.0


    rh_weighted_sum = 0.0

    wind_weighted_sum = 0.0

    total_days = 0


    for month in [
        9,
        10,
        11
    ]:

        days = calendar.monthrange(
            year,
            month
        )[1]


        # Select monthly values.
        
        precip_month = ds_p[
            "tp"
        ].sel(
            valid_time=(
                ds_p.valid_time.dt.year
                ==
                year
            )
            &
            (
                ds_p.valid_time.dt.month
                ==
                month
            )
        )


        rh_month = rh.sel(
            valid_time=(
                rh.valid_time.dt.year
                ==
                year
            )
            &
            (
                rh.valid_time.dt.month
                ==
                month
            )
        )


        wind_month = wind_speed.sel(
            valid_time=(
                wind_speed.valid_time.dt.year
                ==
                year
            )
            &
            (
                wind_speed.valid_time.dt.month
                ==
                month
            )
        )


        # Skip missing months.
        
        if precip_month.size == 0:

            continue


        precip_month = precip_month.squeeze()

        rh_month = rh_month.squeeze()

        wind_month = wind_month.squeeze()


        precip_spatial = spatial_mean(
            precip_month
        )


        rh_spatial = spatial_mean(
            rh_month
        )


        wind_spatial = spatial_mean(
            wind_month
        )


        # ERA5 monthly averaged daily precipitation:
        # m/day -> mm/month.
        
        precip_month_mm = (
            precip_spatial
            *
            1000
            *
            days
        )


        precip_total_mm += (
            precip_month_mm
        )


        rh_weighted_sum += (
            rh_spatial
            *
            days
        )


        wind_weighted_sum += (
            wind_spatial
            *
            days
        )


        total_days += days


    meteorology_rows.append({

        "year":
            year,

        "precip_mm":
            precip_total_mm,

        "rh_percent":
            rh_weighted_sum
            /
            total_days,

        "wind_ms":
            wind_weighted_sum
            /
            total_days

    })


meteorology = pd.DataFrame(
    meteorology_rows
)


print(
    "\n--- AUTUMN ERA5 METEOROLOGY ---"
)

print(
    meteorology
    .round(3)
    .to_string(index=False)
)


# 28. METEOROLOGICAL PERIOD COMPARISON

meteorology_changes = []


for variable in [

    "precip_mm",

    "rh_percent",

    "wind_ms"

]:

    pre = meteorology[
        meteorology["year"]
        .between(
            2012,
            2017
        )
    ][variable]


    post = meteorology[
        meteorology["year"]
        .between(
            2018,
            2022
        )
    ][variable]


    pre_mean = pre.mean()

    post_mean = post.mean()


    change_percent = (

        (
            post_mean
            -
            pre_mean
        )

        /

        pre_mean

        *

        100

    )


    meteorology_changes.append({

        "variable":
            variable,

        "2012_2017_mean":
            pre_mean,

        "2018_2022_mean":
            post_mean,

        "change_percent":
            change_percent

    })


meteorology_changes = pd.DataFrame(
    meteorology_changes
)


print(
    "\n--- METEOROLOGY PERIOD COMPARISON ---"
)

print(
    meteorology_changes
    .round(3)
    .to_string(index=False)
)


# 29. METEOROLOGICAL ADJUSTMENT

met_regression_results = []


for pollutant in POLLUTANTS:

    emission = gfed[
        (gfed["pollutant"] == pollutant)
        &
        (gfed["season"] == "Autumn")
        &
        (
            gfed["year"]
            .between(
                2001,
                2022
            )
        )
    ][
        [
            "year",
            "emission_Gg"
        ]
    ].copy()


    analysis = emission.merge(

        meteorology,

        on="year",

        how="inner"

    )


    analysis = (
        analysis
        .dropna()
        .sort_values(
            "year"
        )
        .copy()
    )


    analysis[
        "log_emission"
    ] = np.log(

        analysis[
            "emission_Gg"
        ]

    )


    analysis[
        "time"
    ] = (

        analysis[
            "year"
        ]

        -

        analysis[
            "year"
        ].min()

    )


    analysis[
        "post2018"
    ] = (

        analysis[
            "year"
        ]

        >=

        2018

    ).astype(int)


    # Z-standardisation.
    
    for variable in [

        "precip_mm",

        "rh_percent",

        "wind_ms"

    ]:

        analysis[
            f"z_{variable}"
        ] = (

            (
                analysis[
                    variable
                ]

                -

                analysis[
                    variable
                ].mean()
            )

            /

            analysis[
                variable
            ].std(
                ddof=0
            )

        )


    # Unadjusted model

    X1 = sm.add_constant(

        analysis[
            [
                "time",
                "post2018"
            ]
        ]

    )


    model1 = sm.OLS(

        analysis[
            "log_emission"
        ],

        X1

    ).fit(

        cov_type="HAC",

        cov_kwds={
            "maxlags":
                1
        }

    )


    unadjusted_change = (

        (
            np.exp(

                model1.params[
                    "post2018"
                ]

            )

            -

            1

        )

        *

        100

    )


  
    # Meteorology-adjusted model

    X2 = sm.add_constant(

        analysis[
            [
                "time",
                "post2018",
                "z_precip_mm",
                "z_rh_percent",
                "z_wind_ms"
            ]
        ]

    )


    model2 = sm.OLS(

        analysis[
            "log_emission"
        ],

        X2

    ).fit(

        cov_type="HAC",

        cov_kwds={
            "maxlags":
                1
        }

    )


    adjusted_change = (

        (
            np.exp(

                model2.params[
                    "post2018"
                ]

            )

            -

            1

        )

        *

        100

    )


    met_regression_results.append({

        "pollutant":
            pollutant,

        "unadjusted_change_percent":
            unadjusted_change,

        "unadjusted_post2018_p":
            model1.pvalues[
                "post2018"
            ],

        "R2_unadjusted":
            model1.rsquared,

        "adjusted_change_percent":
            adjusted_change,

        "adjusted_post2018_p":
            model2.pvalues[
                "post2018"
            ],

        "R2_adjusted_model":
            model2.rsquared,

        "precip_beta":
            model2.params[
                "z_precip_mm"
            ],

        "precip_p":
            model2.pvalues[
                "z_precip_mm"
            ],

        "rh_beta":
            model2.params[
                "z_rh_percent"
            ],

        "rh_p":
            model2.pvalues[
                "z_rh_percent"
            ],

        "wind_beta":
            model2.params[
                "z_wind_ms"
            ],

        "wind_p":
            model2.pvalues[
                "z_wind_ms"
            ]

    })


met_regression_results = pd.DataFrame(
    met_regression_results
)


print(
    "\n--- METEOROLOGICAL ADJUSTMENT RESULTS ---"
)

print(
    met_regression_results
    .round(4)
    .to_string(index=False)
)


# 30. FINN RAW EXTRACTION
#     MODIS-only, exact NE-China polygon mask

FINN_POLLUTANTS = {

    "BC":
        "fire_modis_BC",

    "CO":
        "fire_modis_CO",

    "OC":
        "fire_modis_OC",

    "PM2.5":
        "fire_modis_PM25"

}


FINN_MOLAR_MASS = {

    "BC":
        12.00,

    "CO":
        28.01,

    "OC":
        12.00,

    "PM2.5":
        12.00

}


AVOGADRO = (
    6.02214076e23
)


SECONDS_PER_DAY = (
    86400
)


# Use first FINN CO file to define grid.

sample_file = sorted(

    glob.glob(

        os.path.join(
            FINN_FOLDER,
            "CO",
            "*.nc"
        )

    )

)[0]


sample = xr.open_dataset(
    sample_file
)


sample = sample.sel(

    lat=sample.lat[
        (sample.lat >= min_lat)
        &
        (sample.lat <= max_lat)
    ],

    lon=sample.lon[
        (sample.lon >= min_lon)
        &
        (sample.lon <= max_lon)
    ]

)


finn_lon2d, finn_lat2d = np.meshgrid(

    sample.lon.values,

    sample.lat.values

)


finn_inside_ne = shapely.contains_xy(

    geometry,

    finn_lon2d,

    finn_lat2d

)


# Exact spherical grid-cell area.

earth_radius_m = (
    6371000.0
)


lat_values = (
    sample.lat.values
)

lon_values = (
    sample.lon.values
)


dlat_deg = np.abs(
    np.mean(
        np.diff(
            lat_values
        )
    )
)


dlon_deg = np.abs(
    np.mean(
        np.diff(
            lon_values
        )
    )
)


lat_edges = np.concatenate([

    [
        lat_values[0]
        -
        dlat_deg / 2
    ],

    (
        lat_values[:-1]
        +
        lat_values[1:]
    )
    /
    2,

    [
        lat_values[-1]
        +
        dlat_deg / 2
    ]

])


lon_edges = np.concatenate([

    [
        lon_values[0]
        -
        dlon_deg / 2
    ],

    (
        lon_values[:-1]
        +
        lon_values[1:]
    )
    /
    2,

    [
        lon_values[-1]
        +
        dlon_deg / 2
    ]

])


lat_edges_rad = np.deg2rad(
    lat_edges
)

lon_edges_rad = np.deg2rad(
    lon_edges
)


dlon_rad = np.diff(
    lon_edges_rad
)


area_m2_lat = (

    earth_radius_m ** 2

    *

    (
        np.sin(
            lat_edges_rad[
                1:
            ]
        )

        -

        np.sin(
            lat_edges_rad[
                :-1
            ]
        )
    )

)


cell_area_m2 = np.abs(

    np.outer(
        area_m2_lat,
        dlon_rad
    )

)


cell_area_cm2 = (
    cell_area_m2
    *
    10000
)


cell_area_cm2 = np.where(

    finn_inside_ne,

    cell_area_cm2,

    np.nan

)


finn_area_da = xr.DataArray(

    cell_area_cm2,

    coords={

        "lat":
            sample.lat,

        "lon":
            sample.lon

    },

    dims=[
        "lat",
        "lon"
    ]

)


sample.close()


# 31. FINN DAILY -> Gg -> SEASONAL TOTALS

finn_rows = []


for pollutant, variable in (
    FINN_POLLUTANTS.items()
):

    folder = os.path.join(

        FINN_FOLDER,

        (
            "PM25"
            if pollutant
            ==
            "PM2.5"

            else
            pollutant
        )

    )


    files = sorted(

        glob.glob(

            os.path.join(
                folder,
                "*.nc"
            )

        )

    )


    for file in files:

        ds = xr.open_dataset(
            file
        )


        ds = ds.sel(

            lat=ds.lat[
                (ds.lat >= min_lat)
                &
                (ds.lat <= max_lat)
            ],

            lon=ds.lon[
                (ds.lon >= min_lon)
                &
                (ds.lon <= max_lon)
            ]

        )


        flux = ds[
            variable
        ]


        # molecules cm^-2 s^-1
        # x cm2
        # x seconds/day
        # = molecules/day
        
        daily_molecules = (

            flux

            *

            finn_area_da

            *

            SECONDS_PER_DAY

        ).sum(

            dim=[
                "lat",
                "lon"
            ],

            skipna=True

        )


        # molecules -> mol -> grams -> Gg.
        
        daily_Gg = (

            daily_molecules

            /

            AVOGADRO

            *

            FINN_MOLAR_MASS[
                pollutant
            ]

            /

            1e9

        )


        dates = pd.to_datetime(

            daily_Gg[
                "time"
            ].values

        )


        daily_df = pd.DataFrame({

            "date":
                dates,

            "emission_Gg":
                daily_Gg.values

        })


        daily_df[
            "year"
        ] = daily_df[
            "date"
        ].dt.year


        daily_df[
            "month"
        ] = daily_df[
            "date"
        ].dt.month


        for year in sorted(

            daily_df[
                "year"
            ].unique()

        ):

            year_data = daily_df[
                daily_df[
                    "year"
                ]
                ==
                year
            ]


            annual_total = (
                year_data[
                    "emission_Gg"
                ]
                .sum()
            )


            spring_total = (
                year_data[
                    year_data[
                        "month"
                    ].isin(
                        [
                            3,
                            4,
                            5
                        ]
                    )
                ]
                [
                    "emission_Gg"
                ]
                .sum()
            )


            autumn_total = (
                year_data[
                    year_data[
                        "month"
                    ].isin(
                        [
                            9,
                            10,
                            11
                        ]
                    )
                ]
                [
                    "emission_Gg"
                ]
                .sum()
            )


            for season, value in [

                (
                    "Annual",
                    annual_total
                ),

                (
                    "Spring",
                    spring_total
                ),

                (
                    "Autumn",
                    autumn_total
                )

            ]:

                finn_rows.append({

                    "year":
                        int(
                            year
                        ),

                    "pollutant":
                        pollutant,

                    "season":
                        season,

                    "emission_Gg":
                        value

                })


        ds.close()


finn = pd.DataFrame(
    finn_rows
)


finn = (
    finn
    .sort_values(
        [
            "pollutant",
            "year",
            "season"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n--- FINN FINAL DATA CHECK ---"
)

print(
    "Years:",
    finn["year"].min(),
    "-",
    finn["year"].max()
)

print(
    finn.head()
)


# 32. GFAS FINAL PROCESSED DATA

gfas = pd.read_csv(
    GFAS_FILE
)


# Standardise PM2.5 spelling if necessary.

gfas[
    "pollutant"
] = gfas[
    "pollutant"
].replace({

    "PM25":
        "PM2.5",

    "PM2P5":
        "PM2.5"

})


# If your final GFAS file uses another Gg column name,
# rename it here.

if (
    "emission_Gg"
    not in
    gfas.columns
):

    possible_gfas_columns = [

        "integrated_emission_Gg",

        "emissions_Gg",

        "Gg"

    ]


    for column in (
        possible_gfas_columns
    ):

        if column in gfas.columns:

            gfas = gfas.rename(

                columns={
                    column:
                        "emission_Gg"
                }

            )

            break


if (
    "emission_Gg"
    not in
    gfas.columns
):

    raise ValueError(

        "GFAS file must contain final integrated emissions in Gg."

    )


# NB:
# The raw GFAS workflow used kg m^-2 s^-1,
# multiplied by grid-cell area and daily time,
# converted kg to Gg, and shifted valid_time back by one day.
#
# actual_date = valid_time - 1 day

# 33. CROSS-INVENTORY DATA

gfed_cross = gfed[
    [
        "year",
        "pollutant",
        "season",
        "emission_Gg"
    ]
].copy()


gfed_cross[
    "inventory"
] = "GFED"


finn_cross = finn[
    [
        "year",
        "pollutant",
        "season",
        "emission_Gg"
    ]
].copy()


finn_cross[
    "inventory"
] = "FINN"


gfas_cross = gfas[
    [
        "year",
        "pollutant",
        "season",
        "emission_Gg"
    ]
].copy()


gfas_cross[
    "inventory"
] = "GFAS"


cross_inventory = pd.concat(

    [
        gfed_cross,
        finn_cross,
        gfas_cross
    ],

    ignore_index=True

)



cross_inventory = cross_inventory[

    (
        cross_inventory[
            "season"
        ]
        ==
        "Autumn"
    )

    &

    (
        cross_inventory[
            "pollutant"
        ]
        .isin(
            COMMON_INVENTORY_POLLUTANTS
        )
    )

    &

    (
        cross_inventory[
            "year"
        ]
        .between(
            2012,
            2022
        )
    )

].copy()


# 34. CROSS-INVENTORY NORMALIZATION
#    Each inventory x pollutant: 2012-2017 mean = 100

inventory_baseline = (

    cross_inventory[
        cross_inventory[
            "year"
        ]
        .between(
            2012,
            2017
        )
    ]

    .groupby(
        [
            "inventory",
            "pollutant"
        ]
    )

    ["emission_Gg"]

    .mean()

    .rename(
        "baseline_2012_2017"
    )

)


cross_inventory = (
    cross_inventory.join(

        inventory_baseline,

        on=[
            "inventory",
            "pollutant"
        ]

    )
)


cross_inventory[
    "emission_index"
] = (

    cross_inventory[
        "emission_Gg"
    ]

    /

    cross_inventory[
        "baseline_2012_2017"
    ]

    *

    100

)

# 35. CROSS-INVENTORY PERIOD COMPARISON

cross_inventory_results = []


for inventory in [
    "GFED",
    "FINN",
    "GFAS"
]:

    for pollutant in (
        COMMON_INVENTORY_POLLUTANTS
    ):

        data = cross_inventory[

            (
                cross_inventory[
                    "inventory"
                ]
                ==
                inventory
            )

            &

            (
                cross_inventory[
                    "pollutant"
                ]
                ==
                pollutant
            )

        ]


        pre = data[
            data["year"]
            .between(
                2012,
                2017
            )
        ][
            "emission_Gg"
        ]


        post = data[
            data["year"]
            .between(
                2018,
                2022
            )
        ][
            "emission_Gg"
        ]


        pre_mean = (
            pre.mean()
        )


        post_mean = (
            post.mean()
        )


        change_percent = (

            (
                post_mean
                -
                pre_mean
            )

            /

            pre_mean

            *

            100

        )


        cross_inventory_results.append({

            "inventory":
                inventory,

            "pollutant":
                pollutant,

            "2012_2017_mean_Gg":
                pre_mean,

            "2018_2022_mean_Gg":
                post_mean,

            "change_percent":
                change_percent

        })


cross_inventory_results = pd.DataFrame(
    cross_inventory_results
)


print(
    "\n--- CROSS-INVENTORY AUTUMN CHANGE ---"
)

print(
    cross_inventory_results
    .round(2)
    .to_string(index=False)
)


# 36. KEY RESULTS FOR CHECKING AGAINST THE DISSERTATION

print(
    "\n"
    +
    "=" * 80
)

print(
    "KEY CHECKS AGAINST FINAL DISSERTATION"
)

print(
    "=" * 80
)


print(
    "\n1. Spring long-term trend p-values:"
)

print(
    trend_results[
        trend_results[
            "season"
        ]
        ==
        "Spring"
    ][
        [
            "pollutant",
            "p_value"
        ]
    ]
    .round(4)
    .to_string(index=False)
)


print(
    "\n2. Autumn 2018 breakpoint:"
)

print(
    breakpoint_results[
        (
            breakpoint_results[
                "season"
            ]
            ==
            "Autumn"
        )
        &
        (
            breakpoint_results[
                "breakpoint"
            ]
            ==
            2018
        )
    ][
        [
            "pollutant",
            "level_change_percent",
            "level_change_p"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


print(
    "\n3. Restricted 2012-2022 breakpoint:"
)

print(
    restricted_results[
        [
            "pollutant",
            "level_change_percent",
            "level_change_p"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


print(
    "\n4. IQR years:"
)

print(
    extreme_years
)


print(
    "\n5. Agricultural FDR results:"
)

print(
    agricultural_tests
    .sort_values(
        "p_nominal"
    )
    .round(3)
    .to_string(index=False)
)


print(
    "\n6. Soybean sensitivity excluding Jilin:"
)

print(
    "Production change:",
    round(
        soybean_change_no_jilin,
        2
    ),
    "%"
)

print(
    "rho =",
    round(
        rho_soy_level,
        3
    ),
    ", p =",
    round(
        p_soy_level,
        3
    )
)


print(
    "\n7. Meteorological adjustment:"
)

print(
    met_regression_results
    .round(3)
    .to_string(index=False)
)


print(
    "\n8. Cross-inventory comparison:"
)

print(
    cross_inventory_results
    .round(2)
    .to_string(index=False)
)