import numpy as np
import pandas as pd
    # Feature engineering: precip_depth_1
    # Convert -1 and NaN precipitation to 0
    # Create trace rain indicator
    # Create NaN indicator
def precip_depth_1_hr_FE(df, m):
    df['precip_depth_1_hr_nan'] = df['precip_depth_1_hr'].isna()
    if m:
        df.loc[df['precip_depth_1_hr'].isna(), 'precip_depth_1_hr'] = m
    else:
        m = df['precip_depth_1_hr'].median()
        df.loc[df['precip_depth_1_hr'].isna(), 'precip_depth_1_hr'] = m

    df['precip_depth_1_hr_isTrace'] = (df['precip_depth_1_hr'] == -1)
    df.loc[df['precip_depth_1_hr'] == -1, 'precip_depth_1_hr'] = 0
    return df, m

# Feature engineering: wind_direction
# Replace nan with median wind_directin angle
# Create nan indicator
# Convert to sine and cosine features

def wind_direction_FE(df, m=None):
    df['wind_direction_nan'] = df['wind_direction'].isna()

    if m:
        df.loc[df['wind_direction'].isna(), 'wind_direction'] = m
    else:
        m = df['wind_direction'].median()
        df.loc[df['wind_direction'].isna(), 'wind_direction'] = m

    df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction']))
    df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction']))
    return df, m

