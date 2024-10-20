# Empirical exercise for "Income Risk, Global Shocks, and International Capital Flows"
# October 2024
# Estimation functions

def extract_country_and_MP_shocks(df, country, MP_shock_vars):
    '''Extracts the country and MP shocks from the full dataset
    Input is a dataframe with the full dataset and the output is a dataframe with only the country and MP shocks
    
    Args:
        df: Pandas dataframe with the full dataset
        country: String with the country code
        MP_shock_vars: List with the MP shock variables
    
    Returns:
        df_country: Pandas dataframe with the country and MP shocks
    '''
    df_country = df.filter(like=country)
    df_country = df_country.join(df[MP_shock_vars], how='outer')
    return df_country

MP_shock_vars = [
    "BS_MPS",
    "BS_MPS_ORTH",
    "Romer & Romer (2004)",
    "Miranda-Agrippino (2014) quarterly",
    "jf_dY_prod",
    "jf_dY_inc",
    "jf_dY",
    "jf_dhours",
    "jf_dLP",
    "jf_dk",
    "jf_dLQ_BLS_interpolated",
    "jf_dLQ_Aaronson_Sullivan",
    "jf_dLQ",
    "jf_alpha",
    "jf_dtfp",
    "jf_dutil",
    "jf_dtfp_util",
    "jf_relativePrice",
    "jf_invShare",
    "jf_dtfp_I",
    "jf_dtfp_C",
    "jf_du_invest",
    "jf_du_consumption",
    "jf_dtfp_I_util",
    "jf_dtfp_C_util",
    "BRW_shocks",
    "jk_pc1_hf",
    "jk_SP500_hf",
    "jk_MP_pm",
    "jk_CBI_pm",
    "jk_MP_median",
    "jk_CBI_median"
]