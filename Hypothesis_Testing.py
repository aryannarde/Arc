import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import pandas as pd


data = pd.read_csv('sales.csv')
np.random.seed(42)
data['Region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(data))


region_groups = data.groupby('Region')['TotalAmount']


test_results = {}
regions = data['Region'].unique()
for i, region1 in enumerate(regions):
    for region2 in regions[i + 1:]:
        group1 = data[data['Region'] == region1]['TotalAmount']
        group2 = data[data['Region'] == region2]['TotalAmount']
        t_stat, p_value = ttest_ind(group1, group2)
        test_results[f'{region1} vs {region2}'] = (t_stat, p_value)


bins = [0, 5, 10, 15, np.inf]
labels = ['1-5', '6-10', '11-15', '16+']
data['QuantityCategory'] = pd.cut(data['Quantity'], bins=bins, labels=labels)


contingency_table = pd.crosstab(data['Region'], data['QuantityCategory'])


chi2_stat, p, dof, expected = chi2_contingency(contingency_table)


def display_results():
    print("T-Test Results for Sales Performance Between Regions:")
    for comparison, result in test_results.items():
        print(f"{comparison}: t-stat = {result[0]:.4f}, p-value = {result[1]:.4f}")

    print("\nChi-Square Test for Independence Between Region and Quantity Category:")
    print(f"Chi2 Stat = {chi2_stat:.4f}, p-value = {p:.4f}")

display_results()
