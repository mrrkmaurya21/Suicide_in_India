#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.ticker as ticker


# In[2]:


# Load the CSV file
suicides_data=pd.read_csv("Suicides in India 2001-2012.csv")
suicides_data.head()


# In[3]:


suicides_data.shape


# In[4]:


# returns the number of unique values for each column
suicides_data.nunique()


# In[5]:


suicides_data['State'].value_counts()


# In[6]:


#Remove categories such as Total (All India), Total (Uts) and otal (States)

suicides_data=suicides_data[(suicides_data['State']!='Total (All India)') & (suicides_data['State']!='Total (States)') & (suicides_data['State']!='Total (Uts)')]


# In[7]:


suicides_data = suicides_data[suicides_data['Age_group'] != '0-100+']


# In[29]:


# returns the number of unique values for each column
suicides_data.nunique()


# In[8]:


# Display basic information about the dataset
print("\nBasic information about the dataset:")
suicides_data.info()


# In[9]:


# Create a DataFrame with year-wise suicides based on gender
#year_total_data = suicides_data.groupby(['Year']).sum()['Total'].reset_index()
year_total_data = suicides_data.groupby(['Year']).sum()['Total'].reset_index()

# Display the resulting DataFrame
year_total_data


# In[10]:


# Line plot for the distribution of suicides across years
plt.figure(figsize=(10, 4))
sns.lineplot(x='Year', y='Total', data=suicides_data, estimator='sum', errorbar=None)
plt.title('Distribution of Suicides Across Years')
plt.xlabel('Year')
plt.ylabel('Total Suicides')
plt.show()


# In[11]:


# Create a DataFrame with year-wise suicides based on gender
year_gender_data = suicides_data.groupby(['Year', 'Gender']).sum()['Total'].unstack()

# Add a 'Total' column
year_gender_data['Total'] = year_gender_data['Female'] + year_gender_data['Male']

# Display the resulting DataFrame
print(year_gender_data)


# In[12]:


# Create a DataFrame with year-wise suicides based on gender
year_gender_data = suicides_data.groupby(['Year', 'Gender']).sum(numeric_only=True)['Total'].unstack()

# Plotting
plt.figure(figsize=(10, 4))
sns.lineplot(data=year_gender_data, markers=True, palette={'Male': 'blue', 'Female': 'pink'})
plt.title('Year-wise Suicides by Gender')
plt.xlabel('Year')
plt.ylabel('Total Suicides')
plt.legend(title='Gender', loc='upper right')
plt.show()


# In[13]:


# Aggregate total suicides for each age group and year
age_year_totals = suicides_data.groupby(['Age_group', 'Year']).sum(numeric_only=True)['Total'].reset_index()

# Pivot the DataFrame to have 'Year' as columns
age_year_totals_pivot = age_year_totals.pivot(index='Age_group', columns='Year', values='Total')

# Display the resulting DataFrame
age_year_totals_pivot



# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Line plot for the distribution of suicides across age groups
plt.figure(figsize=(10, 4))
ax = sns.lineplot(x='Age_group', y='Total', hue='Year', data=suicides_data, estimator='sum', errorbar=None, palette='viridis')
plt.title('Distribution of Suicides Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Total Suicides')

# Format y-axis ticks with commas as thousand separators
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# Add a legend
plt.legend(title='Year', loc='upper right')

plt.show()


# In[15]:


# Create a DataFrame with year-wise suicides based on gender
year_type_total_data = suicides_data.groupby(['Year', 'Type_code']).sum(numeric_only=True)['Total'].unstack()

# Add a 'Total' column
year_type_total_data['Total'] = year_type_total_data.sum(axis=1)

year_type_total_data


# In[16]:


# Line plot for year-wise suicides based on gender and type code
plt.figure(figsize=(10, 4))
ax = sns.lineplot(data=year_type_total_data.iloc[:, :-1], marker='o')

# Format y-axis ticks with commas as thousand separators
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# Add titles and labels
plt.title('Year-wise Suicides Based on Gender and Type Code')
plt.xlabel('Year')
plt.ylabel('Total Suicides')

# Add a legend
plt.legend(title='Type Code', loc='upper right')

plt.show()


# In[17]:


# Create a DataFrame with year-wise suicides based on gender and age group
year_age_gender_data = suicides_data.groupby(['Year', 'Age_group', 'Gender']).sum(numeric_only=True)['Total'].unstack(level=[1, 2])

# Add a 'Total' column
year_age_gender_data['Total'] = year_age_gender_data.sum(axis=1)

# Display the resulting DataFrame
year_age_gender_data

# Display the resulting DataFrame
year_age_gender_data


# In[18]:


# Create a DataFrame with year-wise suicides based on gender
year_gender_data = suicides_data.groupby(['State', 'Year']).sum(numeric_only=True)['Total'].unstack()
# Display the resulting DataFrame
#year_gender_data


# In[19]:


state_suicides = suicides_data.groupby('State')['Total'].sum().reset_index()

# Calculate the percentage of total suicides for each state
state_suicides_percentage = (state_suicides['Total'] / state_suicides['Total'].sum()) * 100

# Sort the states by total suicides in descending order
sorted_states = state_suicides.sort_values(by='Total', ascending=False)

# Extract top 5 and bottom 5 states
top_5_states = sorted_states.head(5)
bottom_5_states = sorted_states.tail(5)

# Display top 5 states with percentage
print("Top 5 Suicide States:")
for rank, (state, suicides) in enumerate(top_5_states[['State', 'Total']].itertuples(index=False, name=None), start=1):
    percentage = (suicides / state_suicides['Total'].sum()) * 100
    print(f"{rank}. {state}: {suicides} ({percentage:.3f}%)")

# Display bottom 5 states with percentage
print("\nBottom 5 Suicide States:")
for rank, (state, suicides) in enumerate(bottom_5_states[['State', 'Total']].itertuples(index=False, name=None), start=1):
    percentage = (suicides / state_suicides['Total'].sum()) * 100
    print(f"{rank}. {state}: {suicides} ({percentage:.3f}%)")


# In[20]:


# Assuming 'State' column contains state names
state_suicides = suicides_data.groupby('State').sum()['Total']
total_suicides = state_suicides.sum()

# Calculate the percentage of total suicides for each state
state_suicides_percentage = (state_suicides / total_suicides) * 100

# Sort the states by total suicides in descending order (for top 10)
sorted_states = state_suicides_percentage.sort_values(ascending=False)

# Extract top 10 states
top_10_states = sorted_states.head(10)

# Plot the bar chart
plt.figure(figsize=(10, 5))
state_fig = top_10_states.plot(kind='bar', color='red', width=0.5)

# Add labels and annotations
state_fig.bar_label(state_fig.containers[0], fontsize=10, fmt='%.2f%%')
state_fig.set_title('Percentage of Suicides Across States (Top 10)')
state_fig.set_xlabel('State')
state_fig.set_ylabel('Percentage of Total Suicides')

# Display the plot
plt.show()


# In[21]:


# Assuming 'State' column contains state names
state_suicides = suicides_data.groupby('State').sum()['Total']
total_suicides = state_suicides.sum()

# Calculate the percentage of total suicides for each state
state_suicides_percentage = (state_suicides / total_suicides) * 100

# Sort the states by total suicides in ascending order (for bottom 10)
sorted_states = state_suicides_percentage.sort_values()

# Extract bottom 10 states
bottom_10_states = sorted_states.head(10)

# Plot the bar chart
plt.figure(figsize=(10, 4))
state_fig = bottom_10_states.plot(kind='bar', color='green', width=0.5)

# Add labels and annotations
state_fig.bar_label(state_fig.containers[0], fontsize=10, fmt='%.3f%%')
state_fig.set_title('Percentage of Suicides Across States (Bottom 10)')
state_fig.set_xlabel('State')
state_fig.set_ylabel('Percentage of Total Suicides')

# Display the plot
plt.show()


# In[23]:


# Create a DataFrame with year-wise suicides based on type
year_type_data = suicides_data.groupby(['Year', 'Type']).sum(numeric_only=True)['Total'].unstack()

# Calculate the total suicides for each year
year_type_data['Total'] = year_type_data.sum(axis=1)

# Get the top 5 suicide types for each year
top_types_per_year = year_type_data.iloc[:, :-1].apply(lambda row: row.nlargest(5), axis=1)

# Display top 5 suicide types in decreasing order with percentage and year-wise
for year, top_types in top_types_per_year.iterrows():
    total_suicides = year_type_data.loc[year, 'Total']
    print(f'Top 5 Suicide Types in {year} (Percentage of Total Suicides):')

    # Drop NaN values and sort in descending order
    top_types = top_types.dropna().sort_values(ascending=False)

    for type_name, suicides in top_types.items():
        percentage = (suicides / total_suicides) * 100
        print(f'{type_name}: {suicides:.0f} ({percentage:.2f}%)')

    print('\n' + '-'*50 + '\n')  # Separate each year's information


# # Conducting statistical tests to check the assumptions of normality and homogeneity of variance for two groups (male and female data).

# Normality: The data within each group should be approximately normally distributed. You can check this assumption visually using histograms or quantitatively using normality tests (e.g., Shapiro-Wilk test).
# 
# Homogeneity of Variance: The variances of the different groups should be approximately equal. You can check this assumption using statistical tests such as Levene's test.
# 
# Independence: Observations within each group should be independent of each other.

# In[ ]:





# In[52]:


from scipy.stats import shapiro, levene

male_data = suicides_data[suicides_data['Gender'] == 'Male']['Total']
female_data = suicides_data[suicides_data['Gender'] == 'Female']['Total']

# Assumption 1: Normality
_, male_p_value = shapiro(male_data)
_, female_p_value = shapiro(female_data)

print(f"Shapiro-Wilk Test - Normality Assumption:")
print(f"Male Data - P-Value: {male_p_value}")
print(f"Female Data - P-Value: {female_p_value}")

# Assumption 2: Homogeneity of Variances
_, levene_p_value = levene(male_data, female_data)

print(f"\nLevene's Test - Homogeneity of Variances:")
print(f"P-Value: {levene_p_value}")
print()
# Assuming a significance level of 0.05
if p_value_shapiro_male > 0.05 and p_value_shapiro_female > 0.05:
    print("Normality assumption is met for both groups.")
else:
    print("Normality assumption is violated for one or both groups.")

if p_value_levene > 0.05:
    print("Homogeneity of variance assumption is met.")
else:
    print("Homogeneity of variance assumption is violated.")


# # Mann-Whitney U Test

# In[41]:


from scipy.stats import mannwhitneyu

# Assuming you have a DataFrame named 'suicides_data' with columns 'Gender' and 'Total'
male_data = suicides_data[suicides_data['Gender'] == 'Male']['Total']
female_data = suicides_data[suicides_data['Gender'] == 'Female']['Total']

# Perform Mann-Whitney U test
statistic, p_value = mannwhitneyu(male_data, female_data, alternative='two-sided')

print(f"\nMann-Whitney U Test - Total Suicides between Male and Female:")
print(f"U Statistic: {statistic}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference in the total number of suicides between Male and Female.")
else:
    print("There is no significant difference in the total number of suicides between Male and Female.")


# In[43]:


# Shapiro-Wilk Test for Normality Assumption
shapiro_results = {}

for age_group, data in anova_results.items():
    shapiro_statistic, shapiro_p_value = shapiro(data)
    shapiro_results[age_group] = shapiro_p_value

print("\nShapiro-Wilk Test - Normality Assumption:")
for age_group, p_value in shapiro_results.items():
    print(f"{age_group} - P-Value: {p_value}")

# Levene's Test for Homogeneity of Variances
levene_statistic, levene_p_value = levene(*anova_results.values())

print("\nLevene's Test - Homogeneity of Variances:")
print(f"P-Value: {levene_p_value}")


# # Kruskal-Wallis Test

# In[44]:


from scipy.stats import kruskal

# Create a dictionary to store data for each age group
age_group_data = {age_group: suicides_data[suicides_data['Age_group'] == age_group]['Total'] for age_group in age_groups}

# Perform Kruskal-Wallis test
h_statistic, p_value = kruskal(*age_group_data.values())

print("\nKruskal-Wallis Test - Suicide Rates Among Age Groups:")
print(f"H-Statistic: {h_statistic}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference in suicide rates among the age groups.")
else:
    print("There is no significant difference in suicide rates among the age groups.")


# In[ ]:




