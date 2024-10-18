#!/usr/bin/env python
# coding: utf-8

# # Statistical-Analysis-of-the-2024-World-Happiness-Index-by-Regions

# In[91]:


import pandas as pd

# Load data set
data=pd.read_csv('2024.csv')


# In[92]:


data.columns


# In[93]:


data.shape


# In[94]:


data.dtypes


# In[95]:


data.info()


# In[96]:


data.isnull().sum()


# In[97]:


data.describe()


# In[98]:


# Summary statistics
print("\nSummary Statistics:")
print(data.describe())


# In[99]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data set
df = pd.read_csv('2024.csv')

# Use correct column names in DataFrame
df.columns = [
    'Country name', 'Ladder score', 'upperwhisker', 'lowerwhisker',
    'Log GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Dystopia + residual'
]

# Sort DataFrame by Ladder score
df_sorted = df.sort_values(by='Ladder score', ascending=False)

#Discover the 25 happiest countries
happiest_countries = df_sorted.head(25)

# List the 25 unhappiest countries
unhappiest_countries = df_sorted.tail(25)

# Show results
print("25 Happiest Countries:")
print(happiest_countries[['Country name', 'Ladder score']])

print("\n25 Unhappiest Countries:")
print(unhappiest_countries[['Country name', 'Ladder score']])

#Graphics
fig, axs = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)

# Chart for the happiest countries
sns.barplot(x='Ladder score', y='Country name', data=happiest_countries, ax=axs[0], palette='viridis')
axs[0].set_title('25 Happiest Countries')
axs[0].set_xlabel('Happiness Score')
axs[0].set_ylabel('Country')

# Write scores on bars
for index, value in enumerate(happiest_countries['Ladder score']):
    axs[0].text(value, index, f'{value:.2f}', color='black', ha="left", va="center")

# Chart for the unhappiest countries
sns.barplot(x='Ladder score', y='Country name', data=unhappiest_countries, ax=axs[1], palette='magma')
axs[1].set_title('25 Unhappiest Countries')
axs[1].set_xlabel('Happiness Score')
axs[1].set_ylabel('Country')

# Write scores on bars
for index, value in enumerate(unhappiest_countries['Ladder score']):
    axs[1].text(value, index, f'{value:.2f}', color='black', ha="left", va="center")

# Showing graphics
plt.show()


# In[100]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="viridis", fmt=".4f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# In[101]:


# Creating correlation matrix
correlation_matrix


# In[102]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Eksik değerlerin olup olmadığını kontrol edin
print(data.isnull().sum())

# Eksik değerleri içeren satırları kaldır
data_cleaned = data.dropna()

# 3. Regresyon Analizi
X = data_cleaned[['Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption']]
y = data_cleaned['Ladder score']

# Sabit terimi ekleyelim
X = sm.add_constant(X)

# Modeli oluştur ve uygula
model = sm.OLS(y, X).fit()
results = model.summary()

print(results)


# In[103]:


import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
data = pd.read_csv("2024.csv")

# Eksik değerleri içeren satırları kaldır
data_cleaned = data.dropna()

# Pairplot
sns.pairplot(data_cleaned, vars=['Ladder score', 'Explained by: Log GDP per capita', 'Explained by: Social support', 
                                 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 
                                 'Explained by: Generosity', 'Explained by: Perceptions of corruption'])
plt.show()


# # Bölgeler Açısından ANOVA

# In[104]:


# Define the regions and their countries
regions = {
    'Europe': ['Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 
               'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 
               'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kazakhstan', 'Kosovo', 'Latvia', 'Liechtenstein', 
               'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 
               'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 
               'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom', 'Vatican City'],
    
    'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 
             'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 
             'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman', 
             'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 
             'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 
             'Yemen'],
    
    'America': ['Antigua and Barbuda', 'Argentina', 'Bahamas', 'Barbados', 'Belize', 'Bolivia', 'Brazil', 'Canada', 'Chile', 
                'Colombia', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Grenada', 
                'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 
                'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Suriname', 'Trinidad and Tobago', 
                'United States', 'Uruguay', 'Venezuela'],
    
    'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 
               'Chad', 'Comoros', 'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 
               'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 
               'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 
               'Republic of the Congo', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 
               'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe'],
    
    'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 
                'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu']
}

# Function to determine the region of a country
def get_region(country):
    for region, countries in regions.items():
        if country in countries:
            return region
    return 'Other'

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


# Add the Region column to the dataframe
df['Region'] = df['Country name'].apply(get_region)

# Group different regions and their happiness scores
regions = df['Region'].unique()
region_scores = [df[df['Region'] == region]['Ladder score'] for region in regions]

# ANOVA test
f_stat, p_value = f_oneway(*region_scores)
print(f"Hypothesis 3: ANOVA test of the difference between happiness scores in different regions: F-statistic: {f_stat}, p-value: {p_value}")

# Visualization
plt.figure(figsize=(12, 8))
sns.boxplot(x='Region', y='Ladder score', data=df)
plt.title('Happiness Scores in Different Regions')
plt.xlabel('Regions')
plt.ylabel('Ladder Skoru')
plt.show()


# # Tukey Test

# Since there is a statistically significant difference between the groups, post-hoc tests can be applied to determine which groups differ. One of these, Tukey's HSD (Honestly Significant Difference), is a widely used method to compare differences between groups. We can use the statsmodels library to apply this test.

# In[105]:


import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey's HSD test
tukey_results = pairwise_tukeyhsd(df['Ladder score'], df['Region'])

# View results
print(tukey_results)


# In[ ]:





# In[ ]:





# In[ ]:




