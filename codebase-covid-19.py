import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime

# Replace with your Google Drive file ID which has public view access
file_id = "1r054TYsGmBbIob_rM_KCjjbORV58ayCr"

# Construct the download URL
url = f"https://drive.google.com/uc?export=download&id={file_id}"

try:
  # Read CSV directly into pandas
  df_covid = pd.read_csv(url)

  # Show first few rows
  print(df_covid.head())
except Exception as e:
    print(f"Error loading data: {e}")

# Subsetting the dataset for analysis
subset_cols = [
    "Date_reported",
    "Country",
    "WHO_region",
    "New_cases",
    "New_deaths",
    "Cumulative_cases",
    "Cumulative_deaths"
]

df_covid_subset = df_covid[subset_cols].copy()

 #DATA TRIMMING
  # Assuming your dataframe is named df and Date_reported is already in datetime format
df_covid_subset['Date_reported'] = pd.to_datetime(df_covid_subset['Date_reported'])

# Define the date range
start_date = "2020-03-01"
end_date = "2023-08-31"

# Filter the rows
df_covid_trimmed = df_covid_subset[(df_covid_subset['Date_reported'] >= start_date) & (df_covid_subset['Date_reported'] <= end_date)]

# Check subset
print("Subset shape:", df_covid_trimmed.shape)
print(df_covid_trimmed.head())
print(df_covid_trimmed.tail())
# 1. Line plots for daily cases in the top 5 affected countries

# Find top 5 affected countries by cumulative cases
top5_countries = (
    df_covid_trimmed.groupby("Country")["Cumulative_cases"]
    .max()
    .sort_values(ascending=False)
    .head(5)
    .index
)

# Filter data
df_top5 = df_covid_trimmed[df_covid_trimmed["Country"].isin(top5_countries)]

# Plotting the data based on daily new cases
plt.figure(figsize=(12,6))
for country in top5_countries:
    country_data = df_top5[df_top5["Country"] == country]
    plt.plot(country_data["Date_reported"], country_data["New_cases"], label=country)

plt.xlabel("Date")
plt.ylabel("Daily New Cases")
plt.title("Daily New Cases - Top 5 Affected Countries (Matplotlib)")
plt.legend()
plt.show()

bottom5_countries = (
    df_covid_trimmed.groupby("Country")["Cumulative_cases"]
    .max()
    .sort_values(ascending=True)   # ascending instead of descending
    .head(5)
    .index
)

# Filter data
df_bottom5 = df_covid_trimmed[df_covid_trimmed["Country"].isin(bottom5_countries)]

# Plotting the data based on daily new cases
plt.figure(figsize=(12,6))
for country in bottom5_countries:
    country_data = df_bottom5[df_bottom5["Country"] == country]
    plt.plot(country_data["Date_reported"], country_data["New_cases"], label=country)

plt.xlabel("Date")
plt.ylabel("Daily New Cases")
plt.title("Daily New Cases - 5 Least Affected Countries (Matplotlib)")
plt.legend()
plt.show()

# Grouping by date and sum new cases globally
df_global = (
    df_covid_trimmed.groupby("Date_reported")["New_cases"]
    .sum()
    .reset_index()
)

# Plotting the global daily new cases
plt.figure(figsize=(12,6))
plt.plot(df_global["Date_reported"], df_global["New_cases"], color="red", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Global Daily New Cases")
plt.title("Global Daily New COVID-19 Cases (Mountain Shape)")
plt.fill_between(df_global["Date_reported"], df_global["New_cases"], color="red", alpha=0.3)  # shaded area for mountain effect
plt.show()

# Cases vs Deaths Over Time (Quarterly)


# Creating quarterly data for cleaner visualization
df_copy = df_covid_trimmed.copy()
df_copy['Quarter'] = df_copy['Date_reported'].dt.to_period('Q')

# Aggregating by quarter
quarterly_data = df_copy.groupby('Quarter').agg({'New_cases': 'sum', 'New_deaths': 'sum'}).reset_index()

# Converting period to string for plotting
quarterly_data['Quarter_str'] = quarterly_data['Quarter'].astype(str)

# Creating the stacked bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Creating bars
width = 0.6
cases_bars = ax.bar(quarterly_data['Quarter_str'], quarterly_data['New_cases'],
                       width, label='New Cases', color='lightcoral', alpha=0.8)
deaths_bars = ax.bar(quarterly_data['Quarter_str'], quarterly_data['New_deaths'],
                        width, bottom=quarterly_data['New_cases'],
                        label='New Deaths', color='darkred', alpha=0.9)

# Customizing the plot
ax.set_title('The Timeline of Tragedy: COVID-19 Cases vs Deaths by Quarter',
                fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Quarter', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of People Affected', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Rotating x-axis labels
plt.xticks(rotation=45, ha='right')

# Adding value annotations on the bars
for i, (cases, deaths) in enumerate(zip(quarterly_data['New_cases'], quarterly_data['New_deaths'])):
  # Annotating total at the top
  total = cases + deaths
  ax.annotate(f'{total:,.0f}',
                xy=(i, total),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom',
                fontweight='bold', fontsize=10)

  plt.tight_layout()
  plt.show()

  # Printing insights
  max_quarter = quarterly_data.loc[quarterly_data['New_cases'].idxmax(), 'Quarter_str']
  max_cases = quarterly_data['New_cases'].max()
import numpy as np

# Preparing the x positions
x = np.arange(len(quarterly_data))  # positions for quarters
width = 0.35  # width of each bar

fig, ax = plt.subplots(figsize=(14, 8))

# Creating side-by-side bars
cases_bars = ax.bar(x - width/2, quarterly_data['New_cases'], width,
                    label='New Cases', color='lightcoral', alpha=0.8)
deaths_bars = ax.bar(x + width/2, quarterly_data['New_deaths'], width,
                     label='New Deaths', color='darkred', alpha=0.9)

# Customizing the plot
ax.set_title('The Timeline of Tragedy: COVID-19 Cases vs Deaths by Quarter',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Quarter', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of People Affected', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(quarterly_data['Quarter_str'], rotation=45, ha='right')
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Adding value annotations on the bars
for bars in [cases_bars, deaths_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()

# Printing insights
max_quarter = quarterly_data.loc[quarterly_data['New_cases'].idxmax(), 'Quarter_str']
max_cases = quarterly_data['New_cases'].max()
print(f"Quarter with highest cases: {max_quarter} ({max_cases:,.0f} cases)")


# Finding top 10 countries by cumulative deaths
top10_deaths = (
    df_covid_trimmed.groupby("Country")["Cumulative_deaths"]
    .max()
    .sort_values(ascending=False)
    .head(10)
)

# Pie chart
plt.figure(figsize=(10, 8))
plt.pie(
    top10_deaths,
    labels=top10_deaths.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Reds(np.linspace(0.3, 0.9, 10)),  # shades of red
    wedgeprops={'edgecolor': 'black'}
)

plt.title("Top 10 Countries by Cumulative COVID-19 Deaths", fontsize=16, fontweight='bold')
plt.axis('equal')  # equal aspect ratio for perfect circle
plt.show()

#Create a powerful heatmap showing intensity across regions and time
# Create monthly data for cleaner visualization
df_monthly = df_covid_trimmed.copy()
df_monthly['Month_Year'] = df_monthly['Date_reported'].dt.to_period('M')

# Aggregate by WHO region and month
heatmap_data = df_monthly.groupby(['WHO_region', 'Month_Year'])['New_cases'].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index='WHO_region', columns='Month_Year', values='New_cases')

# Fill missing values with 0
heatmap_pivot = heatmap_pivot.fillna(0)

# Create the heatmap
plt.figure(figsize=(16, 8))

# Use a dramatic color scheme
sns.heatmap(heatmap_pivot,
            cmap='Reds',
            cbar_kws={'label': 'Monthly New Cases'},
            linewidths=0.5,
            linecolor='white')

plt.title('Global COVID-19 Intensity Map: When and Where the World Burned',
    fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Month-Year', fontsize=12)
plt.ylabel('WHO Region', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Add colorbar label
cbar = plt.gca().collections[0].colorbar
cbar.ax.yaxis.label.set_size(12)

plt.tight_layout()
plt.show()

# Creating quarterly data
df_quarterly = df_covid_trimmed.copy()
df_quarterly['Quarter'] = df_quarterly['Date_reported'].dt.to_period('Q')

# Aggregating by WHO region and quarter (new deaths)
heatmap_deaths = df_quarterly.groupby(['WHO_region', 'Quarter'])['New_deaths'].sum().reset_index()

# Pivoting for heatmap
heatmap_pivot_deaths = heatmap_deaths.pivot(index='WHO_region', columns='Quarter', values='New_deaths')

# Fill missing with 0
heatmap_pivot_deaths = heatmap_pivot_deaths.fillna(0)

# Plot heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(
    heatmap_pivot_deaths,
    cmap="Blues",   # different scheme for deaths
    cbar_kws={'label': 'Quarterly New Deaths'},
    linewidths=0.5,
    linecolor='white'
)

plt.title("Global COVID-19 Death Intensity Map: Quarterly Regional Trends",
          fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Quarter", fontsize=12)
plt.ylabel("WHO Region", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Customizing colorbar
cbar = plt.gca().collections[0].colorbar
cbar.ax.yaxis.label.set_size(12)

plt.tight_layout()
plt.show()

# Finding top 10 countries by cumulative cases
top10_countries = (
    df_covid_trimmed.groupby("Country")["Cumulative_cases"]
    .max()
    .sort_values(ascending=False)
    .head(10)
    .index
)

# Filtering only top 10 countries
df_top10 = df_covid_trimmed[df_covid_trimmed["Country"].isin(top10_countries)].copy()

# Creating month-year column
df_top10['Month_Year'] = df_top10['Date_reported'].dt.to_period('M')

# Aggregating monthly new cases by country
heatmap_cases = df_top10.groupby(['Country', 'Month_Year'])['New_cases'].sum().reset_index()

# Pivoting for heatmap
heatmap_pivot_cases = heatmap_cases.pivot(index='Country', columns='Month_Year', values='New_cases')

# Filling missing with 0
heatmap_pivot_cases = heatmap_pivot_cases.fillna(0)

# Plottng heatmap
plt.figure(figsize=(18, 8))
sns.heatmap(
    heatmap_pivot_cases,
    cmap="Reds",  # intense red shades for cases
    cbar_kws={'label': 'Monthly New Cases'},
    linewidths=0.5,
    linecolor='white'
)

plt.title("COVID-19 Monthly New Cases Intensity - Top 10 Countries",
          fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Month-Year", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjusting colorbar font size
cbar = plt.gca().collections[0].colorbar
cbar.ax.yaxis.label.set_size(12)

plt.tight_layout()
plt.show()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



# ---- 1. Line Chart: Global New Cases Over Time ----
fig_cases = px.line(
    df_covid_trimmed.groupby('Date_reported', as_index=False).sum(),
    x="Date_reported",
    y="New_cases",
    title="Global New COVID-19 Cases Over Time"
)

# ---- 2. Line Chart: Global New Deaths Over Time ----
fig_deaths = px.line(
    df_covid_trimmed.groupby('Date_reported', as_index=False).sum(),
    x="Date_reported",
    y="New_deaths",
    title="Global New COVID-19 Deaths Over Time",
    color_discrete_sequence=["red"]
)

# ---- 3. Stacked Bar Chart: New Cases vs New Deaths grouped by WHO Region ----
region_grouped = df_covid_trimmed.groupby(['Date_reported', 'WHO_region'], as_index=False)[['New_cases', 'New_deaths']].sum()

fig_stacked = go.Figure()
fig_stacked.add_trace(go.Bar(
    x=region_grouped["Date_reported"],
    y=region_grouped["New_cases"],
    name="New Cases",
    marker_color="blue"
))
fig_stacked.add_trace(go.Bar(
    x=region_grouped["Date_reported"],
    y=region_grouped["New_deaths"],
    name="New Deaths",
    marker_color="red"
))
fig_stacked.update_layout(
    barmode="stack",
    title="New Cases vs New Deaths (Stacked) by WHO Region",
    xaxis_title="Date",
    yaxis_title="Count",
)

# ---- 4. Choropleth Map: Total Cases by Country ----
country_grouped = df_covid_trimmed.groupby('Country', as_index=False)['Cumulative_cases'].max()

fig_map = px.choropleth(
    country_grouped,
    locations="Country",
    locationmode="country names",
    color="Cumulative_cases",
    hover_name="Country",
    color_continuous_scale="Viridis",
    title="Global Distribution of Total COVID-19 Cases"
)

# ---- Display all interactive charts ----
fig_cases.show()
fig_deaths.show()
fig_stacked.show()
fig_map.show()
