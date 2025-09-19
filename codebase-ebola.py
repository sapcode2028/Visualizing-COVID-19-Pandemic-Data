import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime

file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"

url = f"https://drive.google.com/uc?export=download&id={file_id}"

try:
    df_ebola = pd.read_csv(url)

  print(df_ebola.head())
except Exception as e:
    print(f"Error loading data: {e}")

import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Reading CSV
df_ebola = pd.read_csv(url)

# ---- Preprocessing ----
# Ensure proper column names (adjust if your CSV headers differ)
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]

# Converting Date to datetime
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Sorting by country + date to calculate daily new cases
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Computing daily new cases from cumulative cases
df_ebola["New_cases"] = df_ebola.groupby("Country")["Cumulative_cases"].diff().fillna(df_ebola["Cumulative_cases"])

# Finding Top 5 affected countries
top5_countries = (
    df_ebola.groupby("Country")["Cumulative_cases"]
    .max()
    .sort_values(ascending=False)
    .head(5)
    .index
)

df_top5 = df_ebola[df_ebola["Country"].isin(top5_countries)]

# Plotting daily new cases
plt.figure(figsize=(12,6))
for country in top5_countries:
    country_data = df_top5[df_top5["Country"] == country]
    plt.plot(country_data["Date"], country_data["New_cases"], label=country)

plt.xlabel("Date")
plt.ylabel("Daily New Cases")
plt.title("Daily New Ebola Cases - Top 5 Affected Countries")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Read CSV
df_ebola = pd.read_csv(url)

# Preprocessing
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]

# Converting Date to datetime
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Sorting for correct diff calculation
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Daily new cases
df_ebola["New_cases"] = df_ebola.groupby("Country")["Cumulative_cases"].diff().fillna(df_ebola["Cumulative_cases"])

# Finding Least 5 affected countries
least5_countries = (
    df_ebola.groupby("Country")["Cumulative_cases"]
    .max()
    .sort_values(ascending=True)
    .head(5)
    .index
)

df_least5 = df_ebola[df_ebola["Country"].isin(least5_countries)]

# Plot
plt.figure(figsize=(12,6))
for country in least5_countries:
    country_data = df_least5[df_least5["Country"] == country]
    plt.plot(country_data["Date"], country_data["New_cases"], label=country)

plt.xlabel("Date")
plt.ylabel("Daily New Cases")
plt.title("Daily New Ebola Cases - Least 5 Affected Countries")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Loading dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df_ebola = pd.read_csv(url)

# Preprocessing
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]

# Converting date column
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Sorting for consistency
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Computing daily new cases per country
df_ebola["New_cases"] = df_ebola.groupby("Country")["Cumulative_cases"].diff().fillna(df_ebola["Cumulative_cases"])

# Aggregating to global level
df_global = df_ebola.groupby("Date", as_index=False)["New_cases"].sum()

# Plot as a mountain shape
plt.figure(figsize=(12,6))
plt.fill_between(df_global["Date"], df_global["New_cases"], color="orange", alpha=0.6)
plt.plot(df_global["Date"], df_global["New_cases"], color="red", linewidth=2)

plt.title("Global New Ebola Cases Over Time (2014–2016)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Loading dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df_ebola = pd.read_csv(url)

# Preprocessing
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Sorting
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Computing daily new cases & deaths
df_ebola["New_cases"] = df_ebola.groupby("Country")["Cumulative_cases"].diff().fillna(df_ebola["Cumulative_cases"])
df_ebola["New_deaths"] = df_ebola.groupby("Country")["Cumulative_deaths"].diff().fillna(df_ebola["Cumulative_deaths"])

# Aggregating by quarter (global)
df_ebola["Quarter"] = df_ebola["Date"].dt.to_period("Q")
df_quarterly = df_ebola.groupby("Quarter", as_index=False)[["New_cases", "New_deaths"]].sum()

# Visualization: Stacked Bar
fig, ax = plt.subplots(figsize=(12,6))

ax.bar(df_quarterly["Quarter"].astype(str),
       df_quarterly["New_cases"],
       label="New Cases",
       color="skyblue")

ax.bar(df_quarterly["Quarter"].astype(str),
       df_quarterly["New_deaths"],
       bottom=df_quarterly["New_cases"],
       label="New Deaths",
       color="red")

ax.set_title("Quarterly Ebola New Cases and Deaths (2014–2016)", fontsize=14)
ax.set_ylabel("Count")
ax.legend()
plt.xticks(rotation=45)

ax.spines['bottom'].set_position(('data', 0))
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color("black")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Loading dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df_ebola = pd.read_csv(url)

# Preprocessing
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Sorting
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Computing daily new cases & deaths
df_ebola["New_cases"] = df_ebola.groupby("Country")["Cumulative_cases"].diff().fillna(df_ebola["Cumulative_cases"])
df_ebola["New_deaths"] = df_ebola.groupby("Country")["Cumulative_deaths"].diff().fillna(df_ebola["Cumulative_deaths"])

# Aggregating by quarter (global)
df_ebola["Quarter"] = df_ebola["Date"].dt.to_period("Q")
df_quarterly = df_ebola.groupby("Quarter", as_index=False)[["New_cases", "New_deaths"]].sum()

# Visualization: Double Bar Chart
x = np.arange(len(df_quarterly))   # positions for quarters
width = 0.35                       # width of each bar

fig, ax = plt.subplots(figsize=(12,6))

bars_cases = ax.bar(x - width/2,
                    df_quarterly["New_cases"],
                    width,
                    label="New Cases",
                    color="skyblue")

bars_deaths = ax.bar(x + width/2,
                     df_quarterly["New_deaths"],
                     width,
                     label="New Deaths",
                     color="red")

#  X-axis settings
ax.set_title("Quarterly Ebola New Cases vs Deaths (2014–2016)", fontsize=14)
ax.set_ylabel("Count")
ax.set_xlabel("Quarter")
ax.set_xticks(x)
ax.set_xticklabels(df_quarterly["Quarter"].astype(str), rotation=45)

# For x-axis line at y=0
ax.axhline(0, color="black", linewidth=1.2)

ax.legend()
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

#  Loading dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df_ebola = pd.read_csv(url)

# Preprocessing
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Get the latest cumulative deaths for each country
latest_deaths = (
    df_ebola.sort_values("Date")
            .groupby("Country")["Cumulative_deaths"]
            .last()
            .reset_index()
)

# Top 10 countries by cumulative deaths
top10_deaths = latest_deaths.sort_values(by="Cumulative_deaths", ascending=False).head(10)

# Pie Chart
fig, ax = plt.subplots(figsize=(10, 8))

# Function to show percentage + absolute number
def func(pct, values):
    absolute = int(round(pct/100.*sum(values)))
    return f"{pct:.1f}%\n({absolute:,})"

wedges, texts, autotexts = ax.pie(
    top10_deaths["Cumulative_deaths"],
    autopct=lambda pct: func(pct, top10_deaths["Cumulative_deaths"]),
    startangle=90,
    colors=plt.cm.tab10.colors,  # distinct colors
    textprops=dict(color="black", fontsize=10)
)

# Adding legend for clarity
ax.legend(
    wedges,
    top10_deaths["Country"],
    title="Countries",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=10
)

plt.title("Top 10 Countries Most Affected by Ebola (Cumulative Deaths)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Loading dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df_ebola = pd.read_csv(url)

# Preprocessing
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Sorting
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Computing daily new cases
df_ebola["New_cases"] = df_ebola.groupby("Country")["Cumulative_cases"].diff().fillna(df_ebola["Cumulative_cases"])

# Monthly Aggregation by Country
df_ebola["Month_Year"] = df_ebola["Date"].dt.to_period("M")

monthly_country = (
    df_ebola.groupby(["Country", "Month_Year"])["New_cases"]
    .sum()
    .reset_index()
)

# Pivoting for heatmap
heatmap_data = monthly_country.pivot(index="Country", columns="Month_Year", values="New_cases").fillna(0)

#  Heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(
    heatmap_data,
    cmap="Reds",
    linewidths=0.4,
    linecolor="white",
    cbar_kws={'label': 'Monthly New Cases'}
)

plt.title("Ebola Intensity Heatmap: Monthly New Cases by Country", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Month-Year", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Loading dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df_ebola = pd.read_csv(url)

# Preprocessing
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Sorting
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Computing daily new deaths
df_ebola["New_deaths"] = df_ebola.groupby("Country")["Cumulative_deaths"].diff().fillna(df_ebola["Cumulative_deaths"])

# Adding Quarter
df_ebola["Quarter"] = df_ebola["Date"].dt.to_period("Q")

# Region Mapping
region_map = {
    "Guinea": "West Africa",
    "Liberia": "West Africa",
    "Sierra Leone": "West Africa",
    "Nigeria": "West Africa",
    "Mali": "West Africa",
    "Senegal": "West Africa",
    "United States": "North America",
    "Spain": "Europe",
    "United Kingdom": "Europe",
    "Italy": "Europe"
}
df_ebola["Region"] = df_ebola["Country"].map(region_map).fillna("Other")

#Quarterly Aggregation by Region
quarterly_region = (
    df_ebola.groupby(["Region", "Quarter"])["New_deaths"]
    .sum()
    .reset_index()
)

# Pivot for heatmap
heatmap_data = quarterly_region.pivot(index="Region", columns="Quarter", values="New_deaths").fillna(0)

# Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_data,
    cmap="Purples",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={'label': 'Quarterly New Deaths'}
)

plt.title("Ebola Quarterly Death Intensity by Region", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Quarter", fontsize=12)
plt.ylabel("Region", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df_ebola = pd.read_csv(url)

#  Preprocessing
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")

# Sort
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Computing daily new cases
df_ebola["New_cases"] = df_ebola.groupby("Country")["Cumulative_cases"].diff().fillna(df_ebola["Cumulative_cases"])

#  Adding Month-Year
df_ebola["Month_Year"] = df_ebola["Date"].dt.to_period("M").astype(str)

# Selecting Top 10 Countries by Total Cases -
top10_countries = (
    df_ebola.groupby("Country")["Cumulative_cases"].max()
    .sort_values(ascending=False)
    .head(10)
    .index
)

df_top10 = df_ebola[df_ebola["Country"].isin(top10_countries)]

# Aggregating by Month & Country
monthly_country = (
    df_top10.groupby(["Country", "Month_Year"])["New_cases"]
    .sum()
    .reset_index()
)

# Pivoting for heatmap
heatmap_data = monthly_country.pivot(index="Country", columns="Month_Year", values="New_cases").fillna(0)

#  Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data,
    cmap="OrRd",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={'label': 'Monthly New Cases'}
)

plt.title("Ebola Monthly New Cases Intensity (Top 10 Countries)", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Month-Year", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Loading dataset
file_id = "1hf-UA5kAmqCDW8GPfCkgIuK6FZuaAv3W"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

df_ebola = pd.read_csv(url)
df_ebola.columns = ["Country", "Date", "Cumulative_cases", "Cumulative_deaths"]
df_ebola["Date"] = pd.to_datetime(df_ebola["Date"], errors="coerce")
df_ebola = df_ebola.sort_values(["Country", "Date"])

# Computing new cases/deaths
df_ebola["New_cases"] = df_ebola.groupby("Country")["Cumulative_cases"].diff().fillna(df_ebola["Cumulative_cases"])
df_ebola["New_deaths"] = df_ebola.groupby("Country")["Cumulative_deaths"].diff().fillna(df_ebola["Cumulative_deaths"])

# Adding Quarter for aggregation
df_ebola["Quarter"] = df_ebola["Date"].dt.to_period("Q").astype(str)

# Building Dashboard
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Ebola Dashboard", style={"textAlign": "center"}),

    # Dropdown for selecting country
    html.Label("Select Country:"),
    dcc.Dropdown(
        id="country-dropdown",
        options=[{"label": c, "value": c} for c in sorted(df_ebola["Country"].unique())],
        value="Guinea",  # default
        multi=False
    ),
   # Line chart - Daily new cases & deaths
    dcc.Graph(id="line-cases-deaths"),

    # Cumulative chart
    dcc.Graph(id="cumulative-trends"),

    # Stacked bar chart - quarterly
    dcc.Graph(id="stacked-bar"),

    # Choropleth map - total cases
    dcc.Graph(id="choropleth-map")
])

# Callbacks
@app.callback(
    [Output("line-cases-deaths", "figure"),
     Output("cumulative-trends", "figure"),
     Output("stacked-bar", "figure"),
     Output("choropleth-map", "figure")],
    [Input("country-dropdown", "value")]
)
def update_dashboard(selected_country):
    # Filtering by selected country
    df_country = df_ebola[df_ebola["Country"] == selected_country]

    # 1. Line chart (daily new cases/deaths)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df_country["Date"], y=df_country["New_cases"],
                                  mode="lines", name="New Cases", line=dict(color="blue")))
    fig_line.add_trace(go.Scatter(x=df_country["Date"], y=df_country["New_deaths"],
                                  mode="lines", name="New Deaths", line=dict(color="red")))
    fig_line.update_layout(title=f"Daily New Cases & Deaths - {selected_country}",
                           xaxis_title="Date", yaxis_title="Count")

    # 2. Cumulative chart
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=df_country["Date"], y=df_country["Cumulative_cases"],
                                        mode="lines", name="Cumulative Cases", line=dict(color="blue")))
    fig_cumulative.add_trace(go.Scatter(x=df_country["Date"], y=df_country["Cumulative_deaths"],
                                        mode="lines", name="Cumulative Deaths", line=dict(color="red")))
    fig_cumulative.update_layout(title=f"Cumulative Cases & Deaths - {selected_country}",
                                 xaxis_title="Date", yaxis_title="Total Count")
   # 3. Stacked bar chart (quarterly global cases vs deaths)
    quarterly_data = df_ebola.groupby("Quarter")[["New_cases", "New_deaths"]].sum().reset_index()
    fig_stacked = go.Figure()
    fig_stacked.add_trace(go.Bar(x=quarterly_data["Quarter"], y=quarterly_data["New_cases"],
                                 name="New Cases", marker_color="blue"))
    fig_stacked.add_trace(go.Bar(x=quarterly_data["Quarter"], y=quarterly_data["New_deaths"],
                                 name="New Deaths", marker_color="red"))
    fig_stacked.update_layout(barmode="stack",
                              title="Quarterly Cases vs Deaths (Global)",
                              xaxis_title="Quarter", yaxis_title="Count")

    # 4. Choropleth map - Total Cases by country
    country_grouped = df_ebola.groupby("Country", as_index=False)["Cumulative_cases"].max()
    fig_map = px.choropleth(
        country_grouped,
        locations="Country",
        locationmode="country names",
        color="Cumulative_cases",
        hover_name="Country",
        color_continuous_scale="Viridis",
        title="Total Ebola Cases by Country"
    )

    return fig_line, fig_cumulative, fig_stacked, fig_map


if __name__ == "__main__":
    app.run(debug=True)

df_top5 = df_ebola[df_ebola["Country"].isin(top5_countries)]
