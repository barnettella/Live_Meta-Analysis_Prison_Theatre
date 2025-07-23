#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In[39]:


# Code below taken from ShinyLive documentation 

from io import StringIO
import pyodide.http
from shiny import App, render, ui

git_repo_url = "https://api.github.com/repos/barnettella/Live_Meta-Analysis_Prison_Theatre/contents"


async def get_github_contents(path=None):
    url = f"{git_repo_url}/{path}" if path else git_repo_url
    response = await pyodide.http.pyfetch(url)
    if response.status != 200:
        raise Exception(f"Error fetching {url}: {response.status}")
    return await response.json()


async def download_file(github_file_info):
    url = github_file_info["download_url"]
    response = await pyodide.http.pyfetch(url)
    if response.status != 200:
        raise Exception(f"Error fetching {url}: {response.status}")
    return {"name": github_file_info["name"], "data": await response.string()}


async def get_df_from_github(github_file_info):
    file = await download_file(github_file_info)
    df = pd.read_csv(StringIO(file["data"]))
    return { "name": file["name"], "df": df }


async def get_dfs_from_github(top_level_folder):
    sd_files = await get_github_contents(f"{top_level_folder}/Standard%20deviation")
    z_score_files = await get_github_contents(f"{top_level_folder}/Z%20score")

    return {
        "sd_files": [await get_df_from_github(file) for file in sd_files],
        "z_score_files": [await get_df_from_github(file) for file in z_score_files],
    }

all_data = None
async def get_all_data():
    global all_data
    # Ensure all_data is only fetched once
    if all_data is None:
        all_data = {key: await get_dfs_from_github(key) for key in ["Empathy", "Identity"]}
    return all_data


def compute_effect_from_df(df, study_name, r=0.5):
   # Read the Dataframe and make all of the column titles lower case
    df.columns = map(str.lower, df.columns)

    # Compute hedges g using within-subject design hedges g formula + return effect size 
    def compute_hedges_g(row):
        sd_change = np.sqrt(row['sd_1']**2 + row['sd_2']**2 - 2 * r * row['sd_1'] * row['sd_2']) # Calculate standard deviation change
        d = (row['mean_2'] - row['mean_1']) / sd_change # calculate cohens d
        n = row['sample size']  # Number of paired observations
        dof = n - 1 # degrees of freedom
        J = 1 - (3 / (4 * dof - 1)) # adjustment for small sample sizes
        g = d * J # calculation for hedges g 
        return g

    # compute the variance in this to establish confidence intervals
    def compute_var_g(row):
        n = row['sample size']
        g = row['hedges_g']
        return (1 / n) + (g**2 / (2 * n)) # formula for var g 

    # apply functions to dataframe and add in new columns with calculated values + study name
    df['hedges_g'] = df.apply(compute_hedges_g, axis=1)
    df['var_g'] = df.apply(compute_var_g, axis=1)
    df['study'] = study_name  
    return df

# TO DO: make sure that this takes all sd from sd folder, and all z scores for the z score buddies and brings them togetehr


# In[92]:

def compute_effect_from_df_z(df, study_name):
    df.columns = map(str.lower, df.columns) 

    def compute_hedges_g_from_z(row):
        d = abs(row['z_score']) / np.sqrt(row['sample size']) # cohens d calculation (from zscore)
        n = row['sample size']  # Number of paired observations
        dof = n - 1 # degrees of freedom
        J = 1 - (3 / (4 * dof - 1))# adjustment for small sample sizes
        g = d * J # calculation for hedges g 
        return g

    # compute the variance in this to establish confidence intervals
    def compute_var_g_from_z(row):
        g = row['hedges_g']
        n = row['sample size']
        return (1 / n) + (g**2 / (2 * n))# formula for var g 

    #  apply functions to dataframe and add in new columns with calculate values + study name
    df['hedges_g'] = df.apply(compute_hedges_g_from_z, axis=1)
    df['var_g'] = df.apply(compute_var_g_from_z, axis=1)
    df['study'] = study_name  # Track study name ! for plotting
    
    return df

def pooled_effect_size(df):
    # Calculate weights for each study (1 / variance of g)
    weights = 1 / df['var_g']
    
    # Weighted mean of the effect sizes (Hedges' g)
    weighted_g = np.sum(df['hedges_g'] * weights) / np.sum(weights)
    
    # Estimate the total variance of the pooled effect size (random-effects model)
    tau_squared = np.var(df['hedges_g'], ddof=1)  # Between-study variance (random-effects)
    total_variance = 1 / np.sum(weights) + tau_squared / np.sum(weights)  # Total variance
    pooled_var = total_variance / len(df)  # Variance of the pooled estimate
    
    # Calculate 95% CI for the pooled effect size
    ci_lower = weighted_g - 1.96 * np.sqrt(pooled_var)
    ci_upper = weighted_g + 1.96 * np.sqrt(pooled_var)
    
    return weighted_g, ci_lower, ci_upper


def plot_forest(df, pooled_g, ci_lower, ci_upper):
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each study's effect size with CI
    for i, row in df.iterrows():
        ax.plot([row['hedges_g'] - 1.96 * np.sqrt(row['var_g']), row['hedges_g'] + 1.96 * np.sqrt(row['var_g'])],
                [i, i], color='black', lw=2)  # CI line
        ax.scatter(row['hedges_g'], i, color='black', zorder=5)  # Point for effect size
    
    # Plot the pooled effect size
    ax.plot([ci_lower, ci_upper], [-1, -1], color='red', lw=2, label='Pooled Effect Size')
    ax.scatter(pooled_g, -1, color='red', zorder=5)
    
    # Add labels and title
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['study'], fontsize=10)
    ax.set_xlabel("Hedges' g", fontsize=12)
    ax.set_title("Forest Plot: Pooled Effect Size and Individual Studies", fontsize=14)
    ax.axvline(x=0, color='black', linestyle='--')  # Zero line (no effect)
    
    ax.legend()
    
    # Return the figure for shinylive rendering
    return fig


app_ui = ui.page_fluid(
    ui.h2("Plots with Preloaded Data"),
    ui.output_plot("empathy_forest_plot"),
    ui.output_plot("identity_forest_plot"),
    ui.output_plot("empathy_funnel_plot"),
    ui.output_plot("identity_funnel_plot"),
)


async def plot_funnel(df):
    pooled_g, _, _ = pooled_effect_size(df)
    effect_sizes = df['hedges_g'].values
    standard_errors = np.sqrt(df['var_g'].values)
    labels = df['study'].values 

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.scatter(effect_sizes, standard_errors, color='blue', alpha=0.7)

    # Reference line at combined effect size
    ax.axvline(x=pooled_g, linestyle='--', color='red', label="Combined Effect")

    for x, y, label in zip(effect_sizes, standard_errors, labels):
        ax.text(x + 0.02, y, label, fontsize=9, ha='left', va='center')

    # Confidence interval bounds (pseudo 95% CI lines for funnel shape)
    se_range = np.linspace(min(standard_errors), max(standard_errors), 100)
    lower_bound = pooled_g - 1.96 * se_range
    upper_bound = pooled_g + 1.96 * se_range
    ax.plot(lower_bound, se_range, linestyle='--', color='gray', label="95% Confidence Intervals")
    ax.plot(upper_bound, se_range, linestyle='--', color='gray')

    # Invert y-axis: smaller SE (larger studies) at the top
    ax.invert_yaxis()

    # Labels and styling
    ax.set_xlabel("Effect Size (Hedges' g)")
    ax.set_ylabel("Standard Error")
    ax.set_title("Funnel Plot")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    return fig

async def compute_effects(folder_name):
    df_data = await get_all_data()

    #goes through data stuffs enough times for all the stuffs
    computed_effects = [
        compute_effect_from_df(df['df'], df['name'])[['hedges_g', 'var_g','measure','study']]
        for df in df_data[folder_name]["sd_files"]
    ]

    computed_effects_z = [
        compute_effect_from_df_z(df['df'], df['name'])[['hedges_g', 'var_g','measure','study']]
        for df in df_data[folder_name]["z_score_files"]
    ]

    # Concatenate the selected rows vertically (axis=0)
    concatenated_data = pd.concat(computed_effects + computed_effects_z, axis=0)
    concatenated_data = concatenated_data.reset_index(drop=True)

    return concatenated_data

async def compute_and_plot_forest(folder_name):
    concatenated_data = await compute_effects(folder_name)
    # Calculate pooled effect size and confidence intervals
    pooled_g, ci_lower, ci_upper = pooled_effect_size(concatenated_data)
    # Plot the forest plot
    return plot_forest(concatenated_data, pooled_g, ci_lower, ci_upper)

async def compute_and_plot_funnel(folder_name):
    concatenated_data = await compute_effects(folder_name)
    return await plot_funnel(concatenated_data)

def server(input, output, session):
    @output()
    @render.plot
    async def empathy_forest_plot():
        return await compute_and_plot_forest("Empathy")
    
    @output()
    @render.plot
    async def identity_forest_plot():
        return await compute_and_plot_forest("Identity")
    
    @output()
    @render.plot
    async def empathy_funnel_plot():
        return await compute_and_plot_funnel("Empathy")
    
    @output()
    @render.plot
    async def identity_funnel_plot():
        return await compute_and_plot_funnel("Identity")

    
app = App(app_ui, server)