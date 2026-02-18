# Environment shows up as workspace (2)
"""
8-K Item Timeliness and Materiality Dashboard

This dashboard analyzes regression results examining the timeliness (currentness)
and materiality of 8-K disclosure items.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import io
import os
import re


# Page configuration
st.set_page_config(
    page_title="8-K Item Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("8-K Item Timeliness and Materiality Dashboard")
st.markdown("---")

# Filing type selector at the top
filing_type = st.selectbox(
    "Filing Type",
    options=["All Filings", "Gap Filings", "No Gap Filings"],
    index=0
)

st.markdown("---")

# Load data
@st.cache_data
def load_data(filing_type):
    """Load regression results and residual standard deviations"""
    try:
        # Determine file prefix based on filing type
        if filing_type == "Gap Filings":
            prefix = "gap_"
        elif filing_type == "No Gap Filings":
            prefix = "nogap_"
        else:  # All Filings
            prefix = ""

        # Load and combine regression results
        results_nonretail = pd.read_csv(f'{prefix}regression_results_parsed.csv')
        results_retail = pd.read_csv(f'{prefix}regression_retail_results_parsed.csv')
        results = pd.concat([results_nonretail, results_retail], ignore_index=True)

        # Rename columns to match expected format
        results = results.rename(columns={
            'estimate': 'Coefficient',
            'statistic': 'T_Statistic',
            'term': 'Item'
        })

        # Create Item_Description if missing
        if 'Item_Description' not in results.columns:
            item_desc_map = {
                'Item_1.01': 'Entry into Material Agreement',
                'Item_1.02': 'Termination of Material Agreement',
                'Item_1.03': 'Bankruptcy or Receivership',
                'Item_1.04': 'Mine Safety',
                'Item_1.05': 'Material Cybersecurity Incident',
                'Item_2.01': 'Completion of Acquisition/Disposition',
                'Item_2.02': 'Results of Operations',
                'Item_2.03': 'Creation of Direct Financial Obligation',
                'Item_2.04': 'Triggering Events Accelerating Obligation',
                'Item_2.05': 'Costs Associated with Exit Activities',
                'Item_2.06': 'Material Impairments',
                'Item_3.01': 'Notice of Delisting',
                'Item_3.02': 'Unregistered Sales of Equity',
                'Item_3.03': 'Material Modification to Rights of Security Holders',
                'Item_4.01': 'Changes in Registrant Certifying Accountant',
                'Item_4.02': 'Non Reliance on Previously Issued Financial Statements',
                'Item_5.01': 'Changes in Control',
                'Item_5.02': 'Departure/Election of Directors or Officers',
                'Item_5.03': 'Amendments to Articles/Bylaws',
                'Item_5.04': 'Temporary Suspension of Trading',
                'Item_5.05': 'Amendments or Waiver to the Code of Ethics',
                'Item_5.06': 'Transaction in Which a Company Ceases to be a Shell Company',
                'Item_5.07': 'Results of Shareholder Voting',
                'Item_5.08': 'Shareholder Director Nominations',
                'Item_7.01': 'Regulation FD Disclosure',
                'Item_8.01': 'Other Events',
                'Item_9.01': 'Financial Statements and Exhibits'
            }
            results['Item_Description'] = results['Item'].map(item_desc_map).fillna('Unknown Item')

        # Load and combine residuals
        residual_sds_nonretail = pd.read_csv('residuals_all_parsed.csv')
        residual_sds_retail = pd.read_csv('residuals_retail_parsed.csv')
        residual_sds = pd.concat([residual_sds_nonretail, residual_sds_retail], ignore_index=True)

        # Rename columns to match expected format
        residual_sds = residual_sds.rename(columns={
            'residual_sd': 'Residual_SD'
        })

        return results, residual_sds
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}. Please ensure all required CSV files are in the directory.")
        st.stop()

@st.cache_data
def load_disclosures():
    """Load the disclosures_items_only.csv file with filing-level data"""
    try:

        disclosures = pd.read_csv('disclosures_items_only.csv', low_memory=False)
        return disclosures
    except FileNotFoundError as e:
        st.error(f"disclosures_items_only.csv not found: {e}")
        st.stop()

results_df, residual_sds_df = load_data(filing_type)

disclosures_df = load_disclosures()

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Time horizon selection
time_horizon = st.sidebar.selectbox(
    "Time Horizon",
    options=["48 hours", "24 hours", "2 hours", "1 hour"],
    index=1  # Default to 24 hours
)

# The Window column in your data is already "48 hours", "24 hours", etc.
# We just use time_horizon directly as the window
selected_window = time_horizon

# Significance threshold
sig_threshold = st.sidebar.slider(
    "Significance Level for 'Current' Classification (%)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=1.0
) / 100

# Significance threshold for materiality
materiality_sig_threshold = st.sidebar.slider(
    "Significance Level for 'Material' Classification (%)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=1.0
) / 100

# Coefficient scaling toggle
scale_by_sd = st.sidebar.checkbox(
    "Scale coefficients by residual standard deviations",
    value=False
)

# Materiality threshold (changes based on scaling)
if scale_by_sd:
    materiality_threshold = st.sidebar.slider(
        "Materiality Threshold (as % of Residual SD)",
        min_value=5.0,
        max_value=50.0,
        value=20.0,
        step=1.0
    ) / 100
else:
    materiality_threshold = st.sidebar.slider(
        "Materiality Threshold (basis points)",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    ) / 10000  # Convert basis points to decimal

# Filing-level statistics toggle
show_filing_stats = st.sidebar.checkbox(
    "Show filing-level statistics",
    value=False,
    help="Calculate and display statistics about filings (slower to compute)"
)

# Item 9.01 inclusion toggle (only show if filing stats are enabled)
if show_filing_stats:
    include_item_901 = st.sidebar.checkbox(
        "Include Item 9.01 in filing-level statistics",
        value=True,
        help="Item 9.01 (Financial Statements and Exhibits) occurs frequently with other items"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("*Adjust controls to dynamically update results*")

# Helper functions
def get_critical_value(alpha):
    """Get critical value for two-tailed test"""
    return stats.norm.ppf(1 - alpha /2)

def is_significant(t_stat, alpha):
    """Check if t-statistic is significant at given alpha level"""
    critical_value = get_critical_value(alpha)
    return abs(t_stat) > critical_value

def get_significance_stars(t_stat):
    """Get significance stars based on t-statistic"""
    if abs(t_stat) > stats.norm.ppf(1 - 0.01 /2):
        return "***"
    elif abs(t_stat) > stats.norm.ppf(1 - 0.05 /2):
        return "**"
    elif abs(t_stat) > stats.norm.ppf(1 - 0.10 /2):
        return "*"
    else:
        return ""

def get_coefficient_value(item, dv, window, horizon, scaled=False):
    """Get coefficient value for specific item, DV, window, and horizon"""
    row = results_df[
        (results_df['Item'] == item) &
        (results_df['Dependent_Variable'] == dv) &
        (results_df['Window'] == window) &
        (results_df['Horizon'] == horizon)
        ]

    if row.empty:
        return None, None

    coef = row['Coefficient'].values[0]
    t_stat = row['T_Statistic'].values[0]

    if scaled:
        sd_row = residual_sds_df[
            (residual_sds_df['Dependent_Variable'] == dv) &
            (residual_sds_df['Window'] == window) &
            (residual_sds_df['Horizon'] == horizon)
            ]
        if not sd_row.empty:
            sd = sd_row['Residual_SD'].values[0]
            coef = coef / sd

    return coef, t_stat

def is_current(item):
    """Check if item is current based on total volume"""
    post_coef, post_t = get_coefficient_value(item, 'Total Volume', selected_window, 'Post')
    pre_coef, pre_t = get_coefficient_value(item, 'Total Volume', selected_window, 'Pre')

    if post_coef is None or pre_coef is None:
        return False

    # Post must be positive and significant
    post_significant = is_significant(post_t, sig_threshold)
    post_positive = post_coef > 0

    # Pre must be insignificant or negative
    pre_insignificant = not is_significant(pre_t, sig_threshold)
    pre_negative = pre_coef <= 0

    return post_positive and post_significant and (pre_insignificant or pre_negative)

def is_material(item):
    """Check if item is material based on absolute returns"""
    post_coef, post_t = get_coefficient_value(item, 'Absolute Returns', selected_window, 'Post')
    pre_coef, pre_t = get_coefficient_value(item, 'Absolute Returns', selected_window, 'Pre')

    if post_coef is None or pre_coef is None:
        return False

    # Treat non-significant coefficients as zero
    if not is_significant(post_t, materiality_sig_threshold):
        post_coef = 0
    if not is_significant(pre_t, materiality_sig_threshold):
        pre_coef = 0

    total_effect = pre_coef + post_coef

    if scale_by_sd:
        # Get residual SD for returns (use Post period)
        sd_row = residual_sds_df[
            (residual_sds_df['Dependent_Variable'] == 'Absolute Returns') &
            (residual_sds_df['Window'] == selected_window) &
            (residual_sds_df['Horizon'] == 'Post')
            ]

        if sd_row.empty:
            return False

        sd = sd_row['Residual_SD'].values[0]
        # Sum of pre and post > threshold * SD
        return total_effect > (materiality_threshold * sd)
    else:
        # Direct comparison in basis points (already converted to decimal)
        return total_effect > materiality_threshold

def has_retail_flag(item):
    """Check if retail responds when total volume doesn't"""
    # Get Retail-specific data (where Retail_Flag == 'Retail')
    retail_post_coef, retail_post_t = get_coefficient_value_retail(item, 'Total Volume', selected_window, 'Post', 'Retail')
    if retail_post_coef is None:
        return False

    retail_significant = is_significant(retail_post_t, sig_threshold)
    retail_positive = retail_post_coef > 0

    # Total volume post must be insignificant or not positive (from non-retail data)
    total_post_coef, total_post_t = get_coefficient_value_retail(item, 'Total Volume', selected_window, 'Post', 'Not Retail')
    if total_post_coef is None:
        return False

    total_insignificant = not is_significant(total_post_t, sig_threshold)
    total_not_positive = total_post_coef <= 0

    return retail_positive and retail_significant and (total_insignificant or total_not_positive)

def get_coefficient_value_retail(item, dv, window, horizon, retail_flag, scaled=False):
    """Get coefficient value for specific item, DV, window, horizon, and retail flag"""
    row = results_df[
        (results_df['Item'] == item) &
        (results_df['Dependent_Variable'] == dv) &
        (results_df['Window'] == window) &
        (results_df['Horizon'] == horizon) &
        (results_df['Retail_Flag'] == retail_flag)
        ]

    if row.empty:
        return None, None

    coef = row['Coefficient'].values[0]
    t_stat = row['T_Statistic'].values[0]

    if scaled:
        sd_row = residual_sds_df[
            (residual_sds_df['Dependent_Variable'] == dv) &
            (residual_sds_df['Window'] == window) &
            (residual_sds_df['Horizon'] == horizon) &
            (residual_sds_df['Retail_Flag'] == retail_flag)
            ]
        if not sd_row.empty:
            sd = sd_row['Residual_SD'].values[0]
            coef = coef / sd

    return coef, t_stat

# Classify all items
items = results_df['Item'].unique()
classifications = {
    'current_material': [],
    'current_not_material': [],
    'material_not_current': [],
    'neither': []
}

for item in items:
    current = is_current(item)
    material = is_material(item)

    if current and material:
        classifications['current_material'].append(item)
    elif current and not material:
        classifications['current_not_material'].append(item)
    elif material and not current:
        classifications['material_not_current'].append(item)
    else:
        classifications['neither'].append(item)

def calculate_filing_statistics(classifications, disclosures_df, filing_type, include_901=True):
    """Calculate filing-level statistics based on item classifications"""

    # Filter disclosures based on filing type
    if filing_type == "Gap Filings":
        filings = disclosures_df[disclosures_df['EventGap_dummy'] == 1].copy()
    elif filing_type == "No Gap Filings":
        filings = disclosures_df[disclosures_df['EventGap_dummy'] == 0].copy()
    else:  # All Filings
        filings = disclosures_df.copy()

    total_filings = len(filings)

    if total_filings == 0:
        return None

    # Get item columns
    item_columns = [col for col in filings.columns if col.startswith('Item_')]

    # Optionally exclude Item_9.01
    if not include_901 and 'Item_9.01' in item_columns:
        item_columns.remove('Item_9.01')

    # Create sets of items for each classification
    current_material_items = set(classifications['current_material'])
    current_items = set(classifications['current_material'] + classifications['current_not_material'])
    material_items = set(classifications['current_material'] + classifications['material_not_current'])
    neither_items = set(classifications['neither'])

    # Initialize counters
    stats = {
        'current_material': {'only': 0, 'at_least_one': 0, 'none': 0},
        'current': {'only': 0, 'at_least_one': 0, 'none': 0},
        'material': {'only': 0, 'at_least_one': 0, 'none': 0},
        'neither': {'only': 0, 'at_least_one': 0, 'none': 0}
    }

    # Analyze each filing
    for idx, row in filings.iterrows():
        # Get items present in this filing
        present_items = [col for col in item_columns if row[col] == 1]
        present_items_set = set(present_items)

        if len(present_items) == 0:
            continue

        # Check Current & Material
        cm_present = present_items_set & current_material_items
        if len(cm_present) > 0 and present_items_set.issubset(current_material_items):
            stats['current_material']['only'] += 1
        if len(cm_present) > 0:
            stats['current_material']['at_least_one'] += 1
        if len(cm_present) == 0:
            stats['current_material']['none'] += 1

        # Check Current (Current & Material + Current Not Material)
        c_present = present_items_set & current_items
        if len(c_present) > 0 and present_items_set.issubset(current_items):
            stats['current']['only'] += 1
        if len(c_present) > 0:
            stats['current']['at_least_one'] += 1
        if len(c_present) == 0:
            stats['current']['none'] += 1

        # Check Material (Current & Material + Material Not Current)
        m_present = present_items_set & material_items
        if len(m_present) > 0 and present_items_set.issubset(material_items):
            stats['material']['only'] += 1
        if len(m_present) > 0:
            stats['material']['at_least_one'] += 1
        if len(m_present) == 0:
            stats['material']['none'] += 1

        # Check Neither
        n_present = present_items_set & neither_items
        if len(n_present) > 0 and present_items_set.issubset(neither_items):
            stats['neither']['only'] += 1
        if len(n_present) > 0:
            stats['neither']['at_least_one'] += 1
        if len(n_present) == 0:
            stats['neither']['none'] += 1

    # Convert to percentages
    for category in stats:
        for metric in stats[category]:
            stats[category][metric] = (stats[category][metric] / total_filings) * 100

    stats['total_filings'] = total_filings
    return stats

# Summary statistics
st.header("Summary")

# Item-level summary
summary_text = f"""Under current settings, **{len(classifications['current_material'])} items** are Current & Material,
**{len(classifications['current_not_material'])} items** are Current but Not Material,
**{len(classifications['material_not_current'])} items** are Material but Not Current, and
**{len(classifications['neither'])} items** are Neither Current nor Material."""
st.markdown(summary_text)

# Filing-level statistics (only if enabled)
if show_filing_stats:
    st.subheader("Filing-Level Statistics")
    filing_stats = calculate_filing_statistics(classifications, disclosures_df, filing_type, include_item_901)

    if filing_stats:
        st.markdown(f"*Based on {filing_stats['total_filings']:,} filings" +
                    (" (excluding Item 9.01)" if not include_item_901 else "") + "*")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Current & Material Items**")
            st.markdown(f"- Only these items: **{filing_stats['current_material']['only']:.1f}%**")
            st.markdown(f"- At least one: **{filing_stats['current_material']['at_least_one']:.1f}%**")
            st.markdown(f"- None: **{filing_stats['current_material']['none']:.1f}%**")

            st.markdown("")
            st.markdown("**Neither Current nor Material**")
            st.markdown(f"- Only these items: **{filing_stats['neither']['only']:.1f}%**")

        with col2:
            st.markdown("**Current Items**")
            st.markdown(f"- Only current items: **{filing_stats['current']['only']:.1f}%**")
            st.markdown(f"- At least one: **{filing_stats['current']['at_least_one']:.1f}%**")
            st.markdown(f"- No current items: **{filing_stats['current']['none']:.1f}%**")

        with col3:
            st.markdown("**Material Items**")
            st.markdown(f"- Only material items: **{filing_stats['material']['only']:.1f}%**")
            st.markdown(f"- At least one: **{filing_stats['material']['at_least_one']:.1f}%**")
            st.markdown(f"- No material items: **{filing_stats['material']['none']:.1f}%**")

st.markdown("---")

# Build classification table
def build_classification_table():
    """Build the full classification table"""
    table_data = []

    for category in ['current_material', 'current_not_material', 'material_not_current', 'neither']:
        for item in classifications[category]:
            # Get item description
            desc_row = results_df[results_df['Item'] == item].iloc[0]
            description = desc_row['Item_Description']

            # Get all coefficients
            row_data = {
                'Category': category.replace('_', ' ').title(),
                'Item': item.replace('_', ' '),
                'Description': description,
                'N': f"{disclosures_df[item].sum():,.0f}"
            }

            # Total Volume (Non-Retail)
            tv_pre_coef, tv_pre_t = get_coefficient_value_retail(item, 'Total Volume', selected_window, 'Pre', 'Not Retail', scale_by_sd)
            tv_post_coef, tv_post_t = get_coefficient_value_retail(item, 'Total Volume', selected_window, 'Post', 'Not Retail', scale_by_sd)
            if tv_pre_coef is not None and tv_post_coef is not None:
                row_data['Total_Vol_Pre'] = f"{tv_pre_coef:.4f}{get_significance_stars(tv_pre_t)}"
                row_data['Total_Vol_Post'] = f"{tv_post_coef:.4f}{get_significance_stars(tv_post_t)}"
            else:
                row_data['Total_Vol_Pre'] = "N/A"
                row_data['Total_Vol_Post'] = "N/A"

            # Retail Volume
            rv_pre_coef, rv_pre_t = get_coefficient_value_retail(item, 'Total Volume', selected_window, 'Pre', 'Retail', scale_by_sd)
            rv_post_coef, rv_post_t = get_coefficient_value_retail(item, 'Total Volume', selected_window, 'Post', 'Retail', scale_by_sd)
            if rv_pre_coef is not None and rv_post_coef is not None:
                row_data['Retail_Vol_Pre'] = f"{rv_pre_coef:.4f}{get_significance_stars(rv_pre_t)}"
                row_data['Retail_Vol_Post'] = f"{rv_post_coef:.4f}{get_significance_stars(rv_post_t)}"
            else:
                row_data['Retail_Vol_Pre'] = "N/A"
                row_data['Retail_Vol_Post'] = "N/A"

            # Absolute Returns (Non-Retail)
            ar_pre_coef, ar_pre_t = get_coefficient_value_retail(item, 'Absolute Returns', selected_window, 'Pre', 'Not Retail', scale_by_sd)
            ar_post_coef, ar_post_t = get_coefficient_value_retail(item, 'Absolute Returns', selected_window, 'Post', 'Not Retail', scale_by_sd)
            if ar_pre_coef is not None and ar_post_coef is not None:
                row_data['Returns_Pre'] = f"{ar_pre_coef:.4f}{get_significance_stars(ar_pre_t)}"
                row_data['Returns_Post'] = f"{ar_post_coef:.4f}{get_significance_stars(ar_post_t)}"
            else:
                row_data['Returns_Pre'] = "N/A"
                row_data['Returns_Post'] = "N/A"

            # Retail flag
            row_data['Retail_Flag'] = "🔔" if has_retail_flag(item) else ""

            table_data.append(row_data)

    return pd.DataFrame(table_data)

# Create and display table
st.header("Classification Table")
classification_df = build_classification_table()

# Display table grouped by category
for category in ['Current Material', 'Current Not Material', 'Material Not Current', 'Neither']:
    st.subheader(category)
    category_data = classification_df[classification_df['Category'] == category]
    if not category_data.empty:
        display_df = category_data.drop('Category', axis=1)
        st.dataframe(display_df, width='stretch', hide_index=True)
    else:
        st.write("*No items in this category*")

# Download button for table
st.markdown("---")
csv_buffer = io.StringIO()
classification_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Table as CSV",
    data=csv_buffer.getvalue(),
    file_name="8k_classification_table.csv",
    mime="text/csv"
)

st.markdown("---")

# Visualization section
st.header("Coefficient Visualizations")
st.markdown("*Bars show raw coefficients with 95% confidence intervals*")

def create_bar_chart(dv, dv_name, retail_flag='Not Retail'):
    """Create bar chart for a specific dependent variable"""
    # Get data for all items
    chart_data = []

    for item in items:
        pre_coef, pre_t = get_coefficient_value_retail(item, dv, selected_window, 'Pre', retail_flag, scaled=False)
        post_coef, post_t = get_coefficient_value_retail(item, dv, selected_window, 'Post', retail_flag, scaled=False)

        if pre_coef is not None and post_coef is not None:
            # Calculate standard errors and confidence intervals
            pre_se = pre_coef / pre_t if pre_t != 0 else 0
            post_se = post_coef / post_t if post_t != 0 else 0

            pre_ci_lower = pre_coef - 1.96 * abs(pre_se)
            pre_ci_upper = pre_coef + 1.96 * abs(pre_se)
            post_ci_lower = post_coef - 1.96 * abs(post_se)
            post_ci_upper = post_coef + 1.96 * abs(post_se)

            chart_data.append({
                'item': item,
                'pre_coef': pre_coef,
                'post_coef': post_coef,
                'pre_ci_lower': pre_ci_lower,
                'pre_ci_upper': pre_ci_upper,
                'post_ci_lower': post_ci_lower,
                'post_ci_upper': post_ci_upper
            })

    # Sort by post coefficient (descending)
    chart_data = sorted(chart_data, key=lambda x: x['post_coef'], reverse=True)

    # Create figure
    fig = go.Figure()

    # Add post-period bars (green)
    fig.add_trace(go.Bar(
        name='Post-Period',
        y=[d['item'] for d in chart_data],
        x=[d['post_coef'] for d in chart_data],
        orientation='h',
        marker_color='green',
        error_x=dict(
            type='data',
            symmetric=False,
            array=[d['post_ci_upper'] - d['post_coef'] for d in chart_data],
            arrayminus=[d['post_coef'] - d['post_ci_lower'] for d in chart_data]
        )
    ))

    # Add pre-period bars (red)
    fig.add_trace(go.Bar(
        name='Pre-Period',
        y=[d['item'] for d in chart_data],
        x=[d['pre_coef'] for d in chart_data],
        orientation='h',
        marker_color='red',
        error_x=dict(
            type='data',
            symmetric=False,
            array=[d['pre_ci_upper'] - d['pre_coef'] for d in chart_data],
            arrayminus=[d['pre_coef'] - d['pre_ci_lower'] for d in chart_data]
        )
    ))

    # Update layout
    fig.update_layout(
        title=f"{dv_name} Coefficients (Sorted by Post-Period)",
        xaxis_title="Coefficient Estimate",
        yaxis_title="8-K Item",
        barmode='group',
        height=800,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        yaxis=dict(autorange="reversed")
    )

    return fig

# Create charts for each dependent variable
st.subheader("Total Volume")
fig_tv = create_bar_chart('Total Volume', 'Total Volume', 'Not Retail')
st.plotly_chart(fig_tv, width='stretch')

st.subheader("Retail Volume")
fig_rv = create_bar_chart('Total Volume', 'Retail Volume', 'Retail')
st.plotly_chart(fig_rv, width='stretch')

st.subheader("Absolute Returns")
fig_ar = create_bar_chart('Absolute Returns', 'Absolute Returns', 'Not Retail')
st.plotly_chart(fig_ar, width='stretch')

# Download button for charts (requires kaleido and Chrome)
st.markdown("---")
try:
    col1, col2, col3 = st.columns(3)

    with col1:
        img_bytes_tv = fig_tv.to_image(format="png", width=1200, height=800)
        st.download_button(
            label="Download Total Volume Chart",
            data=img_bytes_tv,
            file_name="total_volume_chart.png",
            mime="image/png"
        )

    with col2:
        img_bytes_rv = fig_rv.to_image(format="png", width=1200, height=800)
        st.download_button(
            label="Download Retail Volume Chart",
            data=img_bytes_rv,
            file_name="retail_volume_chart.png",
            mime="image/png"
        )

    with col3:
        img_bytes_ar = fig_ar.to_image(format="png", width=1200, height=800)
        st.download_button(
            label="Download Returns Chart",
            data=img_bytes_ar,
            file_name="returns_chart.png",
            mime="image/png"
        )
except Exception as e:
    st.info \
        ("💡 Chart downloads require Kaleido and Chrome. You can still interact with the charts above and use your browser's screenshot feature to save them.")

# Footer
st.markdown("---")
st.markdown \
    ("*Dashboard for analyzing 8-K item timeliness and materiality | Data updates dynamically with control changes*")