import json
import pandas as pd
from pathlib import Path
import numpy as np
# import modelop.utils as utils
# import modelop_sdk.restclient.moc_client as moc_client
from typing import Dict, Any, List, Generator

# --- Constants for Chart Generation ---
# Max unique categories for a "group by" or "column" variable to avoid cluttered charts
MAX_CATEGORICAL_GROUPS = 10
# Max unique categories for an "index" or "axis" variable
MAX_CATEGORICAL_CARDINALITY = 50
# Max records to plot for a scatter plot to avoid performance issues
MAX_RECORDS_FOR_SCATTER = 1000


# modelop.init
def init(init_param):
    job = json.loads(init_param["rawJson"])


def get_available_monitors(data: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
    """
    Analyzes a DataFrame and returns a dictionary of available monitors
    (chart types) and the column combinations that can be used for them.
    
    Args:
        data: The preprocessed DataFrame to analyze.

    Returns:
        A dictionary where keys are monitor types (e.g., 'time_line', 'bar')
        and values are lists of "specs" (dictionaries of column roles).
    """
    print("Analyzing DataFrame for available monitor specifications...")
    
    # Provide more granular chart-type buckets to match chart_mapping.json
    specs: Dict[str, List[Dict[str, str]]] = {
        "time_line_graph": [],
        "generic_line_graph": [],
        "decimal_line_graph": [],
        "generic_bar_graph": [],
        "horizontal_bar_graph": [],
        "generic_scatter_plot": [],
        "generic_pie_chart": [],
        "generic_donut_chart": []
    }

    # Collect base spec lists first, then distribute into the granular buckets
    time_specs: List[Dict[str, str]] = []
    bar_specs: List[Dict[str, str]] = []
    pie_specs: List[Dict[str, str]] = []
    scatter_specs: List[Dict[str, str]] = []

    # --- 1. Identify Column Types ---
    
    time_cols = list(data.select_dtypes(include=['datetime64[ns]']).columns)
    numeric_cols = list(data.select_dtypes(include=['float64', 'int64']).columns)
    
    # Categorical columns are those with 'object' type or low-cardinality integer types
    # (like 'Actual_Performance')
    categorical_cols = []
    potential_cat_cols = list(data.select_dtypes(include=['object', 'category']).columns)
    
    # Add low-cardinality numeric columns (like 0/1 flags)
    for col in numeric_cols:
        if data[col].nunique() <= MAX_CATEGORICAL_CARDINALITY:
            potential_cat_cols.append(col)
            
    # Filter for reasonable cardinality
    for col in potential_cat_cols:
        if col in data.columns and data[col].nunique() <= MAX_CATEGORICAL_CARDINALITY:
            categorical_cols.append(col)
            
    # We want "true" numeric columns, not low-cardinality flags
    true_numeric_cols = [
        col for col in numeric_cols 
        if col not in categorical_cols or data[col].nunique() > 2
    ]
    
    # Use the first available time column as the default
    time_col = time_cols[0] if time_cols else None
    
    print(f"Found: time_col='{time_col}', numeric_cols={true_numeric_cols}, categorical_cols={categorical_cols}")

    # --- 2. Define Chart "Specs" ---

    # Spec: Time-Line (Time vs. Numeric, grouped by Category)
    if time_col:
        for num_col in true_numeric_cols:
            for cat_col in categorical_cols:
                if data[cat_col].nunique() <= MAX_CATEGORICAL_GROUPS:
                    time_specs.append({
                        "time": time_col,
                        "numeric": num_col,
                        "category": cat_col
                    })

    # Spec: Bar Chart (Numeric by Category, grouped by a second Category)
    for num_col in true_numeric_cols:
        for cat_index in categorical_cols:
            for cat_cols in categorical_cols:
                if cat_index == cat_cols:
                    continue
                if data[cat_cols].nunique() <= MAX_CATEGORICAL_GROUPS:
                    bar_specs.append({
                        "index": cat_index,
                        "columns": cat_cols,
                        "numeric": num_col
                    })

    # Spec: Pie Chart (Count by Category)
    for cat_col in categorical_cols:
        pie_specs.append({"category": cat_col})

    # Spec: Scatter Plot (Numeric vs. Numeric)
    # Avoid duplicate plots (e.g., A vs B and B vs A)
    plotted_pairs = set()
    for x_col in true_numeric_cols:
        for y_col in true_numeric_cols:
            if x_col == y_col:
                continue
            if (y_col, x_col) not in plotted_pairs:
                scatter_specs.append({"x": x_col, "y": y_col})
                plotted_pairs.add((x_col, y_col))

    # --- Distribute base specs into the more granular buckets ---
    # Distribute time specs into three line-chart categories round-robin
    for i, s in enumerate(time_specs):
        if i % 3 == 0:
            specs["time_line_graph"].append(s)
        elif i % 3 == 1:
            specs["generic_line_graph"].append(s)
        else:
            specs["decimal_line_graph"].append(s)

    # Distribute bar specs between generic and horizontal alternately
    for i, s in enumerate(bar_specs):
        if i % 2 == 0:
            specs["generic_bar_graph"].append(s)
        else:
            specs["horizontal_bar_graph"].append(s)

    # Distribute pie specs between pie and donut alternately
    for i, s in enumerate(pie_specs):
        if i % 2 == 0:
            specs["generic_pie_chart"].append(s)
        else:
            specs["generic_donut_chart"].append(s)

    # Scatter stays as generic_scatter_plot
    specs["generic_scatter_plot"] = scatter_specs

    # Use the actual keys created above. Avoid KeyError when keys don't exist.
    print(
        f"Generated {len(specs.get('time_line_graph', []))} time-line, "
        f"{len(specs.get('generic_bar_graph', []))} bar, "
        f"{len(specs.get('generic_pie_chart', []))} pie, "
        f"{len(specs.get('generic_scatter_plot', []))} scatter specs."
    )
    return specs


# --- Helper Functions to Build Charts from Specs ---

def _generate_time_line_charts(data: pd.DataFrame, specs: List[Dict[str, str]]) -> Dict[str, Dict]:
    """Builds time_line_graph JSON objects from specs."""
    charts = {}
    for i, spec in enumerate(specs):
        t, n, c = spec['time'], spec['numeric'], spec['category']
        chart_key = f"timeline_{n.lower()}_by_{c.lower()}_{i}"
        chart_title = f"{n} over Time by {c}"
        
        try:
            # Group by time and category, get the mean of the numeric value
            df_agg = data.groupby([t, c])[n].mean().reset_index()
            
            chart_data = {}
            # Pivot the data into the required format: {category_name: [[timestamp, value], ...]}
            for cat_name, group_df in df_agg.groupby(c):
                series_data = group_df[[t, n]].apply(
                    lambda x: [x.iloc[0].isoformat(), round(x.iloc[1], 4)], 
                    axis=1
                ).tolist()
                chart_data[str(cat_name)] = series_data
            
            if not chart_data:
                continue

            charts[chart_key] = {
                "title": chart_title,
                "x_axis_label": t,
                "y_axis_label": n,
                "data": chart_data
            }
        except Exception as e:
            print(f"Failed to build {chart_key}: {e}")
    return charts

def _generate_bar_charts(data: pd.DataFrame, specs: List[Dict[str, str]]) -> Dict[str, Dict]:
    """Builds generic_bar_graph JSON objects from specs."""
    charts = {}
    for i, spec in enumerate(specs):
        idx, cols, num = spec['index'], spec['columns'], spec['numeric']
        chart_key = f"bar_{num.lower()}_by_{idx.lower()}_and_{cols.lower()}_{i}"
        chart_title = f"Avg. {num} by {idx} (Grouped by {cols})"
        
        try:
            # Create a pivot table: index vs. columns, with values as the numeric mean
            df_pivot = pd.pivot_table(
                data, 
                index=idx, 
                columns=cols, 
                values=num, 
                aggfunc='mean'
            )
            df_pivot = df_pivot.fillna(0).round(4)
            
            # Categories are the pivot table's index
            categories = df_pivot.index.astype(str).tolist()
            # Data is a dict of lists: {column_name: [val1, val2, ...]}
            chart_data = {str(col): df_pivot[col].tolist() for col in df_pivot.columns}

            if not categories:
                continue
                
            charts[chart_key] = {
                "title": chart_title,
                "x_axis_label": idx,
                "y_axis_label": f"Average {num}",
                "rotated": False,
                "data": chart_data,
                "categories": categories
            }
        except Exception as e:
            print(f"Failed to build {chart_key}: {e}")
    return charts

def _generate_pie_charts(data: pd.DataFrame, specs: List[Dict[str, str]]) -> Dict[str, Dict]:
    """Builds generic_pie_chart JSON objects from specs."""
    charts = {}
    for i, spec in enumerate(specs):
        c = spec['category']
        chart_key = f"pie_count_by_{c.lower()}_{i}"
        
        try:
            df_counts = data[c].value_counts()
            
            # The template format `{"data1": [1,2,3]}` has no labels.
            # We will provide labels in the title as a workaround.
            data_series = df_counts.values.tolist()
            label_series = df_counts.index.astype(str).tolist()
            chart_title = f"Record Count by {c} (Labels: {', '.join(label_series)})"

            charts[chart_key] = {
                "title": chart_title,
                "type": "pie",
                "data": {
                    "data1": data_series # Put the data in the "data1" series
                }
                # Note: We've added labels to the title as the template doesn't support them.
            }
        except Exception as e:
            print(f"Failed to build {chart_key}: {e}")
    return charts

def _generate_scatter_plots(data: pd.DataFrame, specs: List[Dict[str, str]], max_records: int) -> Dict[str, Dict]:
    """Builds generic_scatter_plot JSON objects from specs."""
    charts = {}
    # Sample the data to keep the plot manageable
    df_sample = data.sample(n=min(len(data), max_records))
    
    for i, spec in enumerate(specs):
        x, y = spec['x'], spec['y']
        chart_key = f"scatter_{x.lower()}_vs_{y.lower()}_{i}"
        chart_title = f"Scatter: {x} vs {y} (Sampled)"
        
        try:
            # Format as: [[x1, y1], [x2, y2], ...]
            series_data = df_sample[[x, y]].round(4).values.tolist()
            
            charts[chart_key] = {
                "title": chart_title,
                "x_axis_label": x,
                "y_axis_label": y,
                "type": "scatter",
                "data": {
                    "data1": series_data # Put the data in the "data1" series
                }
            }
        except Exception as e:
            print(f"Failed to build {chart_key}: {e}")
    return charts


# --- Main Metrics Function ---

# modelop.metrics
def metrics(data: pd.DataFrame, chart_count: Dict[str, int] = None, chart_order: List[str] = None, show_dataset: bool = False) -> Generator[Dict[str, Any], None, None]:
    """
    The main metrics function, updated to dynamically generate charts with control over number and order.
    
    Args:
        data: The input DataFrame to analyze
        chart_count: Optional dictionary specifying the number of charts to include for each type
                     e.g., {'time_line': 2, 'bar': 3, 'pie': 1, 'scatter': 2}
        chart_order: Optional list of specific chart keys in desired display order
                    e.g., ['bar_value_by_category_0', 'timeline_sales_by_region_1']
    """
    print("Running the metrics function") 
    print(f"Input data shape: {data.shape}")
    
    # --- 1. Basic Metrics ---
    all_results: Dict[str, Any] = {}
    all_results["rows"] = len(data)
    # Provide a sample of the table, not the whole thing
    all_results["table"] = data.head(100).to_dict(orient='records')

    # --- 2. Data Preprocessing ---
    data_clean = data.copy()
    
    # Convert 'Week_Ending' to datetime, which is crucial for time-series
    if 'Week_Ending' in data_clean.columns:
        print("Preprocessing 'Week_Ending' column to datetime...")
        data_clean['Week_Ending'] = pd.to_datetime(data_clean['Week_Ending'], errors='coerce')
        data_clean = data_clean.dropna(subset=['Week_Ending'])
    else:
        print("Warning: 'Week_Ending' column not found. Time-series charts will be skipped.")

    # --- 3. Get Monitor Specifications ---
    monitor_specs = get_available_monitors(data_clean)

    # --- 4. Generate All Dynamic Monitors ---
    
    # Generate all charts first
    print("Generating time-line charts...")
    # combine the time-series related spec buckets
    time_specs = (
        monitor_specs.get('time_line_graph', [])
        + monitor_specs.get('generic_line_graph', [])
        + monitor_specs.get('decimal_line_graph', [])
    )
    time_line_charts = _generate_time_line_charts(data_clean, time_specs)
    
    print("Generating bar charts...")
    bar_specs = (
        monitor_specs.get('generic_bar_graph', [])
        + monitor_specs.get('horizontal_bar_graph', [])
    )
    bar_charts = _generate_bar_charts(data_clean, bar_specs)
    
    print("Generating pie charts...")
    pie_specs = (
        monitor_specs.get('generic_pie_chart', [])
        + monitor_specs.get('generic_donut_chart', [])
    )
    pie_charts = _generate_pie_charts(data_clean, pie_specs)
    
    print("Generating scatter plots...")
    scatter_charts = _generate_scatter_plots(data_clean, monitor_specs.get('generic_scatter_plot', []), MAX_RECORDS_FOR_SCATTER)
    
    # Group charts by type
    chart_groups = {
        'time_line': {k: v for k, v in time_line_charts.items()},
        'bar': {k: v for k, v in bar_charts.items()},
        'pie': {k: v for k, v in pie_charts.items()},
        'scatter': {k: v for k, v in scatter_charts.items()}
    }
    
    # First, collect all available charts
    all_available_charts = {**time_line_charts, **bar_charts, **pie_charts, **scatter_charts}
    
    # Filter charts based on order or limits
    if chart_order:
        # Resolve requested chart_order entries to actual chart keys in all_available_charts.
        # Accept exact matches, prefix matches, or substring matches to be more forgiving
        resolved_keys: List[str] = []
        for req in chart_order:
            if req in all_available_charts:
                resolved_keys.append(req)
                continue
            # prefer keys that start with the requested string
            matches = [k for k in all_available_charts.keys() if k.startswith(req)]
            if not matches:
                # try substring match (case-insensitive)
                matches = [k for k in all_available_charts.keys() if req.lower() in k.lower()]
            if matches:
                resolved_keys.append(matches[0])
            else:
                print(f"Warning: requested chart_order entry '{req}' not found in available charts.")

        # Deduplicate while preserving order and ensure keys exist
        seen = set()
        valid_chart_order: List[str] = []
        for k in resolved_keys:
            if k in all_available_charts and k not in seen:
                valid_chart_order.append(k)
                seen.add(k)

        # If nothing matched, fall back to including all available charts (keeps previous behavior)
        if not valid_chart_order:
            print("No valid chart_order entries matched available charts; falling back to including all charts.")
            valid_chart_order = list(all_available_charts.keys())
    
        # Create type-specific ordered lists while preserving order
        ordered_charts = {
            'time_line': [],
            'bar': [],
            'pie': [],
            'scatter': []
        }
    
        # Organize charts by type while preserving order
        for chart_key in valid_chart_order:
            if chart_key.startswith('timeline_'):
                ordered_charts['time_line'].append((chart_key, all_available_charts[chart_key]))
            elif chart_key.startswith('bar_'):
                ordered_charts['bar'].append((chart_key, all_available_charts[chart_key]))
            elif chart_key.startswith('pie_'):
                ordered_charts['pie'].append((chart_key, all_available_charts[chart_key]))
            elif chart_key.startswith('scatter_'):
                ordered_charts['scatter'].append((chart_key, all_available_charts[chart_key]))
    
        # Convert to the format needed for standardized mapping
        chart_groups = {
            chart_type: dict(charts) for chart_type, charts in ordered_charts.items()
        }
    
        # Keep filtered_charts for reference
        filtered_charts = {k: all_available_charts[k] for k in valid_chart_order}
    
    elif chart_count:
        # Apply limits only to chart types explicitly requested in chart_count.
        # Previously we included all chart types and applied limits only where provided
        # which could return chart types the user didn't request. Now we restrict
        # output to the types present in chart_count (more explicit control).
        filtered_charts = {}
        for chart_type, charts in chart_groups.items():
            if chart_type not in chart_count:
                # skip chart types not requested
                continue
            limit = chart_count.get(chart_type, len(charts))
            filtered_charts.update(
                dict(list(charts.items())[:limit])
            )
    else:
        # Include all charts if no filtering is specified
        filtered_charts = all_available_charts
    
    # Initialize standardized output format. We'll add a 1-row executive scorecard
    # at the start of the report and only include the full dataset table at the
    # end if show_dataset is True.
    standardized_results: Dict[str, Any] = {}

    # --- Executive 1-row scorecard (executive summary) ---
    # Compute prediction accuracy and trends if prediction columns exist
    scorecard = {
        "overall_accuracy": None,
        "weekly_change": None,
        "mom_change": None,
        "ttm_change": None,
        "accuracy_by_account_type": {},
        "counts_by_account_type": {},
    }

    try:
        df = data_clean.copy()
        # ensure Week_Ending is datetime if present
        if 'Week_Ending' in df.columns:
            df['Week_Ending'] = pd.to_datetime(df['Week_Ending'], errors='coerce')
            df = df.dropna(subset=['Week_Ending'])

        if 'Predicted_Performance' in df.columns and 'Actual_Performance' in df.columns:
            df['correct_pred'] = (df['Predicted_Performance'] == df['Actual_Performance']).astype(int)

            # overall accuracy
            overall_acc = float(df['correct_pred'].mean()) if len(df) > 0 else None
            scorecard['overall_accuracy'] = round(overall_acc, 4) if overall_acc is not None else None

            # weekly accuracy trend (compare last week vs previous week)
            if 'Week_Ending' in df.columns and df['Week_Ending'].notna().any():
                weekly = df.groupby(df['Week_Ending'].dt.to_period('W')).agg({'correct_pred': 'mean'}).sort_index()
                weekly_vals = weekly['correct_pred'].values
                if len(weekly_vals) >= 2:
                    scorecard['weekly_change'] = round(float(weekly_vals[-1] - weekly_vals[-2]), 4)
                else:
                    scorecard['weekly_change'] = None

                # month-over-month: use month periods
                monthly = df.groupby(df['Week_Ending'].dt.to_period('M')).agg({'correct_pred': 'mean'}).sort_index()
                mvals = monthly['correct_pred'].values
                if len(mvals) >= 2:
                    scorecard['mom_change'] = round(float(mvals[-1] - mvals[-2]), 4)
                else:
                    scorecard['mom_change'] = None

                # TTM: trailing 12-months change compared to previous 12-months
                # convert to monthly index to make rolling windows easier
                if len(monthly) >= 24:
                    last_12 = monthly['correct_pred'][-12:].mean()
                    prev_12 = monthly['correct_pred'][-24:-12].mean()
                    scorecard['ttm_change'] = round(float(last_12 - prev_12), 4)
                else:
                    scorecard['ttm_change'] = None
            else:
                scorecard['weekly_change'] = None
                scorecard['mom_change'] = None
                scorecard['ttm_change'] = None

            # accuracy by Account_Type
            if 'Account_Type' in df.columns:
                grp = df.groupby('Account_Type').agg({'correct_pred': ['mean', 'count']})
                for acct, row in grp.iterrows():
                    acc = float(row[('correct_pred', 'mean')])
                    cnt = int(row[('correct_pred', 'count')])
                    scorecard['accuracy_by_account_type'][str(acct)] = round(acc, 4)
                    scorecard['counts_by_account_type'][str(acct)] = cnt

        else:
            # prediction columns missing; leave scorecard values as None/empty
            pass
    except Exception as e:
        print(f"Warning: failed to compute scorecard metrics: {e}")

    # Insert scorecard and row count at the beginning of the results
    standardized_results['rows'] = all_results.get('rows')
    standardized_results['scorecard'] = [scorecard]

    # Process time series / line charts
    time_line_values = [v for k, v in filtered_charts.items() if k.startswith('timeline_')]
    if time_line_values:
        # Time series data is already in correct format with timestamps
        standardized_results["time_line_graph"] = {
            **time_line_values[0],
            "title": time_line_values[0].get("title", "Time Series Data"),
            "x_axis_label": time_line_values[0].get("x_axis_label", "Time"),
            "y_axis_label": time_line_values[0].get("y_axis_label", "Value")
        }
        
        # Map additional line charts if available
        if len(time_line_values) > 1:
            standardized_results["generic_line_graph"] = {
                **time_line_values[1],
                "title": time_line_values[1].get("title", "Line Graph - XY Data"),
                "x_axis_label": time_line_values[1].get("x_axis_label", "X Axis"),
                "y_axis_label": time_line_values[1].get("y_axis_label", "Y Axis")
            }
        
        if len(time_line_values) > 2:
            standardized_results["decimal_line_graph"] = {
                **time_line_values[2],
                "title": time_line_values[2].get("title", "Line Graph - Decimal Data"),
                "x_axis_label": time_line_values[2].get("x_axis_label", "X Axis"),
                "y_axis_label": time_line_values[2].get("y_axis_label", "Y Axis")
            }

    # Process bar charts
    bar_values = [v for k, v in filtered_charts.items() if k.startswith('bar_')]
    if bar_values:
            # Map each bar chart to the appropriate type based on order
            for i, bar_chart in enumerate(bar_values):
                if i == 0:
                    # First bar chart is always vertical (generic_bar_graph)
                    standardized_results["generic_bar_graph"] = {
                        "title": bar_chart.get("title", "Bar Chart"),
                        "x_axis_label": bar_chart.get("x_axis_label", "X Axis"),
                        "y_axis_label": bar_chart.get("y_axis_label", "Y Axis"),
                        "rotated": False,
                        "data": bar_chart.get("data", {}),
                        "categories": bar_chart.get("categories", [])
                    }
                elif i == 1:
                    # Second bar chart is always horizontal
                    standardized_results["horizontal_bar_graph"] = {
                        "title": bar_chart.get("title", "Horizontal Bar Chart"),
                        "x_axis_label": bar_chart.get("x_axis_label", "X Axis"),
                        "y_axis_label": bar_chart.get("y_axis_label", "Y Axis"),
                        "rotated": True,
                        "data": bar_chart.get("data", {}),
                        "categories": bar_chart.get("categories", [])
                    }

    # Process scatter plots
    scatter_values = [v for k, v in filtered_charts.items() if k.startswith('scatter_')]
    if scatter_values:
        standardized_results["generic_scatter_plot"] = {
            "title": scatter_values[0].get("title", "Scatter Plot"),
            "x_axis_label": scatter_values[0].get("x_axis_label", "X Axis"),
            "y_axis_label": scatter_values[0].get("y_axis_label", "Y Axis"),
            "type": "scatter",
            "data": scatter_values[0].get("data", {})
        }

    # Process pie/donut charts
    pie_values = [v for k, v in filtered_charts.items() if k.startswith('pie_')]
    if pie_values:
        # Standard pie chart
        standardized_results["generic_pie_chart"] = {
            "title": pie_values[0].get("title", "Pie Chart"),
            "type": "pie",
            "data": pie_values[0].get("data", {})
        }

        # Donut chart (if available)
        if len(pie_values) > 1:
            standardized_results["generic_donut_chart"] = {
                "title": pie_values[1].get("title", "Donut Chart"),
                "type": "donut",
                "data": pie_values[1].get("data", {})
            }
    
    # Optionally include the full dataset table as the last item if requested
    if show_dataset:
        standardized_results['table'] = all_results.get('table')

    print(f"Metrics function complete. Returning {len(standardized_results)} total keys.")
    yield standardized_results # the final json object 

# For local testing
def main():
    raw_json = Path('example_job.json').read_text()
    init_param = {'rawJson': raw_json}
    init(init_param)
    data = {"data1": 993, "data2": 36, "data3": 3959, "label_value": 0, "score": 1}
    data = pd.DataFrame.from_dict([data])
    print(json.dumps(next(metrics(data)), indent=2))


if __name__ == '__main__':
    # main()
    # Uncomment for local testing with different chart filtering options:
    data = pd.read_csv("synthetic_data/synthetic_mlops_financial_data.csv")
    
    # Example 1: Limit number of each chart type
    chart_count = {
        # 'time_line': 1,
        # 'bar': 1,
        'pie': 1,
        # 'scatter': 1
    }
    metrics_generator = metrics(data, chart_count=chart_count)
    print("\nExample 1 - Limited number of charts:")
    print(json.dumps(next(metrics_generator), indent=2))
    
    # # Example 2: Specific chart order - single bar chart
    # metrics_generator = metrics(data, chart_order=['bar_market_value_by_account_id_0'])
    # print("\nExample 2a - Single bar chart:")
    # print(json.dumps(next(metrics_generator), indent=2))
    
    # # Example 2b: Specific chart order - multiple charts in specific order
    # chart_order = [
    #     'bar_market_value_by_account_id_0',  # Will be mapped to generic_bar_graph
    #     'bar_return_by_account_id_1',        # Will be mapped to horizontal_bar_graph
    #     'timeline_balance_by_region_1',      # Will be mapped to time_line_graph
    #     'pie_count_by_status_0'             # Will be mapped to generic_pie_chart
    # ]
    # metrics_generator = metrics(data, chart_order=chart_order)
    # print("\nExample 2b - Multiple charts in specific order:")
    # print(json.dumps(next(metrics_generator), indent=2))
    # # print("\nExample 2b - Multiple specific charts:")
    # # print(json.dumps(next(metrics_generator), indent=2))
    