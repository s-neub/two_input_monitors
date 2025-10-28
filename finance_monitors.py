import pandas as pd
import numpy as np
from typing import Dict, Any, List, Generator

# --- Constants for Chart Generation ---
# Max unique categories for a "group by" or "column" variable to avoid cluttered charts
MAX_CATEGORICAL_GROUPS = 10
# Max unique categories for an "index" or "axis" variable
MAX_CATEGORICAL_CARDINALITY = 50
# Max records to plot for a scatter plot to avoid performance issues
MAX_RECORDS_FOR_SCATTER = 1000


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
    
    specs: Dict[str, List[Dict[str, str]]] = {
        "time_line": [],
        "bar": [],
        "pie": [],
        "scatter": []
    }

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
                    specs["time_line"].append({
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
                    specs["bar"].append({
                        "index": cat_index, 
                        "columns": cat_cols, 
                        "numeric": num_col
                    })

    # Spec: Pie Chart (Count by Category)
    for cat_col in categorical_cols:
        specs["pie"].append({"category": cat_col})

    # Spec: Scatter Plot (Numeric vs. Numeric)
    # Avoid duplicate plots (e.g., A vs B and B vs A)
    plotted_pairs = set()
    for x_col in true_numeric_cols:
        for y_col in true_numeric_cols:
            if x_col == y_col:
                continue
            if (y_col, x_col) not in plotted_pairs:
                specs["scatter"].append({"x": x_col, "y": y_col})
                plotted_pairs.add((x_col, y_col))

    print(f"Generated {len(specs['time_line'])} time-line, {len(specs['bar'])} bar, {len(specs['pie'])} pie, {len(specs['scatter'])} scatter specs.")
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

def metrics(data: pd.DataFrame) -> Generator[Dict[str, Any], None, None]:
    """
    The main metrics function, updated to dynamically generate all
    possible charts from the input data.
    """
    print("Running the metrics function") 
    print(f"Input data shape: {data.shape}")
    
    # --- 1. Basic Metrics ---
    results: Dict[str, Any] = {}
    results["rows"] = len(data)
    # Provide a sample of the table, not the whole thing
    results["table"] = data.head(100).to_dict(orient='records')

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
    
    # Each helper function returns a dictionary of charts, which we
    # add to the main results dictionary.
    
    print("Generating time-line charts...")
    results.update(
        _generate_time_line_charts(data_clean, monitor_specs.get('time_line', []))
    )
    
    print("Generating bar charts...")
    results.update(
        _generate_bar_charts(data_clean, monitor_specs.get('bar', []))
    )
    
    print("Generating pie charts...")
    results.update(
        _generate_pie_charts(data_clean, monitor_specs.get('pie', []))
    )
    
    print("Generating scatter plots...")
    results.update(
        _generate_scatter_plots(data_clean, monitor_specs.get('scatter', []), MAX_RECORDS_FOR_SCATTER)
    )

    print(f"Metrics function complete. Returning {len(results)} total keys.")
    yield results # the final json object 

# --- Example of how you would call this ---
if __name__ == "__main__":
    print("--- Loading Data for Example ---")
    try:
        # Load the example data file
        # This mimics the data being passed into the 'metrics' function
        input_data = pd.read_csv("synthetic_data/synthetic_mlops_financial_data.csv")
        
        print(f"Loaded {len(input_data)} rows.")
        
        # --- Running the Metrics Function ---
        # The 'metrics' function is a generator, so we consume it
        metrics_generator = metrics(input_data)
        final_output = next(metrics_generator)
        
        print("\n--- Function Execution Finished ---")
        print(f"\nTotal keys in final output: {len(final_output.keys())}")
        print("Example keys:", list(final_output.keys())[:10])
        
        # Write list to csv
        pd.DataFrame.from_dict(final_output['table']).to_csv("example_output_table_sample.csv", index=False)

        # Example of one generated chart
        example_bar_key = [k for k in final_output.keys() if k.startswith('bar_')][10]
        print(f"\n--- Example Generated Bar Chart ('{example_bar_key}') ---")
        import json
        print(final_output.keys())
        print(json.dumps(final_output['timeline_market_value_by_account_id_0'], indent=2))

    except FileNotFoundError:
        print("\nError: 'synthetic_mlops_financial_data.csv' not found.")
        print("Please run the data generation script from the previous prompt first.")
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")