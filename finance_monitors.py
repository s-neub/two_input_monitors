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


def _canonical_chart_type(name: str) -> str:
    """Map a user-friendly or legacy chart type key to the canonical chart_groups key.

    Accepts short names like 'pie', 'bar', 'time_line', 'scatter' and returns
    the canonical group name used elsewhere in this module.
    """
    if not name:
        return name
    key = name.lower()
    mapping = {
        # line graphs
        'time_line': 'time_line_graph',
        'timeline': 'time_line_graph',
        'time_line_graph': 'time_line_graph',
        'generic_line_graph': 'generic_line_graph',
        'decimal_line_graph': 'decimal_line_graph',
        'line': 'generic_line_graph',
        'd_line': 'decimal_line_graph',
        'decimal': 'decimal_line_graph',
        # bars
        'bar': 'generic_bar_graph',
        'generic_bar_graph': 'generic_bar_graph',
        'h_bar': 'horizontal_bar_graph',
        'horizontal_bar_graph': 'horizontal_bar_graph',
        'horizontal_bar': 'horizontal_bar_graph',
        # pies / donuts
        'pie': 'generic_pie_chart',
        'generic_pie_chart': 'generic_pie_chart',
        'donut': 'generic_donut_chart',
        'generic_donut_chart': 'generic_donut_chart',
        # scatter / table
        'scatter': 'generic_scatter_plot',
        'generic_scatter_plot': 'generic_scatter_plot',
        'table': 'generic_table',
        'generic_table': 'generic_table'
    }
    return mapping.get(key, name)


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

    # Spec: Donut Chart (sum of numeric values by Category) - use float/decimal numeric cols
    donut_specs: List[Dict[str, str]] = []
    for cat_col in categorical_cols:
        for num_col in true_numeric_cols:
            # only consider decimal/float columns for donut (currency-like)
            try:
                if np.issubdtype(data[num_col].dtype, np.floating) and data[num_col].nunique() > 1:
                    donut_specs.append({"category": cat_col, "numeric": num_col})
            except Exception:
                # ignore any unexpected dtype issues
                continue

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

    # Distribute bar specs into horizontal vs generic based on the number of
    # categories present for the index column. Use horizontal for <=8 bars,
    # generic (vertical) for >=9 bars. This mirrors the runtime selection logic.
    for s in bar_specs:
        idx_col = s.get('index')
        try:
            n_cats = int(data[idx_col].nunique()) if idx_col in data.columns else MAX_CATEGORICAL_GROUPS + 1
        except Exception:
            n_cats = MAX_CATEGORICAL_GROUPS + 1

        if n_cats <= 8:
            specs["horizontal_bar_graph"].append(s)
        else:
            specs["generic_bar_graph"].append(s)


    # Assign pie specs (counts) and donut specs (numeric breakdowns)
    for s in pie_specs:
        specs["generic_pie_chart"].append(s)
    for s in donut_specs:
        specs["generic_donut_chart"].append(s)

    # Scatter stays as generic_scatter_plot
    specs["generic_scatter_plot"] = scatter_specs

    # Use the actual keys created above. Avoid KeyError when keys don't exist.
    print(
        f"Generated {len(specs.get('time_line_graph', []))} time-line, "
        f"{len(specs.get('generic_line_graph', []))} generic-line, "
        f"{len(specs.get('decimal_line_graph', []))} decimal-line, "
        f"{len(specs.get('generic_bar_graph', []))} generic-bar, "
        f"{len(specs.get('horizontal_bar_graph', []))} horizontal-bar, "
        f"{len(specs.get('generic_pie_chart', []))} pie, "
        f"{len(specs.get('generic_donut_chart', []))} donut, "
        f"{len(specs.get('generic_scatter_plot', []))} scatter specs."
    )
    return specs


# --- Helper Functions to Build Charts from Specs ---

def _generate_line_graphs(data: pd.DataFrame, specs: List[Dict[str, str]]) -> Dict[str, Dict]:
    """Builds line graph JSON objects from specs.

    This generator produces the same payload as the previous _generate_time_line_charts
    but the name better reflects that these are line graphs (time-series or generic)
    and clarifies intent.
    """
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

            # Build a data mapping where each label becomes a series key so the
            # front-end can display labels correctly instead of relying on a
            # generic "data1" key. Each value is provided as a single-element
            # list to match the expected series format.
            data_dict: Dict[str, List[float]] = {}
            for label, val in df_counts.items():
                # convert to native types for JSON serialization
                try:
                    v = float(val)
                except Exception:
                    v = float(int(val)) if pd.notna(val) else 0.0
                data_dict[str(label)] = [round(v, 4)]

            chart_title = f"Record Count by {c}"

            charts[chart_key] = {
                "title": chart_title,
                "type": "pie",
                "data": data_dict
            }
        except Exception as e:
            print(f"Failed to build {chart_key}: {e}")
    return charts


def _generate_donut_charts(data: pd.DataFrame, specs: List[Dict[str, str]]) -> Dict[str, Dict]:
    """Builds generic_donut_chart JSON objects from specs (numeric breakdown by category)."""
    charts = {}
    for i, spec in enumerate(specs):
        c = spec.get('category')
        num = spec.get('numeric')
        # Use a donut_ prefix so callers can distinguish donut charts from pie-count charts
        chart_key = f"donut_{num.lower()}_by_{c.lower()}_{i}"

        try:
            # Aggregate numeric by category (sum) and present as donut proportions
            df_agg = data.groupby(c)[num].sum().sort_values(ascending=False)
            data_dict: Dict[str, List[float]] = {}
            for label, val in df_agg.items():
                try:
                    v = float(val)
                except Exception:
                    v = 0.0
                data_dict[str(label)] = [round(v, 4)]

            chart_title = f"{num} by {c} (Donut)"

            charts[chart_key] = {
                "title": chart_title,
                "type": "donut",
                "data": data_dict
            }
        except Exception as e:
            print(f"Failed to build donut {chart_key}: {e}")
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
    print("Generating line graphs...")
    # Generate line graphs (time-series) separately for each time-related spec bucket
    time_line_charts = _generate_line_graphs(data_clean, monitor_specs.get('time_line_graph', []))
    generic_line_charts = _generate_line_graphs(data_clean, monitor_specs.get('generic_line_graph', []))
    decimal_line_charts = _generate_line_graphs(data_clean, monitor_specs.get('decimal_line_graph', []))
    
    print("Generating bar charts...")
    bar_specs = (
        monitor_specs.get('generic_bar_graph', [])
        + monitor_specs.get('horizontal_bar_graph', [])
    )
    bar_charts = _generate_bar_charts(data_clean, bar_specs)
    
    print("Generating pie charts (counts) and donut charts (numeric breakdowns)...")
    pie_specs = monitor_specs.get('generic_pie_chart', [])
    donut_specs = monitor_specs.get('generic_donut_chart', [])
    pie_charts = _generate_pie_charts(data_clean, pie_specs)
    donut_charts = _generate_donut_charts(data_clean, donut_specs)
    
    print("Generating scatter plots...")
    scatter_charts = _generate_scatter_plots(data_clean, monitor_specs.get('generic_scatter_plot', []), MAX_RECORDS_FOR_SCATTER)
    
    # Group charts by type
    # Split bar charts into generic vs horizontal based on category counts
    generic_bar_charts: Dict[str, Dict] = {}
    horizontal_bar_charts: Dict[str, Dict] = {}
    for k, v in bar_charts.items():
        n_cats = len(v.get('categories', []) or [])
        if n_cats <= 8:
            horizontal_bar_charts[k] = v
        else:
            generic_bar_charts[k] = v

    # Keep the time-related chart groups separate (they may be empty)
    chart_groups = {
        'time_line_graph': {k: v for k, v in time_line_charts.items()},
        'generic_line_graph': {k: v for k, v in generic_line_charts.items()},
        'decimal_line_graph': {k: v for k, v in decimal_line_charts.items()},
        'generic_bar_graph': generic_bar_charts,
        'horizontal_bar_graph': horizontal_bar_charts,
        'generic_scatter_plot': {k: v for k, v in scatter_charts.items()},
        'generic_pie_chart': {k: v for k, v in pie_charts.items()},
        'generic_donut_chart': {k: v for k, v in donut_charts.items()},
        'generic_table': {"generic_table": {"data": all_results.get('table')}}
    }

    # First, collect all available charts into a single lookup
    all_available_charts = {}
    for grp in ['time_line_graph', 'generic_line_graph', 'decimal_line_graph', 'generic_bar_graph', 'horizontal_bar_graph', 'generic_pie_chart', 'generic_donut_chart', 'generic_scatter_plot', 'generic_table']:
        all_available_charts.update(chart_groups.get(grp, {}))
    
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
            'donut': [],
            'scatter': [],
            'table': []
        }
    
        # Organize charts by type while preserving order (map key prefixes to new group names)
        for chart_key in valid_chart_order:
            if chart_key.startswith('timeline_'):
                ordered_charts['time_line'].append((chart_key, all_available_charts[chart_key]))
            elif chart_key.startswith('bar_'):
                ordered_charts['bar'].append((chart_key, all_available_charts[chart_key]))
            elif chart_key.startswith('donut_'):
                ordered_charts['donut'].append((chart_key, all_available_charts[chart_key]))
            elif chart_key.startswith('pie_'):
                ordered_charts['pie'].append((chart_key, all_available_charts[chart_key]))
            elif chart_key.startswith('scatter_'):
                ordered_charts['scatter'].append((chart_key, all_available_charts[chart_key]))
            elif chart_key == 'generic_table':
                ordered_charts['table'].append((chart_key, all_available_charts[chart_key]))

        # Convert ordered lists to dicts, then map to the final group names used downstream
        ordered_dicts = {k: dict(v) for k, v in ordered_charts.items()}

        # classify ordered bar charts into horizontal vs generic based on categories
        od_bar = ordered_dicts.get('bar', {})
        od_generic_bar = {}
        od_horizontal_bar = {}
        for k, v in od_bar.items():
            n_cats = len(v.get('categories', []) or [])
            if n_cats <= 8:
                od_horizontal_bar[k] = v
            else:
                od_generic_bar[k] = v

        chart_groups = {
            'time_line_graph': ordered_dicts.get('time_line', {}),
            'generic_line_graph': {},
            'decimal_line_graph': {},
            'generic_bar_graph': od_generic_bar,
            'horizontal_bar_graph': od_horizontal_bar,
            'generic_scatter_plot': ordered_dicts.get('scatter', {}),
            'generic_pie_chart': ordered_dicts.get('pie', {}),
            'generic_donut_chart': ordered_dicts.get('donut', {}),
            'generic_table': ordered_dicts.get('table', {})
        }
    
        # Keep filtered_charts for reference
        filtered_charts = {k: all_available_charts[k] for k in valid_chart_order}
    
    elif chart_count:
        # Apply limits only to chart types explicitly requested in chart_count.
        # Previously we included all chart types and applied limits only where provided
        # which could return chart types the user didn't request. Now we restrict
        # output to the types present in chart_count (more explicit control).
        # Normalize user-provided keys to canonical group names so callers can
        # use short names like 'pie' or 'bar'.
        normalized_counts: Dict[str, int] = {}
        for user_key, limit in chart_count.items():
            canon = _canonical_chart_type(user_key)
            if canon in chart_groups:
                normalized_counts[canon] = int(limit)
            else:
                print(f"Warning: requested chart_count key '{user_key}' (normalized '{canon}') not recognized; skipping.")

        filtered_charts = {}
        for chart_type, charts in chart_groups.items():
            if chart_type not in normalized_counts:
                # skip chart types not requested
                continue
            limit = normalized_counts.get(chart_type, len(charts))
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
        # For each candidate bar chart, choose horizontal vs generic based on number
        # of categories: <=8 -> horizontal (rotated=True), >=9 -> generic (rotated=False).
        # Also sort bars by descending magnitude (sum across series) before emitting.
        assigned_generic = False
        assigned_horizontal = False
        for i, bar_chart in enumerate(bar_values):
            categories = bar_chart.get('categories', []) or []
            data_series = bar_chart.get('data', {}) or {}

            # compute total magnitude per category (sum across series)
            try:
                n_cats = len(categories)
                totals = [0.0] * n_cats
                for series in data_series.values():
                    for idx, val in enumerate(series):
                        try:
                            totals[idx] += float(val)
                        except Exception:
                            totals[idx] += 0.0
                # sort indices by totals descending
                sort_idx = sorted(range(n_cats), key=lambda j: totals[j], reverse=True)
                # reorder categories and series accordingly
                sorted_categories = [categories[j] for j in sort_idx]
                sorted_data = {k: [v[j] for j in sort_idx] for k, v in data_series.items()}
            except Exception:
                # fallback: leave as-is
                sorted_categories = categories
                sorted_data = data_series

            # choose chart orientation
            if len(sorted_categories) <= 8 and not assigned_horizontal:
                standardized_results["horizontal_bar_graph"] = {
                    "title": bar_chart.get("title", "Horizontal Bar Chart"),
                    "x_axis_label": bar_chart.get("x_axis_label", "X Axis"),
                    "y_axis_label": bar_chart.get("y_axis_label", "Y Axis"),
                    "rotated": True,
                    "data": sorted_data,
                    "categories": sorted_categories
                }
                assigned_horizontal = True
            elif not assigned_generic:
                standardized_results["generic_bar_graph"] = {
                    "title": bar_chart.get("title", "Bar Chart"),
                    "x_axis_label": bar_chart.get("x_axis_label", "X Axis"),
                    "y_axis_label": bar_chart.get("y_axis_label", "Y Axis"),
                    "rotated": False,
                    "data": sorted_data,
                    "categories": sorted_categories
                }
                assigned_generic = True
            # stop if we have filled both slots
            if assigned_generic and assigned_horizontal:
                break

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

    # Process pie and donut charts separately
    pie_values = [v for k, v in filtered_charts.items() if k.startswith('pie_')]
    donut_values = [v for k, v in filtered_charts.items() if k.startswith('donut_')]

    if pie_values:
        standardized_results["generic_pie_chart"] = {
            "title": pie_values[0].get("title", "Pie Chart"),
            "type": "pie",
            "data": pie_values[0].get("data", {})
        }

    if donut_values:
        standardized_results["generic_donut_chart"] = {
            "title": donut_values[0].get("title", "Donut Chart"),
            "type": "donut",
            "data": donut_values[0].get("data", {})
        }
    
    # Optionally include the full dataset table as the last item if requested
    if show_dataset:
        standardized_results['table'] = all_results.get('table')

    # If the caller explicitly requested the generic_table as part of chart_count/chart_order,
    # include it in the output (it will contain the same table payload).
    if 'generic_table' in filtered_charts:
        standardized_results['generic_table'] = filtered_charts.get('generic_table')

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
    
    # # Example 1: Limit number of each chart type
    # chart_count = {
    #     # 'time_line': 1,
    #     # 'line': 1,
    #     # 'd_line': 1,
    #     # 'bar': 1,
    #     # 'h_bar': 1,
    #     # 'scatter': 1,
    #     # 'pie': 1,
    #     'donut': 1
    # }
    # metrics_generator = metrics(data, chart_count=chart_count)
    # print("\nExample 1 - Limited number of charts:")
    # print(json.dumps(next(metrics_generator), indent=2))
    
    # Example 2: Specific chart order - single bar chart
    metrics_generator = metrics(data, chart_order=['bar_market_value_by_account_id_0'])
    print("\nExample 2a - Single bar chart:")
    print(json.dumps(next(metrics_generator), indent=2))
    
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
    