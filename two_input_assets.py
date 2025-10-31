import pandas as pd
import logging
import json
# import modelop.utils as utils
# import modelop_sdk.restclient.moc_client as moc_client

# logger = utils.configure_logger()

#
# This method gets called when the monitor is loaded by the ModelOp runtime. It sets the GLOBAL values that are
# extracted from the report.txt to obtain the DTS and version info to append to the report
#

# modelop.init
def init(init_param):
	# no-op initializer for compatibility with modelop runtime
	pass

#
# This method is the modelops metrics method.  This is always called with a pandas dataframe that is arraylike, and
# contains individual rows represented in a dataframe format that is representative of all of the data that comes in
# as the results of the first input asset on the job.  This method will not be invoked until all data has been read
# from that input asset.
#
# For this example, we simply echo back the first row of that data as a json object.  This is useful for things like
# reading externally generated metrics from an SQL database or an S3 file and having them interpreted as a model test
# result for the association of these results with a model snapshot.
#
# data - The input data of the first input asset of the job, as a pandas dataframe
#

# modelop.metrics
def metrics(baseline: pd.DataFrame, comparator: pd.DataFrame):
	"""        
	baseline
		- 'synthetic_data/synthetic_mlops_financial_data.csv' convertted into a dataframe for baseline argument
		- time_line_graph (create a dataset with timestamp, data1, data2) using the week ending timestamp for the x axis & account type for the data1, data2, ..., dataN subkeys (according to chart_mapping.json)
	comparator
		- 'synthetic_data/average_weekly_market_value_by_asset_class_market_value.csv' convertted into a dataframe for comparator argument
		- a generic_table displaying the entire table (according to chart_mapping.json)
		- a generic_bar_graph using the account type for the data1, data2, ..., dataN subkeys & Alternative,Cash,Equity,Fixed Income,Grand Total for the category list (according to chart_mapping.json)
	"""
	# configure a basic logger to avoid dependency on modelop.utils during local runs
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)

	# Make local copies to avoid mutating the caller's frames
	b = baseline.copy() if baseline is not None else pd.DataFrame()
	c = comparator.copy() if comparator is not None else pd.DataFrame()

	# ---- Baseline: create time_line_graph data ----
	# Expecting columns: Week_Ending, Account_Type, Market_Value
	if 'Week_Ending' in b.columns and 'Account_Type' in b.columns and 'Market_Value' in b.columns:
		b['Week_Ending'] = pd.to_datetime(b['Week_Ending'])
		# aggregate market value by week and account type
		pivot = b.pivot_table(index='Week_Ending', columns='Account_Type', values='Market_Value', aggfunc='sum', fill_value=0)
		pivot = pivot.sort_index()
		time_data = {}
		for col in pivot.columns:
			acct_name = str(col)
			# build list of [iso_timestamp, numeric_value] with rounded values
			rows = [[ts.strftime('%Y-%m-%dT%H:%M:%S'), round(float(v)) if not pd.isna(v) else "" ] for ts, v in zip(pivot.index, pivot[col])]
			time_data[acct_name] = rows
		time_line_graph = {
			"title": "Baseline - Market Value by Account Type Over Time",
			"x_axis_label": "Week Ending",
			"y_axis_label": "Market Value",
			"data": time_data
		}
	else:
		time_line_graph = {"title": "Baseline - insufficient columns", "data": {}}

	# ---- Comparator: full table for generic_table ----
	# Convert comparator to list-of-dicts for easy JSON serialization
	# For table output: preserve strings but ensure numeric NaNs become ""
	c_table = c.copy()
	for col in c_table.columns:
		# attempt to coerce to numeric; if successful treat NaNs as ""
		conv = pd.to_numeric(c_table[col], errors='coerce')
		# if conversion produced any non-na values or original dtype looked numeric, treat column as numeric
		if conv.notna().any() or c_table[col].dtype.kind in 'iufc':
			# Round numeric values and convert NaN to ""
			c_table[col] = conv.apply(lambda x: round(x) if pd.notna(x) else "")
		else:
			c_table[col] = c_table[col].fillna('')
	baseline_table = c_table.to_dict(orient='records')

	# ---- Comparator: generic_bar_graph ----
	# We expect the comparator CSV to have Account_Type as first column and other columns for categories
	# Dynamically get categories from comparator columns, excluding Account_Type
	categories = [col for col in c.columns if col != 'Account_Type']
	bar_data = {}
	if 'Account_Type' in c.columns:
		# set Account_Type as index so we can iterate rows by account
		rows = c.set_index('Account_Type')
		# create a numeric-safe copy for categories: coerce category columns to numeric
		rows_num = rows.copy()
		for cat in categories:
			if cat in rows_num.columns:
				rows_num[cat] = pd.to_numeric(rows_num[cat], errors='coerce')
		
		for acct in rows_num.index:
			acct_name = str(acct)
			vals = []
			for cat in categories:
				# get value, round if numeric, or use "" for NaN
				val = rows_num.loc[acct][cat] if cat in rows_num.columns else 0
				vals.append(round(float(val)) if pd.notna(val) else "")
			bar_data[acct_name] = vals
		comparator_bar_graph = {
			"title": "Comparator - Asset Class Market Value by Account Type",
			"x_axis_label": "Asset Class",
			"y_axis_label": "Market Value",
			"rotated": False,
			"data": bar_data,
			"categories": categories
		}
	else:
		comparator_bar_graph = {"title": "Comparator - insufficient columns", "data": {}, "categories": categories}

	# yield the synthesized JSON-like dict with all requested representations
	yield {
		"baseline_row_count": len(b),
		"comparator_row_count": len(c),
		"baseline_time_line_graph_1": time_line_graph,
		"baseline_table_1": baseline_table,
		"comparator_bar_graph_1": comparator_bar_graph
	}

#
# This main method is utilized to simulate what the engine will do when calling the above metrics function.  It takes
# the json formatted data, and converts it to a pandas dataframe, then passes this into the metrics function for
# processing.  This is a good way to develop your models to be conformant with the engine in that you can run this
# locally first and ensure the python is behaving correctly before deploying on a ModelOp engine.
#
def main():
	# Need to adjust the path to point to local file with something like below, reading into the metrics function
	baseline_df = pd.read_csv('synthetic_data/synthetic_mlops_financial_data.csv')
	comparator_df = pd.read_csv('synthetic_data/average_weekly_market_value_by_asset_class_market_value.csv')
	# obtain metrics result and write to metrics_example_output.json as a list to match expected shape
	result = next(metrics(baseline_df, comparator_df))

	with open('metrics_example_output.json', 'w') as f:
		json.dump([result], f, indent=4)
	# also print the JSON for quick verification
	print(json.dumps([result], indent=4))

	# with open("example_model_test_results.json", "r") as f:
	# 	contents = f.read()
	# data_dict = json.loads(contents)
	# df = pd.DataFrame.from_dict([data_dict])
	# print(next(metrics(df)))


if __name__ == '__main__':
	main()