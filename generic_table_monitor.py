import json
import pandas as pd
from pathlib import Path
import numpy as np
import modelop.utils as utils
import modelop_sdk.restclient.moc_client as moc_client




# modelop.init
def init(init_param):
    job = json.loads(init_param["rawJson"])


#modelop.metrics
def metrics(data: pd.DataFrame):
    print("Running the metrics function")
    print(data.columns)
    # --- Original Metrics ---
    no_of_rows = len(data)
    table = data.to_dict(orient='records')
    # --- Define Monitor Variables ---
    time_line_graph = {
        "title" : "Example Line Graph - Timeseries Data",
        "x_axis_label": "X Axis",
        "y_axis_label": "Y Axis",
        "data": {
            "data1": [["2023-02-27T20:10:20",100], ["2023-03-01T20:10:20",200], ["2023-03-03T20:10:20", 300]],
            "data2": [["2023-02-28T20:10:20", 350], ["2023-03-02T20:10:20", 250], ["2023-03-04T20:10:20", 150]]
        }
    }
    generic_line_graph = {
        "title" : "Example Line Graph - XY Data",
        "x_axis_label": "X Axis",
        "y_axis_label": "Y Axis",
        "data": {
          "data1": [[1,100], [3,200], [5, 300]],
          "data2": [[2, 350], [4, 250], [6, 150]]
        }
    }
    decimal_line_graph = {
        "title" : "Example Line Graph - Decimal Data",
        "x_axis_label": "X Axis",
        "y_axis_label": "Y Axis",
        "data": {
          "data1": [[1,1.23], [3,2.456], [5, 3.1415]],
          "data2": [[2, 4.75], [4, 2.987], [6, 1.375]]
        }
    }
    generic_bar_graph = {
        "title" : "Example Bar Chart",
        "x_axis_label": "X Axis",
        "y_axis_label": "Y Axis",
        "rotated": False, # Note: Converted from 'false' to Python's 'False'
        "data" : {
          "data1": [1, 2, 3, 4],
          "data2": [4, 3, 2, 1]
        },
        "categories": ["cat1", "cat2", "cat3", "cat4"]
    }
    horizontal_bar_graph = {
        "title" : "Example Bar Chart",
        "x_axis_label": "X Axis",
        "y_axis_label": "Y Axis",
        "rotated": True, # Note: Converted from 'true' to Python's 'True'
        "data" : {
          "data1": [1, 2, 3, 4],
          "data2": [4, 3, 2, 1]
        },
        "categories": ["cat1", "cat2", "cat3", "cat4"]
    }
    generic_scatter_plot = {
        "title": "Example Scatter Plot",
        "x_axis_label": "X Axis",
        "y_axis_label": "Y Axis",
        "type": "scatter",
        "data": {
            "data1": [[1,100], [3, 200], [5, 300], [2,101], [1,105], [1, 320], [2,90], [2,85], [6, 300]],
            "data2": [[2, 350], [4, 250],[6, 150],[1,101], [1,125], [1, 300], [4, 90], [4, 85], [4, 300]]
        }
    }
    generic_pie_chart = {
        "title": "Example Pie Chart",
        "type": "pie",
        "data": {
            "data1": [1, 2, 3, 4],
            "data2": [4, 3, 2, 1],
            "data3": [2, 1],
            "data4": [1]
        }
    }
    generic_donut_chart = {
        "title": "Example Donut Chart",
        "type": "donut",
        "data": {
            "data1": [1, 2, 3, 4],
            "data2": [4, 3, 2, 1],
            "data3": [2, 1],
            "data4": [1]
        }
    }
    # --- Combine all metrics into the final results dictionary ---
    results = {
        # Original metrics
        "rows": no_of_rows,
        "table": table,
        # Added monitors
        "time_line_graph": time_line_graph,
        "generic_line_graph": generic_line_graph,
        "decimal_line_graph": decimal_line_graph,
        "generic_bar_graph": generic_bar_graph,
        "horizontal_bar_graph": horizontal_bar_graph,
        "generic_scatter_plot": generic_scatter_plot,
        "generic_pie_chart": generic_pie_chart,
        "generic_donut_chart": generic_donut_chart
    }
    yield results # the final json object
    
"""def metrics(data: pd.DataFrame):
    
    print("Running the metrics function") 
    print(data.columns)
    no_of_rows=len(data) #return a string/number
    # table={} # See the documentation here to get the table structure: https://modelopdocs.atlassian.net/wiki/spaces/dv33/pages/2051900216/Monitor+Output+Structure
    table = data.to_dict(orient='records')
    ''' DESIRED OUTPUT:
        table=  "generic_table": [
            {"data1" : 1, "data2" : 2, "data3" : 3},
            {"data1" : 2, "data2" : 3, "data3": 4},
            {"data1" :  3, "data2" : 4, "data3" : 5}
	    ]   
    '''

    results={"rows":no_of_rows,"table":table} #this should display a number and a table in the model test results section.
    yield results # the final json object """
        
def main(): #ignore this part and just copy it as it is
    raw_json=Path('example_job.json').read_text()
    init_param={'rawJson':raw_json}
    init(init_param)
    data = {"data1":993,"data2":36,"data3":3959,"label_value":0,"score":1}
    data = pd.DataFrame.from_dict([data])
    print(json.dumps(next(metrics(data)), indent=2))


if __name__ == '__main__':
	main()   
    # Import csv file into pandas dataframe
    # data = pd.read_csv('table_3x3.csv')
    # print(json.dumps(next(metrics(data)), indent=2))