import json
import pandas as pd
from pathlib import Path
import numpy as np
# import modelop.utils as utils
# import modelop_sdk.restclient.moc_client as moc_client




# modelop.init
def init(init_param):
    job = json.loads(init_param["rawJson"])


#modelop.metrics
def metrics(data: pd.DataFrame):
    
    print("Running the metrics function") 
    print(data.columns)
    no_of_rows=len(data) #return a string/number
    # table={} # See the documentation here to get the table structure: https://modelopdocs.atlassian.net/wiki/spaces/dv33/pages/2051900216/Monitor+Output+Structure
    table = data.to_dict(orient='records')
    ''' 
    DESIRED OUTPUT:
        table=  "generic_table": [
            {"data1" : 1, "data2" : 2, "data3" : 3},
            {"data1" : 2, "data2" : 3, "data3": 4},
            {"data1" :  3, "data2" : 4, "data3" : 5}
	    ]   
    '''

    results={"rows":no_of_rows,"table":table} #this should display a number and a table in the model test results section.
    yield results # the final json object 
        
def main(): #ignore this part and just copy it as it is
    raw_json=Path('example_job.json').read_text()
    init_param={'rawJson':raw_json}
    init(init_param)
    data = {"data1":993,"data2":36,"data3":3959,"label_value":0,"score":1}
    data = pd.DataFrame.from_dict([data])
    print(json.dumps(next(metrics(data)), indent=2))


if __name__ == '__main__':
	# main()   
    # Import csv file into pandas dataframe
    data = pd.read_csv('table_3x3.csv')
    print(json.dumps(next(metrics(data)), indent=2))