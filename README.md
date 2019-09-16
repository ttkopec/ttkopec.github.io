# hyperflow-report

Collection of scripts for reports generation related to [HyperFlow](https://github.com/hyperflow-wms/hyperflow), which 
aid developers in scientific workflows analysis and better understanding of computations carried out.

## Requirements

```bash
pip3 install -r requirements.txt
```

## Overview

Project consists of following files:

* [graph.py](graph.py) - graphing script, allows for graph generation. Example usage:
    * Edit `graph.py` and add invocation of graph function, for example:
      
    ```python
    if __name__ == '__main__':
        box_mean(
            'example-dataset/hflow_task.csv',
            'name,tags,time,configId,containerID,download_end,download_start,end,execute_end,execute_start,experiment,'
            'start,taskID,upload_end,upload_start'.split(','))
    ```
  
    * Run script
    
    ```bash
      python3 graph.py
    ```

* [get_influx.py](get_influx.py) - helper script for fetching data from **InfluxDB** HTTP API. Accepts following 
   variables:
   * `influx_host` - address of a host, where **InfluxDB** resides
   * `results` - name of results file
   * `experiment` - experiment date used for data filtering while querying **InfluxDB**
   * `table` - name of **InfluxDB** table
   
* [gen_html.py ](gen_html.py) - generator of static HTML file, which aggregates all useful graphs into one file. Usage:
  <br/>
  ```bash
  python3 gen_html.py [-h] diagnostic performance task
  ```
  where `diagnostic`, `performance` and `task` are paths to `.csv` files generated during workflow run.
  
  Command
  <br/>
  ```bash
  python3 gen_html.py example-dataset/hflow_diagnostic.csv example-dataset/hflow_performance.csv example-dataset/hflow_task.csv
  ```
  will generate `index.html` with embedded graphs.