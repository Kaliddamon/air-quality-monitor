import numpy as np
import pandas as pd

from workshop4.mapreduce_parallel import testMapReduce

#mainpipelineforw4
def run_w4_from_records(records):

    worker_configs = [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (6, 2),
    ]

    chart_data = testMapReduce(records, worker_configs)
    return chart_data





