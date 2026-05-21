#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:03:13 2026

@author: jgiles

This scripts converts old scikit-learn 1.3.0 models saved in pickle files to ONNX.
It should work for any version of scikit-learn as long as this script is executed
using an environment with the same scikit-learn version as the model.
"""

import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Load your working pickle model one last time
with open('/automount/agradar/jgiles/riming_model/gbm_model_23.10.2024.pkl', 'rb') as f:
    riming_model = pickle.load(f)

with open('/automount/agradar/jgiles/riming_model/gbm_zh_zdr_model_23.10.2024.pkl', 'rb') as f:
    riming_model_zh_zdr = pickle.load(f)

# 2. Define the input type and shape
# [None, num_features] means it can accept any batch size, but must have exact number of features.
# Replace `num_features` with the actual number of features your model takes.
num_features = 3
initial_type = [('float_input', FloatTensorType([None, num_features]))]

num_features = 2
initial_type_zh_zdr = [('float_input', FloatTensorType([None, num_features]))]

# 3. Convert the scikit-learn model to an ONNX graph
riming_model_onnx = convert_sklearn(riming_model, initial_types=initial_type)
riming_model_zh_zdr_onnx = convert_sklearn(riming_model_zh_zdr, initial_types=initial_type_zh_zdr)

# 4. Save the ONNX model to disk
with open("/automount/agradar/jgiles/riming_model/gbm_model_23.10.2024.onnx", "wb") as f:
    f.write(riming_model_onnx.SerializeToString())

with open("/automount/agradar/jgiles/riming_model/gbm_zh_zdr_model_23.10.2024.onnx", "wb") as f:
    f.write(riming_model_zh_zdr_onnx.SerializeToString())