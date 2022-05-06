# A Solution Template for Soft Sensor Modeling on Azure - Part 1

The first part goes through the main steps of the process for creating a softsensor:
- data preparation
- feature selection
- model training and validation
- model testing

Ideally, you should run it on a compute resource with a GPU available and in a Python 3 environment with the following packages installed: [azure-ai-anomalydetector](https://docs.microsoft.com/en-us/azure/cognitive-services/anomaly-detector/quickstarts/client-libraries) and [tsai](https://github.com/timeseriesAI/tsai). We suggest you run it on a [Compute Instance on Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance).

The code was prepared to be run as a sequence of Jupyter notebook files, in the sequence according to their file names in ascending order.