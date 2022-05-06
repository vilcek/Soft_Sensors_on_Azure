# A Solution Template for Soft Sensor Modeling on Azure - Part 2

This second part focuses on how one could operationalize this solution template on Azure and potentially be able to scale by processing and modeling hundreds of sensors in parallel.

It is recommended that you follow the steps below to set up your environment on Azure for running the code:
- [create an Azure ML workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=azure-portal)
- [create an Azure Synapse workspace](https://docs.microsoft.com/en-us/azure/synapse-analytics/quickstart-create-workspace)
- [link your Azure Synapse workspace into your Azure ML workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-link-synapse-ml-workspaces)
- [create an Azure Anomaly Detector resource](https://docs.microsoft.com/en-us/azure/cognitive-services/anomaly-detector/how-to/deploy-anomaly-detection-on-container-instances#create-an-anomaly-detector-resource)

Before running the code, you first need to generate the raw data by running the [01_data_generation.ipynb](https://github.com/vilcek/Soft_Sensors_on_Azure/blob/main/standalone/01_data_generation.ipynb) notebook if you haven't done so. Then you need to create an Azure Storage container location named ‘softsensors-raw’, in the default Azure Blob Storage account for your Azure ML Workspace, and upload the raw data saved in [this folder](https://github.com/vilcek/Soft_Sensors_on_Azure/tree/main/standalone/data_generated) to that container. You also need to create a container named 'softsensors-models' in that Azure Blob Storage account.

After creating and configuring the resources above, we suggest you create a [Compute Instance on Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) on your Azure ML workspace to run the code.
