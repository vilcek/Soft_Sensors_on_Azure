from pyspark.sql import SparkSession
from mmlspark.cognitive import *
from pyspark.sql.functions import when
import argparse
import pandas as pd
import numpy as np
from azureml.core import Run

spark= SparkSession.builder.getOrCreate()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input_dataset', required=True)
arg_parser.add_argument('--output_dataset', required=True)
arg_parser.add_argument('--anomaly_key', required=True)
arg_parser.add_argument('--anomaly_service_location', required=True)
arg_parser.add_argument('--anomaly_max_data', required=True)
arg_parser.add_argument('--anomaly_min_data', required=True)

args, unknown_args = arg_parser.parse_known_args()

sdf_sensors = spark.read.parquet(args.input_dataset)
sdf_sensors = sdf_sensors.dropDuplicates()

sensors = sdf_sensors.select('sensor').distinct().rdd.map(lambda r: r[0]).collect()

data_length = sdf_sensors.groupBy('sensor').count().agg({'count': 'max'}).rdd.map(lambda r: r[0]).collect()[0]

idx = pd.date_range(start=0, end=(data_length - 1) * 60 * 1e9, freq='T')
sdf_sensors_prepared = spark.createDataFrame([[str(i)] for i in idx], ['Time'])

# data_length = len(idx)

num_batches = round(data_length / int(args.anomaly_max_data))

batches = []
for i in range(1, num_batches+1):
    batch = np.repeat(i, int(args.anomaly_max_data)).tolist()
    if i == num_batches:
        batch = np.repeat(i, data_length-len(batches)).tolist()
    batches = batches + batch
if len(batch) < int(args.anomaly_min_data):
    batches[-int(args.anomaly_min_data):] = np.repeat(i, int(args.anomaly_min_data)).tolist()

for sensor in sensors:
    sdf_sensor = sdf_sensors[sdf_sensors['sensor'] == sensor].orderBy('time')
    ts = pd.Series(sdf_sensor.toPandas()['value'])
    ts.index = idx
    ts = ts.interpolate(limit_direction='both')
    z = zip([str(i) for i in idx], [t for t in ts], batches)
    sdf = spark.createDataFrame([i for i in z], ['T', sensor, 'group'])

    anamoly_detector = (SimpleDetectAnomalies()
                    .setSubscriptionKey(args.anomaly_key)
                    .setLocation(args.anomaly_service_location)
                    .setTimestampCol('T')
                    .setValueCol(sensor)
                    .setOutputCol('anomalies')
                    .setGroupbyCol('group')
                    .setGranularity('minutely')
                    .setSensitivity(15))
    sdf = anamoly_detector.transform(sdf).select('T', sensor, 'anomalies.expectedValue', 'anomalies.isAnomaly')
    sdf = sdf.withColumn(sensor, when(sdf['isAnomaly']==True, np.nan).otherwise(sdf[sensor]))
    
    ts = pd.Series(sdf.toPandas()[sensor])
    ts.index = idx
    ts = ts.interpolate(limit_direction='both')
    z = zip([str(i) for i in idx], [t for t in ts])
    sdf = spark.createDataFrame([i for i in z], ['T', sensor])
    
    sdf_sensors_prepared = sdf_sensors_prepared.join(sdf, sdf_sensors_prepared.Time == sdf.T).drop('T')

sdf_sensors_prepared.write.parquet(args.output_dataset, mode='overwrite')