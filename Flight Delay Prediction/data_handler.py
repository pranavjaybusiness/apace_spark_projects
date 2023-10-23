from pyspark.sql import SparkSession

class DataHandler:
    def __init__(self, filepath):
        self.data = SparkSession.builder.appName("FlightDelayPrediction").getOrCreate().read.csv(filepath, header=True, inferSchema=True)

    def clean_data(self):
        self.data = self.data.dropna()
        return self.data
