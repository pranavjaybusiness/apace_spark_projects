from pyspark.sql.functions import col, avg, stddev_pop
from pyspark.ml.feature import StringIndexer, VectorAssembler


class FeatureEngineer:

    def __init__(self, data):
        self.data = data
        self.categorical_columns = ["UniqueCarrier", "Origin", "Dest"]
        self.numerical_columns = [
            "Year", "Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime",
            "CRSArrTime", "ActualElapsedTime", "CRSElapsedTime", "AirTime", "ArrDelay",
            "DepDelay", "Distance", "TaxiIn", "TaxiOut", "Cancelled", "Diverted",
            "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"
        ]
        self.feature_columns = self.numerical_columns + [f"{col}_indexed" for col in self.categorical_columns]

    def convert_to_numeric(self):
        for col_name in ["ArrDelay", "DepDelay", "Cancelled", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay",
                         "SecurityDelay", "LateAircraftDelay"]:
            self.data = self.data.withColumn(col_name, col(col_name).cast("double"))
        return self.data

    def index_categorical_columns(self):
        for col in self.categorical_columns:
            indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed").fit(self.data)
            self.data = indexer.transform(self.data)
        return self.data

    def assemble_and_scale_features(self):
        assembler = VectorAssembler(inputCols=self.feature_columns, outputCol="features")
        self.data = assembler.transform(self.data)

        for col_name in self.numerical_columns:
            stats = \
            self.data.agg(avg(col(col_name)).alias("mean"), stddev_pop(col(col_name)).alias("stddev")).collect()[0]
            if stats["stddev"] != 0:
                self.data = self.data.withColumn(col_name, (col(col_name) - stats["mean"]) / stats["stddev"])
            else:
                self.data = self.data.withColumn(col_name, col(col_name) - stats["mean"])

        return self.data

    def transform_features(self):
        self.convert_to_numeric()
        self.index_categorical_columns()
        self.assemble_and_scale_features()
        return self.data
