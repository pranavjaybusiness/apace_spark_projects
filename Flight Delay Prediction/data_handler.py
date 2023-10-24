from pyspark.sql import Row
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
class DataHandler:
    def __init__(self, spark_session, data_path, header):
        self.data_path = data_path
        self.header = header
        self.data = self.load_data(spark_session)

    def load_data(self, spark_session):
        # Load data using Spark's CSV reader
        data_df = spark_session.read.option("header", "true").csv(self.data_path)

        # Check if DataFrame is empty
        if data_df.rdd.isEmpty():
            print("The provided data file is empty!")
            return None
        data_df = data_df.select(data_df.columns[1:])
        # Perform data type conversion for relevant columns

        columns_to_cast = {
            "Year": "integer",
            "Month": "integer",
            "DayofMonth": "integer",
            "DayOfWeek": "integer",
            "DepTime": "double",
            "ArrTime": "double",
            "CRSArrTime": "double",
            "ActualElapsedTime": "double",
            "CRSElapsedTime": "double",
            "CRSDepTime": "double",
            "AirTime": "double",
            "Distance": "double",
            "TaxiIn": "double",
            "TaxiOut": "double"
        }

        for column, dtype in columns_to_cast.items():
            data_df = data_df.withColumn(column, F.col(column).cast(dtype))

        # Remove rows with null values
        data_df = data_df.na.drop()
        #print(data_df.show(15))  # This will print the first 5 rows of the DataFrame.
        return data_df
    @staticmethod
    def map_function(row, header):
        if len(row.split(',')) != len(header):
            return None
        return Row(**{header[i]: value for i, value in enumerate(row.split(','))})
