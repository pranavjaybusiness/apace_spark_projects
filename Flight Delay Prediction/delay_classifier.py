from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

class DelayClassifier:
        def __init__(self, model, features):
            self.model = model
            self.features = features
        @staticmethod
        def classify_delay(delay):
            if delay <= 0:
                return "Early or On-Time"
            elif delay <= 15:
                return "Short Delay"
            elif delay <= 60:
                return "Medium Delay"
            else:
                return "Long Delay"

        @staticmethod
        def add_delay_classification(data):
            classify_udf = udf(DelayClassifier.classify_delay, StringType())
            print("Data before adding delay classification:")
            data.show()



            return data.withColumn("DelayClass", classify_udf(data["prediction"]))
