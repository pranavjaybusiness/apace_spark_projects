from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def engineer_features(self):
        indexers = [
            StringIndexer(inputCol=column, outputCol=column+"_index").fit(self.data)
            for column in ["UniqueCarrier", "Origin", "Dest"]
        ]

        encoders = [
            OneHotEncoder(inputCol=column+"_index", outputCol= column+"_vec")
            for column in ["UniqueCarrier", "Origin", "Dest"]
        ]

        assembler = VectorAssembler(
            inputCols=["UniqueCarrier_vec", "Origin_vec", "Dest_vec", "DepTime", "CRSDepTime", "CRSArrTime", "Distance"],
            outputCol="features"
        )

        pipeline = Pipeline(stages=indexers + encoders + [assembler])
        self.data = pipeline.fit(self.data).transform(self.data)

        return self.data
