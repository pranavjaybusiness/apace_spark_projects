from pyspark.ml.regression import LinearRegression

class FlightDelayPredictor:
    def __init__(self, data):
        self.data = data
        self.model = None

    def train_model(self):
        train_data, test_data = self.data.randomSplit([0.8, 0.2])
        lr = LinearRegression(featuresCol='features', labelCol='ArrDelay')
        self.model = lr.fit(train_data)
        return self.model

    def predict(self, test_data):
        return self.model.transform(test_data)
