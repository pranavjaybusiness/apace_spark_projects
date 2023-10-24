from pyspark.sql import SparkSession
from feature_engineer import FeatureEngineer
from flight_delay_predictor import FlightDelayPredictor
from delay_classifier import DelayClassifier
from data_handler import DataHandler
class FlightDelayPipeline:

    def __init__(self, data_path):
        self.spark = SparkSession.builder \
            .appName("FlightDelayPrediction") \
            .config("spark.executor.heartbeatInterval", "100s") \
            .config("spark.network.timeout", "10000s") \
            .getOrCreate()

        print(f"Using Spark Version: {self.spark.version}")

        self.data_path = data_path
        self.data = None
        self.features = None
        self.model = None
        self.predictions = None

    def run(self):
        # Step 1: Load and preprocess the data
        # At the very beginning of your script, let's test Numpy

        header = ["Year","Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "ArrTime", "CRSArrTime",
                  "UniqueCarrier", "FlightNum", "TailNum", "ActualElapsedTime", "CRSElapsedTime", "AirTime",
                  "ArrDelay", "DepDelay", "Origin", "Dest", "Distance", "TaxiIn", "TaxiOut" , "Cancelled",
                  "CancellationCode", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
                  "LateAircraftDelay"]

        handler = DataHandler(self.spark, self.data_path, header)
        self.data = handler.data

        # Step 2: Feature Engineering
        engineer = FeatureEngineer(self.data)
        self.features = engineer.transform_features()

        # Step 3: Train the model
        predictor = FlightDelayPredictor(self.features)
        self.model = predictor.train_model()

        # Step 4: Predict and classify delays
        classifier = DelayClassifier(self.model, self.features)
        self.predictions = classifier.add_delay_classification(self.model.transform(self.features))

        # (Optional) Show results
        self.predictions.show()

    def stop_spark(self):
        self.spark.stop()

if __name__ == "__main__":
    DATA_PATH = "Data/delayed_flights.txt"
    pipeline = FlightDelayPipeline(DATA_PATH)
    pipeline.run()
    pipeline.stop_spark()
