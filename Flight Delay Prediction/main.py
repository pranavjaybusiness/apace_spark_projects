# FlightDelayPipeline.py

from pyspark.sql import SparkSession
from data_handler import DataHandler
from feature_engineer import FeatureEngineer
from flight_delay_predictor import FlightDelayPredictor
from delay_classifier import DelayClassifier

class FlightDelayPipeline:

    def __init__(self, data_path):
        self.spark = SparkSession.builder \
            .appName("FlightDelayPrediction") \
            .getOrCreate()
        self.data_path = data_path
        self.data = None
        self.features = None
        self.model = None
        self.predictions = None

    def run(self):
        # Step 1: Load and preprocess the data
        handler = DataHandler(self.spark, self.data_path)
        self.data = handler.load_and_preprocess()

        # Step 2: Feature Engineering
        engineer = FeatureEngineer(self.data)
        self.features = engineer.transform_features()

        # Step 3: Train the model
        predictor = FlightDelayPredictor(self.features)
        self.model = predictor.train_model()

        # Step 4: Predict and classify delays
        classifier = DelayClassifier(self.model, self.features)
        self.predictions = classifier.classify_delays()

        # (Optional) Show results
        self.predictions.show()

    def stop_spark(self):
        self.spark.stop()

if __name__ == "__main__":
    DATA_PATH = "path/to/your/data.csv"
    pipeline = FlightDelayPipeline(DATA_PATH)
    pipeline.run()
    pipeline.stop_spark()
