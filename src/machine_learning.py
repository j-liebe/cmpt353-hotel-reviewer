# Train machine learning model called from main.py

import pandas as pd
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, IndexToString
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.mllib.evaluation import MulticlassMetrics

# set up Spark
spark = SparkSession.builder \
    .appName("Machine Learning Models") \
    .master("local[*]") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Schema for reading training and prediction data
review_comments_schema = types.StructType([
    types.StructField('comment_id', types.StringType()),
    types.StructField('review', types.StringType()),
    types.StructField('review_text', types.StringType()),
])

#------------------------------------------------------------------------------------------#
# Function to read training and prediction data from the input file paths
# 
# Returns balanced training and prediction data
def get_balanced_training_prediction_data(training_data_path, prediction_data_path):
    training_data_spark_df = spark.read.load(training_data_path, format="json", schema = review_comments_schema).cache()
    prediction_data_spark_df = spark.read.load(prediction_data_path, format="json", schema = review_comments_schema).cache()

    # remove null from all the columns before running the model
    training_data_spark_df = training_data_spark_df.na.drop()
    prediction_data_spark_df = prediction_data_spark_df.na.drop()
    
    # Balance training data as we have less number of negative review compared to positive reviews in training dataset
    negative_training_reviews = training_data_spark_df.filter(training_data_spark_df.review == "negative")
    positive_training_reviews = training_data_spark_df.filter(training_data_spark_df.review == "positive")

    negative_training_reviews_counts = negative_training_reviews.count()
    original_training_data_count = training_data_spark_df.count()
    sampled_positive_training_reviews = positive_training_reviews.sample((negative_training_reviews_counts/original_training_data_count) + 0.1)
    
    balanced_training_data_spark_df = sampled_positive_training_reviews.union(negative_training_reviews)

    return balanced_training_data_spark_df, prediction_data_spark_df

#------------------------------------------------------------------------------------------#
# Function to merge predicted results with complete review dataset with unique review identifier "comment_id"
# Export merged dataframe as csv.gzip for further analysis
def join_export_results(nb_prediction_pandas_df):
    # Obtain cleaned english data
    english_data = pd.read_csv('./filtered_data/all_cities_cleaned_english_reviews.csv.gz')

    # Perform merge for review taken for predictions. Essentially these reviews are for hotels located in user input city
    nb_prediction_pandas_df['comment_id'] = nb_prediction_pandas_df['comment_id'].astype('int64')
    nb_prediction_for_user_input_city = english_data.merge(nb_prediction_pandas_df, on = 'comment_id', how = 'inner')
    nb_prediction_for_user_input_city = nb_prediction_for_user_input_city.drop("review_text", axis = 1)
    
    nb_prediction_for_user_input_city.to_csv('./predicted_data/nb_prediction_for_user_input_city.csv.gz', index = False, compression = 'gzip')

    return nb_prediction_for_user_input_city

#------------------------------------------------------------------------------------------#
# Funtion to trains Naive Bayes model and obtain predictions.
# Training data is read from   : ./filtered_data/training_data.json
# Prediction data is read from : ./filtered_data/prediction_data.json
# 
# Model uses NLP pipeline with Naive Bayes Multinomal model. It will train on balanced training data containing review text for NLP analysis.
# 
# Return value : Returns a dataframe containing Naive Bayes predictions on basis of NLP analysis on hotel reviews for hotels located in user input city hotel. 
# 
# Adapted the code/tranformers to write NLP pipeline with pyspark library function from this article https://medium.datadriveninvestor.com/nlp-with-pyspark-9e5f1fca7adf 
def naive_bayes():

    print("\nTraining Naive Bayes model...\n")

    # Obtain balanced training and prediction data

    training_data_spark_df, prediction_data_spark_df = get_balanced_training_prediction_data("./filtered_data/training_data.json", "./filtered_data/prediction_data.json")

    # feature engineering on review_text for NLP
    tokenizer = Tokenizer(inputCol = "review_text", 
                        outputCol = "words")

    stopWordsRemover = StopWordsRemover(inputCol = "words", 
                                        outputCol = "words_without_stopwords")

    vectorizer = CountVectorizer(inputCol = "words_without_stopwords", 
                                    outputCol = "features")

    # We have strings(poitive or negative) in reviews which are our expected converting to index
    labelEncoder = StringIndexer(inputCol = "review", 
                                outputCol = "reviewIndexed").fit(training_data_spark_df)

    # Use NaiveBayes multinomial model on features and reviewIndexed 
    naive_bayes = NaiveBayes(modelType = "multinomial", 
                            featuresCol = "features", 
                            labelCol = "reviewIndexed")

    # Convert prediction which are indexed format to labels
    labelConverter = IndexToString(inputCol = "prediction", 
                                outputCol = "predicted_review_label", 
                                labels = labelEncoder.labels)

    # make pipeline
    pipeline = Pipeline(
        stages = [
            tokenizer,
            stopWordsRemover, 
            vectorizer,
            labelEncoder, 
            naive_bayes, 
            labelConverter
        ])

    # fit training data which is non-ny hotel reviews
    naive_bayes_model = pipeline.fit(training_data_spark_df)

    # make predictions on validating data which is ny-hotel reviews
    predictions = naive_bayes_model.transform(prediction_data_spark_df)

    # Accuracy score using same model is recorded using ml_model_testing.ipynb. It was easy to run the models and record output

    nb_prediction_spark_df = predictions.select("comment_id",
                                                "review",
                                                "review_text",
                                                "reviewIndexed",
                                                "prediction",
                                                "predicted_review_label"
                                            )

    # converting results to pandas and merge with original ny-hotel-english-data
    nb_prediction_pandas_df = nb_prediction_spark_df.toPandas()

    nb_prediction_for_user_input_city = join_export_results(nb_prediction_pandas_df)

    return nb_prediction_for_user_input_city

# =====================================================================================================================================================

# Funtion to trains Random Forest Classifer and obtain predictions.
# Training data is read from              : ./filtered_data/training_data.json
# prediction/Prediction data is read from : ./filtered_data/prediction_data.json
# 
# Model uses NLP pipeline with Random Forest Classifer model. It will train on balanced training data containing review text for NLP analysis.
# 
# Return value : Returns a dataframe containing Random Forest Classifer predictions on basis of NLP analysis on hotel reviews for hotels located in user input city hotel.
def random_forest_classifier():
    # Obtain balanced training and prediction data
    training_data_spark_df, prediction_data_spark_df = get_balanced_training_prediction_data("./filtered_data/training_data.json", "./filtered_data/prediction_data.json")
    
    # feature engineering on review_text for NLP
    tokenizer = Tokenizer(inputCol = "review_text", 
                        outputCol = "words")

    stopWordsRemover = StopWordsRemover(inputCol = "words", 
                                        outputCol = "words_without_stopwords")

    vectorizer = CountVectorizer(inputCol = "words_without_stopwords", 
                                    outputCol = "features")

    # We have strings(poitive or negative) in reviews which are our expected converting to index
    labelEncoder = StringIndexer(inputCol = "review", 
                                outputCol = "reviewIndexed").fit(training_data_spark_df)

    # Use NaiveBayes multinomial model on features and reviewIndexed 
    rfc = RandomForestClassifier(numTrees = 100, 
                                maxDepth = 30,
                                featuresCol = "features", 
                                labelCol = "reviewIndexed")

    # Convert prediction which are indexed format to labels
    labelConverter = IndexToString(inputCol = "prediction", 
                                outputCol = "predicted_review_label", 
                                labels = labelEncoder.labels)

    # make pipeline
    pipeline = Pipeline(
        stages = [
            tokenizer,
            stopWordsRemover, 
            vectorizer,
            labelEncoder, 
            rfc, 
            labelConverter
        ])

    # fit training data which is non-ny hotel reviews
    rfc_model = pipeline.fit(training_data_spark_df)

    # make predictions on validating data which is ny-hotel reviews
    rfc_predictions = rfc_model.transform(prediction_data_spark_df)

    # Accuracy score using same model is recorded using ml_model_testing.ipynb. It was easy to run the models and record output

    rfc_prediction_spark_df = rfc_predictions.select("comment_id",
                                                "review",
                                                "review_text",
                                                "reviewIndexed",
                                                "prediction",
                                                "predicted_review_label"
                                            )

    # converting results to pandas and merge with original ny-hotel-english-data
    rfc_prediction_pandas_df = rfc_prediction_spark_df.toPandas()

    rfc_predictions_for_user_input_city = join_export_results(rfc_prediction_pandas_df)

    return rfc_predictions_for_user_input_city

#------------------------------------------------------------------------------------------#    
# Funtion to trains Linear SVC and obtain predictions.
# Training data is read from              : ./filtered_data/training_data.json
# prediction/Prediction data is read from : ./filtered_data/prediction_data.json
# 
# Model uses NLP pipeline with Linear SVC model. It will train on balanced training data containing review text for NLP analysis.
# 
# Return value : Returns a dataframe containing Linear SVC predictions on basis of NLP analysis on hotel reviews for hotels located in user input city hotel. 
def linear_svc():
    # Obtain balanced training and prediction data
    training_data_spark_df, prediction_data_spark_df = get_balanced_training_prediction_data("./filtered_data/training_data.json", "./filtered_data/prediction_data.json")
    
    # feature engineering on review_text for NLP
    tokenizer = Tokenizer(inputCol = "review_text", 
                        outputCol = "words")

    stopWordsRemover = StopWordsRemover(inputCol = "words", 
                                        outputCol = "words_without_stopwords")

    vectorizer = CountVectorizer(inputCol = "words_without_stopwords", 
                                    outputCol = "features")

    # We have strings(poitive or negative) in reviews which are our expected converting to index
    labelEncoder = StringIndexer(inputCol = "review", 
                                outputCol = "reviewIndexed").fit(training_data_spark_df)

    # Use NaiveBayes multinomial model on features and reviewIndexed 
    linear_svc = LinearSVC(maxIter = 20,
                           regParam = 0.01,
                           featuresCol = "features", 
                           labelCol = "reviewIndexed")

    # Convert prediction which are indexed format to labels
    labelConverter = IndexToString(inputCol = "prediction", 
                                outputCol = "predicted_review_label", 
                                labels = labelEncoder.labels)

    # make pipeline
    pipeline = Pipeline(
        stages = [
            tokenizer,
            stopWordsRemover, 
            vectorizer,
            labelEncoder, 
            linear_svc, 
            labelConverter
        ])

    # fit training data which is non-ny hotel reviews
    linear_svc_model = pipeline.fit(training_data_spark_df)

    # make predictions on validating data which is ny-hotel reviews
    linear_svc_predictions = linear_svc_model.transform(prediction_data_spark_df)

    # Accuracy score using same model is recorded using ml_model_testing.ipynb. It was easy to run the models and record output

    linear_svc_prediction_spark_df = linear_svc_predictions.select("comment_id",
                                                "review",
                                                "review_text",
                                                "reviewIndexed",
                                                "prediction",
                                                "predicted_review_label"
                                            )

    # converting results to pandas and merge with original ny-hotel-english-data
    linear_svc_prediction_pandas_df = linear_svc_prediction_spark_df.toPandas()

    svc_predictions_for_user_input_city = join_export_results(linear_svc_prediction_pandas_df)

    return svc_predictions_for_user_input_city

#------------------------------------------------------------------------------------------#

# Based on our analysis, Naive Bayes model provides highest accuracy predictions. 
# It also takes significantly less time to train than Random Forest Classifier and Linear SVC
#
# For this reason, Naive Bayes model will be used for our final hotel recommendations.
# (More description in report)