/*
* CS 6350.002 Big Data Analytics Project
*
* Project Name : Facebook V: Predicting Check-ins

* Project Purpose: The sole purpose of the project is to find the most accurate place where a user would check-in based on varius feature parameters such as x-y co-ordinates, time
* and accuracy. This project has been evaluated based on 3 Machine Learning models namely Random Forests, Logistc Regression and Decision Trees. Based on the various properties and
* tuning parameters, we found accuracy, precision, recall and F-1 score for each of the model. Due to very high number of class/labels (38000), the accuracy would affect.
*
* Professor : Anurag Nagar
*
* Teaching Assistant : Yifan Li
*
* Project Members:
*    Priyank Shah (pss160530)     [Implemented Decision Trees Model]
*    Mohammed Ather Ahmed Shareef (mxs166331)     [Implemented Random Forest Model]
*    Abhishek Jagwani (alj160130)     [Implemented Data Pre-processing, Data Analyzing, Graph Plotting and Model Integration]
*    Jahnavi Mesa (jxm169230)     [Implemented Logistic Regresion Model]
* */




//doing all requied imports
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, DecisionTreeClassifier, DecisionTreeClassificationModel, OneVsRest, OneVsRestModel, RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorAssembler, PCA, MinMaxScaler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Part1
{
  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("Spark Count"))
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    val sqlcon = spark.sqlContext
    import spark.implicits._
    var rawData=sqlcon.read.format("csv").option("header","true").option("inferSchema", "true").load(args(0))
    val data_1x1 = rawData.filter($"x" <= 1.0 && $"y" <=1.0)
    val Array(trainingData, testData) = data_1x1.randomSplit(Array(0.8, 0.2))

    //converting all the string fields using string indexer
    val place_indexer = new StringIndexer()
      .setInputCol("place_id")
      .setOutputCol("place_id_indexed")
      .setHandleInvalid("skip")//interest_level_indexed
      .fit(trainingData)

    val x_indexed= new StringIndexer()
      .setInputCol("x")
      .setOutputCol("x_index")
      .setHandleInvalid("skip")

    val y_indexed= new StringIndexer()
      .setInputCol("y")
      .setOutputCol("y_index")
      .setHandleInvalid("skip")

    val accuracy_indexed= new StringIndexer()
      .setInputCol("accuracy")
      .setOutputCol("accuracy_index")
      .setHandleInvalid("skip")

    val time_indexed= new StringIndexer()
      .setInputCol("time")
      .setOutputCol("time_index")
      .setHandleInvalid("skip")

    //creating datafeatures array as we are having 4 features
    val dataFeatures = Array("x_index", "y_index", "accuracy_index", "time_index")

    val assembler = new
        VectorAssembler().setInputCols(dataFeatures).setOutputCol("data_features")


    //implementing the random forest model
    val rf = new RandomForestClassifier().setLabelCol("place_id_indexed").setFeaturesCol("data_features")
    val rf_Parameter_Grid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(10, 20, 30))
      .addGrid(rf.maxDepth, Array(3, 4, 5))
      .build()
    val rfStages = Array(place_indexer,x_indexed,y_indexed, accuracy_indexed,time_indexed,assembler, rf)

    //implementing pipelines for the random forest
    val pipeline = new Pipeline().setStages(rfStages)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("place_id_indexed")
      .setPredictionCol("prediction")
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(rf_Parameter_Grid)
      .setNumFolds(12)


    val rf_model = cv.fit(trainingData)
    val rfPredictions = rf_model.transform(testData)
    var accuracy1=0.0
    var preci1=0.0
    var recall1=0.0
    var f1_score1=0.0

    //below describing the method which will compute accuracy, precision, recall, f1-score metrics
    def displayMetrics(pAndL : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(pAndL)
      accuracy1 = metrics.accuracy
      preci1 = metrics.weightedPrecision
      recall1 = metrics.weightedRecall
      f1_score1 = metrics.weightedFMeasure
    }
    var output = ""
    val best_rf_model = rf_model.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[RandomForestClassificationModel]
    output+="\n Maximum Depth: " + best_rf_model.getMaxDepth +"\n"
    output+="\n Number of Trees: " + best_rf_model.getNumTrees +"\n"
    val rfPredictionAndLabels = rfPredictions.select("prediction", "place_id_indexed").rdd.map{case Row(prediction: Double, place_id_indexed: Double) => (prediction, place_id_indexed)}
    displayMetrics(rfPredictionAndLabels)

    output += "Random Forest metrics are:\n"
    output+= "Accuracy for Random Forest: \t" + accuracy1 + "\n"
    output+= "Precision for Random Forest: \t" + preci1 + "\n"
    output+= "Recall  for Random Forest: \t" + recall1 + "\n"
    output+= "F1Score  for Random Forest: \t" + f1_score1 + "\n"
    var my_rdd :RDD[String]= null

    //making the Logistic Regression model
    val lr = new LogisticRegression()
    val m = new OneVsRest().setClassifier(lr)
      .setLabelCol("place_id_indexed")
      .setFeaturesCol("data_features")
    val lr_param_grid = new ParamGridBuilder()
      .addGrid(lr.maxIter, Array(100, 110, 90))//Int Param (>=0)
      .addGrid(lr.regParam, Array(0.0, 0.1, 0.2))//Double Param (>=0)
      .build()
    val lr_stages = Array(place_indexer,x_indexed,y_indexed, accuracy_indexed,time_indexed, assembler,m)

    //building pipelines for the logistic regression
    val pipeline1 = new Pipeline().setStages(lr_stages)
    val evaluator1 = new MulticlassClassificationEvaluator()
      .setLabelCol("place_id_indexed")
      .setPredictionCol("prediction")
    val cv1 = new CrossValidator()
      .setEstimator(pipeline1)
      .setEvaluator(evaluator1)
      .setEstimatorParamMaps(lr_param_grid)
      .setNumFolds(12)
    val lr_model = cv1.fit(trainingData)
    val logisticRegressionPredictions = lr_model.transform(testData)

    var accuracy2=0.0
    var preci2=0.0
    var recall2=0.0
    var f1_score2=0.0
    //below describing the accuracy, precision, recall, f1-score metrics
    def displayMetrics1(pAndL : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(pAndL)
      accuracy2 = metrics.accuracy
      preci2 = metrics.weightedPrecision
      recall2 = metrics.weightedRecall
      f1_score2 = metrics.weightedFMeasure
    }
    //Displaying the logistic regression model metrics
    val best_lr_model = lr_model.bestModel.asInstanceOf[PipelineModel].stages(4).asInstanceOf[OneVsRestModel].models(0).asInstanceOf[LogisticRegressionModel]
    output+="Max Iterations: " + best_lr_model.getMaxIter +"\n"
    output+="Regularization Parameter: " + best_lr_model.getRegParam+"\n"
    val lr_prediction_label = logisticRegressionPredictions.select("prediction", "place_id_indexed").rdd.map{case Row(prediction: Double, place_id_indexed: Double) => (prediction, place_id_indexed)}
    displayMetrics1(lr_prediction_label)

    //taking everything in single output
    output += "\n Logistic Regression metrics are:\n"
    output+= "Accuracy for logistic regression: \t" + accuracy2 + "\n"
    output+= "Precision for logistic regression: \t" + preci2 + "\n"
    output+= "Recall  for logistic regression: \t" + recall2 + "\n"
    output+= "F1Score  for logistic regression: \t" + f1_score2 + "\n"

    //building decision trees model
    val decision_tree = new DecisionTreeClassifier()
      .setLabelCol("place_id_indexed")
      .setFeaturesCol("data_features")
    val decision_paramGrid = new ParamGridBuilder()
      .addGrid(decision_tree.maxBins, Array(40, 45, 50))
      .addGrid(decision_tree.maxDepth, Array(5, 6, 8))
      .build()
    val decision_stages = Array(place_indexer,x_indexed,y_indexed, accuracy_indexed,time_indexed,assembler, rf)

    //building the pipelines for the decision tree model
    val pipeline2 = new Pipeline().setStages(decision_stages)
    val evaluator2 = new MulticlassClassificationEvaluator()
      .setLabelCol("place_id_indexed")
      .setPredictionCol("prediction")
    val cv2 = new CrossValidator()
      .setEstimator(pipeline2)
      .setEvaluator(evaluator2)
      .setEstimatorParamMaps(decision_paramGrid)
      .setNumFolds(12)
    val dt_model = cv2.fit(trainingData)
    val decision_predictions = dt_model.transform(testData)

    var accuracy3=0.0
    var preci3=0.0
    var recall3=0.0
    var f1_score3=0.0
    //below describing the accuracy, precision, recall, f1-score metrics
    def displayMetrics3(pAndL : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(pAndL)
      accuracy3 = metrics.accuracy
      preci3 = metrics.weightedPrecision
      recall3 = metrics.weightedRecall
      f1_score3 = metrics.weightedFMeasure
    }

    //printing the decision tree metrices
    val best_decision_model = dt_model.bestModel.asInstanceOf[PipelineModel].stages(4).asInstanceOf[DecisionTreeClassificationModel]
    output+="\n Max Bins: " + best_decision_model.getMaxBins+"\n"
    output+="\n Max Depth: " + best_decision_model.getMaxDepth+"\n"
    val dtPredictionAndLabels = decision_predictions.select("prediction", "place_id_indexed").rdd.map{case Row(prediction: Double, place_id_indexed: Double) => (prediction, place_id_indexed)}
    displayMetrics3(dtPredictionAndLabels)

    //taking everything in single output
    output += "\n Decision Tree metrics are:\n"
    output+= "Accuracy for Decision tree: \t" + accuracy2 + "\n"
    output+= "Precision for Decision tree: \t" + preci2 + "\n"
    output+= "Recall  for Decision tree: \t" + recall2 + "\n"
    output+= "F1Score  for Decision tree: \t" + f1_score2 + "\n"

    //making the rdd of the above computed results so the results of the computed model will be displayed in a single file one after the another.
    my_rdd = sc.parallelize(List(output))
    my_rdd.coalesce(1,true).saveAsTextFile(args(1))

  }
}
