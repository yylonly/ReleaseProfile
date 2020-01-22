package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.sd;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.conf.updater.NesterovsSpace;
import org.deeplearning4j.arbiter.conf.updater.SgdSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.GridSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.math.MathOp;
import org.deeplearning4j.arbiter.optimize.parameter.math.Op;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.scoring.impl.RegressionScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.arbiter.ui.listener.ArbiterStatusListener;
import org.deeplearning4j.arbiter.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.VertxUIServer;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.activations.Activation;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base.Evaluation;
import net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.sd.BasicHyperparameterOptimizationExample.ExampleDataSource;



public class TrainningDisintegrationTimeWithSearch {

	public static Logger log = LoggerFactory.getLogger(TrainningDisintegrationTimeWithSearch.class);
	


	//Random number generator seed, for reproducability
    public static final int seed = 1234567890;
    
    
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 100000;

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int trainsetSize = 541;
    public static final int testsetsize = 136;
    public static final int batchSize = 64;
    
    public static final int candidateNum = 20;
    
    public static final int discretizationCount = 50;
    
    //with api properties
    public static final int numInputs = 33; //from 0
    public static final int numOutputs = 7;

//     public static final int numHiddenNodes = 60;

    
    	
	public static void main(String[] args) {
		
        
        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.01, 0.1);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
        ParameterSpace<Double> dropRateHyperparam = new ContinuousParameterSpace(0.9, 1);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
        ParameterSpace<Integer> numHiddenNodesHyperparam = new IntegerParameterSpace(150, 300);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)
        DiscreteParameterSpace<Activation> activationSpace = new DiscreteParameterSpace(new Activation[]{Activation.TANH});
        DiscreteParameterSpace<WeightInit> weightInitSpace = new DiscreteParameterSpace(new WeightInit[]{WeightInit.RELU, WeightInit.UNIFORM, WeightInit.XAVIER, WeightInit.NORMAL});

        
        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
                //These next few options: fixed values for all models
                .weightInit(weightInitSpace)
//                .l2(new ContinuousParameterSpace(0.2, 0.5))
                //Learning rate hyperparameter: search over different values, applied to all models
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .updater(new NesterovsSpace(learningRateHyperparam))
                .addLayer(new DenseLayerSpace.Builder()
                    //Fixed values for this layer:
                    .nIn(numInputs)  //Fixed input: 28x28=784 pixels for MNIST
                    .activation(activationSpace).dropOut(dropRateHyperparam)
                    //One hyperparameter to infer: layer size
                    .nOut(new MathOp<>(numHiddenNodesHyperparam, Op.MUL, 1))
                    .build())
                .addLayer(new DenseLayerSpace.Builder()
                		.nOut(new MathOp<>(numHiddenNodesHyperparam, Op.MUL, 1))
                        .activation(activationSpace).dropOut(dropRateHyperparam)
                        .build())
                .addLayer(new DenseLayerSpace.Builder()
                		.nOut(new MathOp<>(numHiddenNodesHyperparam, Op.MUL, 1))
                        .activation(activationSpace).dropOut(dropRateHyperparam)
                        .build())
                .addLayer(new DenseLayerSpace.Builder()
                		.nOut(new MathOp<>(numHiddenNodesHyperparam, Op.MUL, 1))
                        .activation(activationSpace).dropOut(dropRateHyperparam)
                        .build())
                .addLayer(new DenseLayerSpace.Builder()
                		.nOut(new MathOp<>(numHiddenNodesHyperparam, Op.MUL, 1))
                        .activation(activationSpace)
                        .build())
                .addLayer(new DenseLayerSpace.Builder()
                		.nOut(numHiddenNodesHyperparam)
                        .activation(activationSpace)
                        .build())
                .addLayer(new DenseLayerSpace.Builder()
                		.nOut(new MathOp<>(numHiddenNodesHyperparam, Op.MUL, 1))
                        .activation(activationSpace)
                        .build())
                .addLayer(new DenseLayerSpace.Builder()
                		.nOut(new MathOp<>(numHiddenNodesHyperparam, Op.MUL, 1))
                        .activation(activationSpace)
                        .build())
                .addLayer(new OutputLayerSpace.Builder()
                    .nOut(numOutputs)
                    .activation(Activation.SIGMOID)
                    .lossFunction(LossFunctions.LossFunction.L2)
                    .build())
                .numEpochs(nEpochs)
                .build();

//        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);
        CandidateGenerator candidateGenerator = new GridSearchCandidateGenerator(hyperparameterSpace, discretizationCount, GridSearchCandidateGenerator.Mode.RandomOrder, null);
        
        Class<? extends DataSource> dataSourceClass = PharmDataSource.class;
        Properties dataSourceProperties = new Properties();
        dataSourceProperties.setProperty("minibatchSize", "64");
        
        String baseSaveDirectory = "src/main/resources/models";
        File f = new File(baseSaveDirectory);
        if (f.exists()) //noinspection ResultOfMethodCallIgnored
            f.delete();
        //noinspection ResultOfMethodCallIgnored
        f.mkdir();
        ResultSaver modelSaver = new FileModelSaver(baseSaveDirectory);
        
        ScoreFunction scoreFunction = new RegressionScoreFunction(RegressionEvaluation.Metric.MSE);

        
        TerminationCondition[] terminationConditions = {
                new MaxTimeCondition(10, TimeUnit.HOURS),
                new MaxCandidatesCondition(candidateNum)};       
        
      //Given these configuration options, let's put them all together:
        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
            .candidateGenerator(candidateGenerator)
            .dataSource(dataSourceClass,dataSourceProperties)
            .modelSaver(modelSaver)
            .scoreFunction(scoreFunction)
            .terminationConditions(terminationConditions)
            .build();

        //And set up execution locally on this machine:
        IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());

        
        //Start the UI. Arbiter uses the same storage and persistence approach as DL4J's UI
        //Access at http://localhost:9000/arbiter
        File modelHistory = new File("src/main/resources/dl4j.db");
        if (modelHistory.exists()) //noinspection ResultOfMethodCallIgnored
        	modelHistory.delete();
        StatsStorage ss = new FileStatsStorage(modelHistory);
        runner.addListeners(new ArbiterStatusListener(ss));
        
        UIServer uiServer = VertxUIServer.getInstance(8888, true, null);
       
        
        uiServer.attach(ss);
        
        if (uiServer == null)
        	System.out.println("null");
        
        
        //Start the hyperparameter optimization
        runner.execute();
        
        
        runner.shutdown(false);
        
        //Print out some basic stats regarding the optimization procedure
        String s = "Best score: " + runner.bestScore() + "\n" +
            "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" +
            "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);


        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference> allResults = runner.getResults();
        
        System.out.println("\n\n Index of best model:" + indexOfBestResult + "\n");

        OptimizationResult bestResult = null;
		try {
			bestResult = allResults.get(indexOfBestResult).getResult();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}  

		
        MultiLayerNetwork bestModel = null;
		try {
			bestModel = (MultiLayerNetwork) bestResult.getResultReference().getResultModel();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

//        System.out.println("\n\nConfiguration of best model:\n");
//        System.out.println(bestModel.getLayerWiseConfigurations().toJson());
//        
        try {
        	ModelSerializer.writeModel(bestModel, new File("src/main/resources/latestModel.bin"), true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
       
        
        //Get the best model:
       //  bestModel =  result.getBestModel();
        
//        //Train the network on the full data set, and evaluate in periodically
//        for( int i=0; i<nEpochs; i++ ){
//            net.fit(iteratortrain);
//            iteratortrain.reset();
//        }
        
//        iteratortrain.reset();
 //       DataSet trainningData = iteratortrain.next();
        
 //       iteratortest.reset();
  //      DataSet testData = iteratortest.next();
        
        
        
        
//        log.info("========================== testing =========================");
//        log.info("========================== latest model =========================");
//        //test on latest model
//        testOnDiffModel(latestModel, trainningData, testData);
//        
//        log.info("========================== best model =========================");
//        //test on best model
//        testOnDiffModel(bestModel, trainningData, testData);
 
	}

	public static void testOnDiffModel(MultiLayerNetwork net, DataSet trainningData,  DataSet testData)
	{
	       // evaluation training set
        RegressionEvaluation evalTrain = new RegressionEvaluation(numOutputs);
        
        INDArray featuresTrain = trainningData.getFeatures();
        INDArray lablesTrain = trainningData.getLabels();
        
        INDArray PredictionTrain = net.output(featuresTrain);
     //   log.info("train label set:\n" + lablesTrain.toString());
     //   log.info("train prediction set:\n" + PredictionTrain.toString());
        evalTrain.eval(lablesTrain, PredictionTrain);	  
        
    //    log.info("training set MSE is:" + String.format("%.10f", (evalTrain.meanSquaredError(0)+evalTrain.meanSquaredError(1)+evalTrain.meanSquaredError(2)+evalTrain.meanSquaredError(3))/4));
        
        double AverTR = 0;
        double AverTMAE = 0;

        for (int i = 0; i < numOutputs-1; i++) {
        	AverTR += evalTrain.correlationR2(i);
        	AverTMAE += evalTrain.meanAbsoluteError(i);
        }
        
        log.info("training set R is:" + String.format("%.4f", AverTR/numOutputs));
        log.info("training set MAE is: " + String.format("%.4f",  AverTMAE/numOutputs));
        
        Evaluation.f2(lablesTrain, PredictionTrain);
        Evaluation.AccuracyMAE(lablesTrain, PredictionTrain, 0.10);
        Evaluation.AccuracyMAE(lablesTrain, PredictionTrain, 0.12);
        Evaluation.AccuracyMAE(lablesTrain, PredictionTrain, 0.15);



        
        // evluation test set
        RegressionEvaluation evalTest = new RegressionEvaluation(numOutputs);
        
        INDArray featuresTest = testData.getFeatures();
    //    log.info("featuresTest" + featuresTest.shapeInfoToString());
    //    log.info("\n" + featuresTest.toString());

        INDArray lablesTest = testData.getLabels();
        
        
//        log.info(evalTest.stats());
        INDArray PredictionTest = net.output(featuresTest);
        
//        log.info("test label value: \n" + lablesTest.toString());
//        log.info("test prediction value: \n" + PredictionTest.toString());

        evalTest.eval(lablesTest, PredictionTest);	  
        
        double AverTestR = 0;
        double AverMAE = 0;
        for (int i = 0; i < numOutputs-1; i++) {
        	AverTestR += evalTest.correlationR2(i);
        	AverMAE += evalTest.meanAbsoluteError(i);
        }
        
    //    log.info("testing set MSE is: " + String.format("%.10f", (evalTest.meanSquaredError(0)+evalTest.meanSquaredError(1)+evalTest.meanSquaredError(2)+evalTest.meanSquaredError(3))/4)); 
        log.info("testing set R is: " + String.format("%.4f",  AverTestR/numOutputs));
        log.info("testing set MAE is: " + String.format("%.4f",  AverMAE/numOutputs));
        
        Evaluation.f2(lablesTest, PredictionTest);
        Evaluation.AccuracyMAE(lablesTest, PredictionTest, 0.10);
        Evaluation.AccuracyMAE(lablesTest, PredictionTest, 0.12);
        Evaluation.AccuracyMAE(lablesTest, PredictionTest, 0.15);



	}
	
	 public static class PharmDataSource implements DataSource {
	        private int minibatchSize;

	        public PharmDataSource() {

	        }

	        @Override
	        public void configure(Properties properties) {
	            this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize", "16"));
	        }

	        @Override
	        public Object trainData() {
	            try {
	            	
	            	 int numLinesToSkip = 1;
	                 String delimiter = ",";
	                 RecordReader recordReadertrain = new CSVRecordReader(numLinesToSkip,delimiter);
	                 try {
	                 	recordReadertrain.initialize(new FileSplit(new ClassPathResource("sd/trainSet.csv").getFile()));
	         		} catch (FileNotFoundException e) {
	         			// TODO Auto-generated catch block
	         			e.printStackTrace();
	         		} catch (IOException e) {
	         			// TODO Auto-generated catch block
	         			e.printStackTrace();
	         		} catch (InterruptedException e) {
	         			// TODO Auto-generated catch block
	         			e.printStackTrace();
	         		}

	                 DataSetIterator iteratortrain = new RecordReaderDataSetIterator(recordReadertrain,batchSize,numInputs,numInputs+numOutputs-1,true);
	            
	            	
	                return iteratortrain;

	            } catch (Exception e) {
	                throw new RuntimeException(e);
	            }
	        }

	        @Override
	        public Object testData() {
	            try {
	            	 int numLinesToSkip = 1;
	                 String delimiter = ",";
	            	 RecordReader recordReadertest = new CSVRecordReader(numLinesToSkip,delimiter);
	                 try {
	                 	recordReadertest.initialize(new FileSplit(new ClassPathResource("sd/devSet.csv").getFile()));
	         		} catch (FileNotFoundException e) {
	         			// TODO Auto-generated catch block
	         			e.printStackTrace();
	         		} catch (IOException e) {
	         			// TODO Auto-generated catch block
	         			e.printStackTrace();
	         		} catch (InterruptedException e) {
	         			// TODO Auto-generated catch block
	         			e.printStackTrace();
	         		}

	                 DataSetIterator iteratortest = new RecordReaderDataSetIterator(recordReadertest,batchSize,numInputs,numInputs+numOutputs-1,true);

	                return iteratortest;

	            } catch (Exception e) {
	                throw new RuntimeException(e);
	            }
	        }

	        @Override
	        public Class<?> getDataType() {
	            return DataSetIterator.class;
	        }
	    }
}
