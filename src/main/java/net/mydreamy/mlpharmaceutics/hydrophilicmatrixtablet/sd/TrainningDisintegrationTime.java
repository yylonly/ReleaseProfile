package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.sd;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base.Evaluation;



public class TrainningDisintegrationTime {

	public static Logger log = LoggerFactory.getLogger(TrainningDisintegrationTime.class);
	


	//Random number generator seed, for reproducability
    public static final int seed = 12345678;
    
    //Number of iterations per minibatch
    public static final int iterations = 1;
    
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 2000;

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int trainsetSize = 541;
    public static final int testsetsize = 136;
    public static final int batchSize = 32;
    
    //Network learning rate
    public static final double learningRate = 0.01;
    
    //with api properties
    public static final int numInputs = 33; //from 0
    public static final int numOutputs = 7;
    public static final int numHiddenNodes = 80;
//     public static final int numHiddenNodes = 60;

    
    	
	public static void main(String[] args) {
		

	
		
		//First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
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

// //       log.info("testData set:" + testData.toString());
//        
////        // Normalization
////        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
//   
//        DataSet trainningData = iteratortrain.next();
//        DataSet testData = iteratortest.next();
//
//        
////        log.info(testData.get(0).toString());
//        
// //       log.info(trainningData.toString());
// //       log.info(testData.toString());
//        
////        normalizer.fitLabel(false);
////        normalizer.fit(trainningData); 
////        normalizer.transform(trainningData); 
////        normalizer.transform(testData); 
////        
//        iteratortrain.reset();
//        iteratortest.reset();
////        
////       log.info(trainningData.toString());
////        log.info(testData.toString());
////        log.info("training data features:\n" + trainningData.getFeatureMatrix().toString());
////        log.info("training data label:\n" + trainningData.getLabels().toString());
////        normalizer.transform(testData); 
////        log.info("training data features:\n" + testData.getFeatureMatrix().toString());
////        log.info("training data label:\n" + testData.getLabels().toString());
        
        // Network Configuration
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .learningRateDecayPolicy(LearningRatePolicy.Exponential)
//                .lrPolicyDecayRate(0.01)
                .weightInit(WeightInit.RELU)
                .l2(2e-3)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//                .dropOut(0.8)
                .updater(new Nesterovs(learningRate, 0.9))
//                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.TANH).dropOut(0.9)
                        .build())             
                .layer(1, new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.TANH).dropOut(0.9)
                        .build())
                .layer(2, new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.TANH).dropOut(0.9)
                        .build())
                .layer(3, new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.TANH).dropOut(0.9)
                        .build())
                .layer(4, new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.TANH).dropOut(0.9)
                        .build())
                .layer(5, new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.TANH).dropOut(0.9)
                        .build())
                .layer(6, new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.TANH).dropOut(0.9)
                        .build())
                .layer(7, new DenseLayer.Builder().nOut(numHiddenNodes)
                        .activation(Activation.TANH).dropOut(0.9)
                        .build())
//                .layer(8, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
//                        .activation("tanh")
//                        .build())
                .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation(Activation.TANH.SIGMOID)
                        .nOut(numOutputs).build())
                .build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        
        
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        
        File modelHistory = new File("src/main/resources/trainning.db");
        if (modelHistory.exists()) //noinspection ResultOfMethodCallIgnored
        	modelHistory.delete();
        StatsStorage statsStorage = new FileStatsStorage(modelHistory);
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        net.setListeners(new StatsListener(statsStorage));
        
        
        List<EpochTerminationCondition> terminationconditions = new LinkedList<EpochTerminationCondition>();
      //  terminationconditions.add(new ScoreImprovementEpochTerminationCondition(100, 1E-10));
//        terminationconditions.add(new BestScoreEpochTerminationCondition(0.01));
//        terminationconditions.add(new MaxEpochsTerminationCondition(1100));
//      terminationconditions.add(new MaxEpochsTerminationCondition(4000));

        terminationconditions.add(new MaxEpochsTerminationCondition(nEpochs));

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        		.epochTerminationConditions(terminationconditions)
        		.scoreCalculator(new DataSetLossCalculator(iteratortest, true))
                .evaluateEveryNEpochs(100)
                .saveLastModel(true)
        		.modelSaver(new LocalFileModelSaver("src/main/resources"))
        		.build();
        
        
        
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, iteratortrain);

        //Conduct early stopping training:
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

       

       //Print out the results:
        log.info("Termination reason: " + result.getTerminationReason());
        log.info("Termination details: " + result.getTerminationDetails());
        log.info("Total epochs: " + result.getTotalEpochs());
        log.info("Best epoch number: " + result.getBestModelEpoch());
        log.info("Score at best epoch: " + result.getBestModelScore());
        
        
        MultiLayerNetwork bestModel = null;
 //   	bestModel = result.getBestModel();
//
        try {
        	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/bestModel.bin"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        MultiLayerNetwork latestModel = null;
    //	bestModel = result.getBestModel();

        try {
        	latestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/latestModel.bin"));
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
}
