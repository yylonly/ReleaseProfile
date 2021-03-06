package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.sd;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base.Prediction;

public class FinalTestResult {

	public static void main(String[] args) {
		
		String bestModelurl = "src/main/resources/best/latestModel.bin";
//		String bestModelurl = "src/main/resources/best/model.bin";

		
//		System.out.println("OMP_NUM_THREADS "+  System.getenv().get("OMP_NUM_THREADS"));
		
//		String bestModelurl = "/Users/yylonly/fstdrive/ReleaseProfile/src/main/resources/models/41/model.bin";
		
		//best Model testing
//		Prediction.prediction("SD-new", "sd/trainSet.csv", 541, "src/main/resources/bestModel.bin", false); 
//		Prediction.prediction("SD-new", "sd/devSet.csv", 136, "src/main/resources/bestModel.bin", false); 
//		Prediction.prediction("SD-new", "sd/testSet.csv", 136, "src/main/resources/bestModel.bin", false); 
//		
		System.out.println("==================== lastested ===================");
		Prediction.prediction("SD-new", "sd/trainSet.csv", 541, bestModelurl, false); 
		Prediction.prediction("SD-new", "sd/devSet.csv", 136, bestModelurl, false); 
		Prediction.prediction("SD-new", "sd/testSet.csv", 136, bestModelurl, false); 
	
		
//		Prediction.prediction("SRMT-craft", "final-craft/trainset.csv", 200, "src/main/resources/final-craft/latestModel.bin", false); 
//		Prediction.prediction("SRMT-craft", "final-craft/devset.csv", 20, "src/main/resources/final-craft/latestModel.bin", false); 
//		Prediction.prediction("SRMT-craft", "final-craft/testset.csv", 20, "src/main/resources/final-craft/latestModel.bin", false); 
		
		MultiLayerNetwork bestModel = null;
		
		 try {
	        	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File(bestModelurl));
//	        	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File("src/main/resources/models/8/model.bin"));

	        } 
	        catch (IOException e) {
	       		e.printStackTrace();
	       	}
	          
	        
	        System.out.println(bestModel.summary());
	        System.out.println(bestModel.getLayerWiseConfigurations().toYaml());
		
	}
}
