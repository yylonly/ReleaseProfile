package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.sd4;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base.Prediction;

public class FinalTestResult {

	public static void main(String[] args) {
		
		
		//best Model testing
		Prediction.prediction("SD-new", "sd/trainSet.csv", 541, "src/main/resources/bestModel.bin", false); 
		Prediction.prediction("SD-new", "sd/devSet.csv", 136, "src/main/resources/bestModel.bin", false); 
		Prediction.prediction("SD-new", "sd/testSet.csv", 136, "src/main/resources/bestModel.bin", false); 
		
		System.out.println("==================== lastested ===================");
		Prediction.prediction("SD-new", "sd/trainSet.csv", 541, "src/main/resources/latestModel.bin", false); 
		Prediction.prediction("SD-new", "sd/devSet.csv", 136, "src/main/resources/latestModel.bin", false); 
		Prediction.prediction("SD-new", "sd/testSet.csv", 136, "src/main/resources/latestModel.bin", false); 
	
		
//		Prediction.prediction("SRMT-craft", "final-craft/trainset.csv", 200, "src/main/resources/final-craft/latestModel.bin", false); 
//		Prediction.prediction("SRMT-craft", "final-craft/devset.csv", 20, "src/main/resources/final-craft/latestModel.bin", false); 
//		Prediction.prediction("SRMT-craft", "final-craft/testset.csv", 20, "src/main/resources/final-craft/latestModel.bin", false); 
		
	}
}
