package ann;

import java.io.FileWriter;
import java.io.IOException;

public class XorMain {

	public static void recordData(NeuralNetwork xorNetwork) {
		try {
			FileWriter fw = new FileWriter("xorData.csv");
			double[] entries = new double[2];
			double[] z;
			String data = ",0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1\n"; // y row for mesh format
			fw.write(data);
			for (double left = 0; left <= 1; left += 0.1) {
				data = ""+left;
				for (double right = 0; right <= 1; right += 0.1) {
					entries[0] = left;
					entries[1] = right;
					z = xorNetwork.calculateOutputs(entries);
					data += "," + z[0];
				}
				data+="\n";
				fw.write(data);
			}
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		double[][] trainingSet = { { .1, .1 }, { .1, .9 }, { .9, .1 }, { .9, .9 } };
		double[][] expectedOutputs = { { .1 }, { .9 }, { .9 }, { .1 } };
		double targetError = 0.0000001;
		int maxSteps = 100000000;
		int[] layers = { 2, 2, 1 };
		double learningRate = 0.5;
		double momentum = 0.2;

		NeuralNetwork xorNetwork = new NeuralNetwork(layers, learningRate, momentum);
		xorNetwork.learn(trainingSet, expectedOutputs, targetError, maxSteps);
		
		recordData(xorNetwork);
		double[] t = {.1,.1};
		System.out.println(xorNetwork.calculateOutputs(t)[0]);

	}

}
