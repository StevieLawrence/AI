package rnn;

import java.util.HashMap;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Map;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.jblas.DoubleMatrix;
import ann.LossFunction;
import java.util.List;


/**
 * Long Short-Term Memory Network
 */
public class LstmNetwork implements Serializable{
	private static final long serialVersionUID = -15L;
	
	LstmCell[] layers;
	int[] layerData;
	Map<String, DoubleMatrix>[] activations;
	double temperature;

	/**
	 * Recurrent Neural network that can handle both short and long term dependencies by useing a series of gates.
	 * @param layerData The input and hidden sizes for the stacked LSTM cells
	 * @param denseSize the output layer
	 * @param initer initiliazer for the weight matrices
	 * @param temperature Parameter for the degree of randomness in the sampling
	 */
	public LstmNetwork(int[] layerData, int denseSize, MatIniter initer, double temperature) {
		this.layerData = layerData;
		layers = new LstmCell[layerData.length - 1];
		this.temperature = temperature;
		activations = (Map<String, DoubleMatrix>[]) new HashMap<?,?>[layers.length];
		for (int i = 0; i < layerData.length - 1; i++) {
			if (i == layerData.length - 2)
				layers[i] = new LstmCell(layerData[i], layerData[i + 1], initer, denseSize);
			else
				layers[i] = new LstmCell(layerData[i], layerData[i + 1], initer);
		}
	}
	
	LstmNetwork(int[] layerData, int denseSize, MatIniter initer){
		this(layerData, denseSize, initer, 0.4);
	}

	// train the network
	public void train(CharData trainingData, int epochs, double learningRate) {
		Map<String, DoubleMatrix> charVector = trainingData.getCharVector();
		List<String> sequences = trainingData.getSequences();

		for (int i = 0; i < epochs; i++) {
			if (i > 0 && i % 30 == 0) {
				learningRate /= 2;
			}
			double error = 0;
			double num = 0;
			double start = System.currentTimeMillis();
			for (int j = 0; j < sequences.size(); j++) {
				String sequence = sequences.get(j);

				for (int a = 0; a < activations.length; a++) {
					Map<String, DoubleMatrix> acts = new HashMap<>();
					activations[a] =  acts;
				}
				//System.out.print(String.valueOf(sequence.charAt(0)));
				for (int t = 0; t < sequence.length() - 1; t++) {
					DoubleMatrix input = charVector.get(String.valueOf(sequence.charAt(t)));
					
					for (int k = 0; k < layers.length; k++) {
						activations[k].put("x" + t, input);
						layers[k].forward(t, activations[k]);
						
						if (k == layers.length - 1) {
							DoubleMatrix predictYt = layers[k].decode(activations[k].get("h" + t));
			                activations[k].put("py" + t, predictYt);
			                DoubleMatrix trueYt = charVector.get(String.valueOf(sequence.charAt(t + 1)));
			                activations[k].put("y" + t, trueYt);
			               // System.out.print(indexChar.get(predictYt.argmax()));
		                    error += LossFunction.getMeanCategoricalCrossEntropy(predictYt, trueYt);
						}
						input = activations[k].get("h" + t);
					}
				}
				//System.out.println();
				
				 // bptt
				boolean stacked = false;
				Map<String, DoubleMatrix> feedBack = new HashMap<>();
				for (int b = layers.length - 1; b >= 0; b--) {
					feedBack = layers[b].bptt(activations[b], sequence.length() - 2, learningRate, stacked, feedBack);
					stacked = true;
				}
				num +=  sequence.length();
			}
			System.out.println("Iter = " + i + ", error = " + error / num + ", time = " + (System.currentTimeMillis() - start) / 1000 + "s");
		}
	}
	
	//Select a character based on its probability
	public int RandomSample(DoubleMatrix X) {
		int index;
		double[] probabilities = new double[X.length];
		int[] indices = new int[X.length];
		for (int i = 0; i < X.length; i++) {
			double val = X.get(i);
			probabilities[i] = val;
			indices[i] = i;
		}
		EnumeratedIntegerDistribution dist = new EnumeratedIntegerDistribution(indices, probabilities);
		index = dist.sample();
		return index;
	}
	
	// Forward pass through the network
	public String feedForward(CharData ctext, int len,  String token, String stopSymbol) {
		Map<Integer, String> indexChar = ctext.getIndexChar();
		Map<String, DoubleMatrix> charVector = ctext.getCharVector();
		String s = "";
		s += token;
		for (int t = 0; t < len; t++) {
			DoubleMatrix input = charVector.get(String.valueOf(token.charAt(0)));
			for (int k = 0; k < layers.length; k++) {
				activations[k].put("x" + t, input);
				layers[k].forward(t, activations[k]);
				if (k == layers.length - 1) {
					DoubleMatrix predictYt = layers[k].decode(activations[k].get("h" + t), temperature);
					int index = RandomSample(predictYt);
					token = indexChar.get(index);
					s += token;
				}
				if (token.equals(stopSymbol))
					break;
				input = activations[k].get("h" + t);
			}
		}
		return s;
	}
	
	public void save() throws IOException {
		ObjectOutputStream objOutStream = new ObjectOutputStream(new FileOutputStream("LstmNetwork.bin"));
		objOutStream.writeObject(this);
		objOutStream.close();
	}
	
	public static LstmNetwork readInNetwork() throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream objInStream = new ObjectInputStream(new FileInputStream("LstmNetwork.bin"));
		LstmNetwork nn = (LstmNetwork) objInStream.readObject();
		objInStream.close();
		return nn;
	}

}
