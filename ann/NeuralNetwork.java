package ann;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;

public class NeuralNetwork implements Serializable {
	static final long serialVersionUID = 1L;
	private Neuron[] inputLayer, outputLayer;
	private Neuron[][] hiddenLayers;
	private double learningRate, momentum;
	public final int[] LAYER_DATA;
	public final int INPUT_SIZE, OUTPUT_SIZE, LAYER_COUNT, HIDDEN_LAYER_COUNT;
	private Random rand;
	int iterations;

	/**
	 * A network of neurons that uses backpropagation to undergo supervised learning
	 * 
	 * @param layerData
	 *            - array consisting of the network layers and how many neurons in
	 *            each
	 * @param learningRate
	 *            - incremental adjustment rate applied to the change in the weights
	 * @param momentum
	 *            - added on term to help the network learn faster
	 */
	public NeuralNetwork(int[] layerData, double learningRate, double momentum) {
		this.LAYER_DATA = layerData;
		this.LAYER_COUNT = LAYER_DATA.length;
		this.HIDDEN_LAYER_COUNT = LAYER_COUNT - 2;
		this.INPUT_SIZE = LAYER_DATA[0];
		this.OUTPUT_SIZE = LAYER_DATA[LAYER_COUNT - 1];
		this.learningRate = learningRate;
		this.momentum = momentum;
		rand = new Random();

		// initialize the network layers
		inputLayer = new Neuron[INPUT_SIZE];
		hiddenLayers = new Neuron[HIDDEN_LAYER_COUNT][];
		outputLayer = new Neuron[OUTPUT_SIZE];

		// initialize the input neurons into the input layer
		for (int i = 0; i < INPUT_SIZE; i++) {
			inputLayer[i] = new Neuron("In" + (i + 1));
		}

		// initialize the hidden neurons and set connections with the input/previous
		// hidden layer
		for (int j = 0; j < HIDDEN_LAYER_COUNT; j++) {
			int layerSize = LAYER_DATA[j + 1];
			hiddenLayers[j] = new Neuron[layerSize];
			for (int i = 0; i < layerSize; i++) {
				hiddenLayers[j][i] = new Neuron("Hi" + (j + 1) + "." + (i + 1));

				if (j == 0) {
					hiddenLayers[j][i].inputConnections(inputLayer); // first hidden layer
				} else {
					hiddenLayers[j][i].inputConnections(hiddenLayers[j - 1]);
				}
				hiddenLayers[j][i].setBiasWeight((rand.nextDouble() * 2) - 1);

			}
		}

		// initialize output neurons and connections with the previous hidden layer
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			outputLayer[i] = new Neuron("Ou" + (i + 1));
			outputLayer[i].inputConnections(hiddenLayers[HIDDEN_LAYER_COUNT - 1]);
			outputLayer[i].setBiasWeight((rand.nextDouble() * 2) - 1);
		}
	}

	/**
	 * Feedforward process of finding the activation for the inputs -> hidden layer
	 * -> outputs
	 * 
	 * @param inputs
	 *            - the inputs into the neural network
	 * @return - array of the output values of the neural network
	 */
	public double[] calculateOutputs(double[] inputs) {
		if (inputs.length != INPUT_SIZE)
			return null;

		// enter inputs
		for (int i = 0; i < INPUT_SIZE; i++) {
			inputLayer[i].input(inputs[i]);
		}

		// calculate activations of the hidden layer(s)
		for (int hLayer = 0; hLayer < HIDDEN_LAYER_COUNT; hLayer++) {
			for (int i = 0; i < hiddenLayers[hLayer].length; i++) {
				hiddenLayers[hLayer][i].calculateOutput();
			}
		}
		// calculate the activations of the output neurons
		double[] outputs = new double[OUTPUT_SIZE];
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			outputLayer[i].calculateOutput();
			outputs[i] = outputLayer[i].getOutput();
		}
		return outputs;
	}

	/**
	 * propagate the error backwards through the network
	 * 
	 * @param targets
	 *            - the expected output(s)
	 * @param numOfPatterns
	 *            - Total number of patters used in RMSE calculation
	 * @return The RMSE error ie. sqrt( (2 * TSSE) / N)
	 */
	public double calculateErrors(double[] targets, int numOfPatterns) {
		// TSSE = 1/2 * SUM( (target - actual)^2 )
		double tsseDoubled = 0;
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			//// calculate and update all of the output neurons error
			outputLayer[i].caluculateError(targets[i]);
			tsseDoubled += Math.pow(outputLayer[i].getError(), 2); // calculate 2 * TSSE
		}

		// Calculate and update the error for all of the hidden layer neurons
		for (int hLayer = HIDDEN_LAYER_COUNT - 1; hLayer >= 0; hLayer--) {
			for (int i = 0; i < hiddenLayers[hLayer].length; i++) {
				hiddenLayers[hLayer][i].calculateError();
			}
		}
		double rmse = Math.sqrt(tsseDoubled / (double) (numOfPatterns)); // calc. RMSE
		return rmse;
	}
	
	public double[] getInGrad() {
		double[] g = new double[INPUT_SIZE];
		for (int i = 0; i < INPUT_SIZE; i++) {
			inputLayer[i].calculateError();
			g[i] = inputLayer[i].getError();
		}
		return g;
	}

	/**
	 * Adjust the weights in the network (learn) based on the error
	 */
	public void calculateWeights() {
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			outputLayer[i].updateBiasWeight(learningRate, momentum);
			outputLayer[i].updateWeights(learningRate, momentum);
		}

		for (int hLayer = HIDDEN_LAYER_COUNT - 1; hLayer >= 0; hLayer--) {
			for (int i = 0; i < hiddenLayers[hLayer].length; i++) {
				hiddenLayers[hLayer][i].updateBiasWeight(learningRate, momentum);
				hiddenLayers[hLayer][i].updateWeights(learningRate, momentum);
			}
		}
	}

	/**
	 * Train the network to some RMSE value or until a max number of iterations have
	 * been reached
	 * 
	 * @param trainingSet
	 *            - The inputs
	 * @param expectedOutputs
	 *            The pairs to the inputs
	 * @param targetError
	 *            - RMSE value
	 * @param maxSteps
	 *            - how many iterations until stop
	 * @return - epochs
	 */
	public int learn(double[][] trainingSet, double[][] expectedOutputs, double targetError, int maxSteps) {
		int step = 0;
		int patterns = trainingSet.length;
		double error = 10000;

		// stochastic mode ie. batch = 1
		while (error > targetError && step < maxSteps) {
			int index = rand.nextInt(trainingSet.length); // random training vector
			// perform backpropagation
			double[] inputs = trainingSet[index];
			double[] targets = expectedOutputs[index];
			calculateOutputs(inputs);
			error = calculateErrors(targets, patterns);
			calculateWeights();
			step++;
		}
		iterations = step;
		int epochs = step / patterns;
		System.out.println("It took " + step + " iterations and " + epochs + " epochs");
		return epochs;
	}

	/**
	 * Save the network
	 * 
	 * @throws IOException
	 */
	public void save() throws IOException {
		ObjectOutputStream objOutStream = new ObjectOutputStream(new FileOutputStream("NeuralNetwork.bin"));
		objOutStream.writeObject(this);
		objOutStream.close();
	}

	/**
	 * Read in the network
	 * 
	 * @return a NeuralNetwork
	 * @throws FileNotFoundException
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static NeuralNetwork readInNetwork() throws FileNotFoundException, IOException, ClassNotFoundException {
		ObjectInputStream objInStream = new ObjectInputStream(new FileInputStream("NeuralNetwork.bin"));
		NeuralNetwork nn = (NeuralNetwork) objInStream.readObject();
		objInStream.close();
		return nn;
	}

}
