package ann;

import java.util.ArrayList;
import java.util.Random;

public class Neuron {
	private String label;
	private double output, biasWeight, error, prevDeltaBiasWeight;
	private Connection[] inputConnections;
	private ArrayList<Connection> outputConnections;

	/**
	 * A Neuron in the netork that has a threshold (bias) and an activation (output)
	 * given its synaptic inputs (weights and previous layer outputs)
	 * 
	 * @param label
	 *            String name for the neuron
	 */
	public Neuron(String label) {
		this.label = label;
		outputConnections = new ArrayList<Connection>();
		prevDeltaBiasWeight = 0;
	}

	public String getLabel() {
		return label;
	}

	/**
	 * Connect pre-synaptic neurons to a post-synaptic neuron
	 * 
	 * @param sourceNeurons
	 *            array of all of the pre-synaptic neurons
	 */
	public void inputConnections(Neuron[] sourceNeurons) {
		Random rand = new Random();
		inputConnections = new Connection[sourceNeurons.length];
		// for every pre-synaptic neuron
		for (int i = 0; i < sourceNeurons.length; i++) {
			Connection connection = new Connection(sourceNeurons[i], this); // form a connection
			inputConnections[i] = connection; // mark it as a pre-synaptic neuron
			// add this neuron to the pre-synaptic neuron's post-synaptic neurons
			inputConnections[i].fromNeuron.outputConnections.add(inputConnections[i]);
			inputConnections[i].setWeight((rand.nextDouble() * 2) - 1); // initialize a random weight between -1 and 1
		}
	}

	public Connection[] getInputConnections() {
		return inputConnections;
	}

	public ArrayList<Connection> getOutputConnections() {
		return outputConnections;
	}

	public double getBiasWeight() {
		return biasWeight;
	}

	public void setBiasWeight(double biasWeight) {
		this.biasWeight = biasWeight;
	}

	/**
	 * For an input neuron set its activation to the input
	 * 
	 * @param input
	 */
	public void input(double input) {
		this.output = input;
	}

	public double getOutput() {
		return output;
	}

	public double getError() {
		return error;
	}

	/**
	 * The transformation function used in computing the activation (output)
	 * 
	 * @param value
	 *            the neurons output
	 * @return a value between -1 and 1
	 */
	public double sigmoid(double value) {
		return 1.0 / (1.0 + (Math.exp(-value)));
	}

	/**
	 * s(x)' = s(x)(1 - s(x))
	 * @param sigmoidValue
	 *            the output from passing a value into the sigmoid function
	 * @return The derivative value of the sigmoid function
	 */
	private double sigmoidDeriv(double sigmoidValue) {
		return sigmoidValue * (1.0 - sigmoidValue);
	}

	/**
	 * Multiply the corresponding weights and inputs and add them up. Then pass the result
	 * to the transformation function.
	 */
	public void calculateOutput() {
		double value = biasWeight;

		for (Connection iCon : inputConnections) {
			value += iCon.getWeight() * iCon.getFromNeuron().getOutput();
		}

		output = sigmoid(value);
	}

	/**
	 * For output neurons the error is (target - actual)*actual(1 - actual)
	 * @param target the expected output value
	 */
	public void caluculateError(double target) {
		error = (target - output) * sigmoidDeriv(output);
	}

	/**
	 * For hidden layer neurons the error is actual(1 - actual) * SUM(posSynapticNeuronError*weight)
	 */
	public void calculateError() {
		error = sigmoidDeriv(output);
		double wkj = 0;
		for (Connection oCon : outputConnections) {
			wkj += oCon.getToNeuron().getError() * oCon.getWeight();
		}
		if (wkj != 0)
			error *= wkj;
	}

	/**
	 * The bias weight is updated by adding the learningRate * neuronError * 1 (1 being the bias)
	 * @param learningRate - the incremental adjustment rate
	 * @param momentum - added on term that may help the network learn faster
	 */
	public void updateBiasWeight(double learningRate, double momentum) {
		double deltaBiasWeight = learningRate * error;
		biasWeight = biasWeight + deltaBiasWeight + (momentum * prevDeltaBiasWeight);
		prevDeltaBiasWeight = deltaBiasWeight;
	}
	
	/**
	 * 
	 * @param learningRate - the incremental adjustment rate
	 * @param momentum - added on term that may help the network learn faster
	 */
	public void updateWeights(double learningRate, double momentum) {
		for (Connection con : inputConnections) {
			double deltaWeight = learningRate * error * con.getFromNeuron().getOutput();
			con.setWeight(con.getWeight() + deltaWeight + (momentum * con.getPrevDeltaWeight()));
			con.setPrevDeltaWeight(deltaWeight);
		}
	}
}
