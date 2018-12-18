package ann;

public class Connection {
	private double weight, prevDeltaWeight; // previous change in weight is used to calculate momentum
	final Neuron fromNeuron, toNeuron;
	
	/**
	 * A connection taken place between a pre-synaptic neuron and a post-synaptic neuron
	 * @param fromNeuron the presynaptic neuron
	 * @param toNeuron the post synaptic neuron
	 */
	public Connection(Neuron fromNeuron, Neuron toNeuron) {
		this.fromNeuron = fromNeuron;
		this.toNeuron = toNeuron;
	}
	
	public double getWeight() {
		return weight;
	}
	
	public void setWeight(double weight) {
		this.weight = weight;
	}
	
	public double getPrevDeltaWeight() {
		return prevDeltaWeight;
	}
	
	public void setPrevDeltaWeight(double prevDeltaWeight) {
		this.prevDeltaWeight = prevDeltaWeight;
	}
	
	public Neuron getFromNeuron() {
		return fromNeuron;
	}
	
	public Neuron getToNeuron() {
		return toNeuron;
	}
	
	public String toString() {
		return fromNeuron.getLabel() + " - " + toNeuron.getLabel();
	}
	
	
}
