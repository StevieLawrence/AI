package ann;

import java.util.Random;

public class SymbolsMain {
	private static final String[] SYMBOL_NAMES = { "plus", "minus", "backslash", "forwardslash", "X", "pike" };
	private static Symbol[] symbols;
	private static int[] hist = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	private static final int MAX_STEPS = 5000000;
	private static final double LEARNING_RATE = 0.5;
	private static final double MOMENTUM = 0.1;

	public static String classifier(double[] outputs, String[] selection) {
		if (outputs.length != selection.length)
			return null;
		double maxOutput = outputs[0];
		int maxIndex = 0;
		for (int i = 0; i < outputs.length; i++) {
			if (outputs[i] > maxOutput) {
				maxOutput = outputs[i];
				maxIndex = i;
			}
		}
		return selection[maxIndex];
	}

	public static NeuralNetwork teachSymbols(double target) {
		symbols = new Symbol[SYMBOL_NAMES.length];
		double[][] trainingSet = new double[symbols.length][];
		double[][] expectedOutputs = new double[symbols.length][symbols.length];
		for (int i = 0; i < symbols.length; i++) {
			symbols[i] = new Symbol(SYMBOL_NAMES[i] + ".png");
			trainingSet[i] = symbols[i].getPixels();
			double[] resultSet = new double[symbols.length];
			for (int output = 0; output < symbols.length; output++) {
				if (output == i)
					resultSet[output] = 0.9;
				else
					resultSet[output] = 0.1;
			}
			expectedOutputs[i] = resultSet;
		}

		symbols[5].saveImage();
		double targetError = target;
		int[] layers = { 25, 15, 6 };

		NeuralNetwork symNetwork = new NeuralNetwork(layers, LEARNING_RATE, MOMENTUM);
		symNetwork.learn(trainingSet, expectedOutputs, targetError, MAX_STEPS);
		return symNetwork;

	}

	public static String noiseTest(NeuralNetwork nn, Symbol sym, double trial) {

		Symbol sample = sym.copy(String.format("noiselvl%.02f%s", trial, ".png"));
		sample.addNoise(trial);
		double[] in = sample.getPixels();
		double[] ans = nn.calculateOutputs(in);
		String winner = classifier(ans, SYMBOL_NAMES);
		 System.out.println("for noise run " + String.format("%.02f", trial) + " network says it's a " + winner);
		 sample.saveImage();
		return winner;

	}

	public static void noiseTrail(NeuralNetwork nn, Symbol sym, String name) {
		String noNoise = classifier(nn.calculateOutputs(sym.getPixels()), SYMBOL_NAMES);
		if (noNoise.equals(name))
			hist[0]++;

		int i = 1;
		for (double trial = 0.1; trial <= 0.9; trial += 0.1) {
			String winner = noiseTest(nn, sym, trial);
			if (winner.equals(name))
				hist[i]++;
			i++;
		}
	}

	public static void printHist() {
		System.out.println("No Noise had " + hist[0]);
		int i = 1;
		for (double trial = 0.1; trial <= 0.9; trial += 0.1) {
			System.out.println(String.format("trial %.02f had %d", trial, hist[i]));
			i++;
		}
	}

	public static void clearHist() {
		for (int i = 0; i < hist.length; i++)
			hist[i] = 0;
	}

	public static void rmseTest(double rmse) {
		NeuralNetwork sym;
		int badApples = 0;
		int rangeN = 0;
		int totalRange = 0;
		int range01;

		while (rangeN < 100) {
			sym = teachSymbols(rmse);
			while (sym.iterations == MAX_STEPS) {
				sym = teachSymbols(rmse);
			}
			;
			clearHist();
			for (int in = 0; in < 10000; in++) {
				noiseTrail(sym, symbols[0], SYMBOL_NAMES[0]);
			}
			if (hist[0] < 10000) {
				badApples++;
			} else {
				rangeN++;
				range01 = hist[0] - hist[1];
				totalRange += range01;
				if (100 % rangeN == 4)
					;
				System.out.println(String.format("rangeN %d and totalRange %d", rangeN, totalRange));
			}
		}
		System.out.println("badApples " + badApples);
		System.out.println("rangeN " + rangeN + " and totalRange " + totalRange);
		System.out.println("avgRange " + (double) totalRange / (double) rangeN);
	}

	public static void main(String[] args) {
		NeuralNetwork symNetwork = teachSymbols(0.000001);
		//symNetwork = NeuralNetwork.readInNetwork();

		Random rand = new Random();
		int num = rand.nextInt(symbols.length);
		//String selected = SYMBOL_NAMES[num];
		Symbol selectedSym = symbols[num];
		System.out.println("Selected the symbol " + selectedSym.getFilname());

		double[] symInputs = selectedSym.getPixels();
		double[] results = symNetwork.calculateOutputs(symInputs);
		System.out.println("No noise test: network says it's a " + classifier(results, SYMBOL_NAMES));
		noiseTrail(symNetwork, symbols[4], SYMBOL_NAMES[4]);
	
		/* Performing noise test
		for (int f = 0; f < 6; f++) {
			clearHist();
			System.out.println(SYMBOL_NAMES[f]);
			for (int in = 0; in < 10000; in++) {
				noiseTrail(symNetwork, symbols[f], SYMBOL_NAMES[f]);
			}
			printHist();
		}
		*/
		
		//rmseTest(0.001);
	}

}
