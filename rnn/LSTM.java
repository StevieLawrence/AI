package rnn;

import java.util.HashMap;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import java.util.List;
import java.util.Map;
import org.jblas.DoubleMatrix;
import ann.LossFunction;
public class LSTM {
	LstmCell state;

	public LSTM(int inSize, int outSize, MatIniter initer) {
		state = new LstmCell(inSize, outSize, initer);
	}

	public void train(CharData cData, double lr) {
		Map<Integer, String> indexChar = cData.getIndexChar();
		Map<String, DoubleMatrix> charVector = cData.getCharVector();
		List<String> sequence = cData.getSequences();
		for (int i = 0; i < 100; i++) {
			if (i > 0 && i % 20 == 0) {
				lr /= 2;
			}
			double error = 0;
			double num = 0;
			int size = sequence.size();
			double start = System.currentTimeMillis();
			for (int s = 0; s < size; s++) {
				String seq = sequence.get(s);
				if (seq.length() < 3) {
					continue;
				}

				Map<String, DoubleMatrix> acts = new HashMap<>();
				// forward pass
				if (i == 39 || i == 10 || i == 25)
					System.out.print(String.valueOf(seq.charAt(0)));
				for (int t = 0; t < seq.length() - 1; t++) {
					DoubleMatrix xt = charVector.get(String.valueOf(seq.charAt(t)));
					acts.put("x" + t, xt);

					state.forward(t, acts);

					DoubleMatrix predcitYt = state.decode(acts.get("h" + t));
					acts.put("py" + t, predcitYt);
					DoubleMatrix trueYt = charVector.get(String.valueOf(seq.charAt(t + 1)));
					acts.put("y" + t, trueYt);

					if (i == 39 || i == 10 || i == 25)
						System.out.print(indexChar.get(predcitYt.argmax()));
					error += LossFunction.getMeanCategoricalCrossEntropy(predcitYt, trueYt);

				}
				if (i == 39 || i == 10 || i == 25)
					System.out.println();

				// bptt
				state.bptt(acts, seq.length() - 2, lr);

				num += seq.length();
			}
			System.out.println("Iter = " + i + ", error = " + error / num + ", time = "
					+ (System.currentTimeMillis() - start) / 1000 + "s");
		}
	}

	// random sampling to choose a character from the probability distribution
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

	// forward pass for a sequence
	public String feedForward(CharData ctext, int len, String token, String stopSymbol) {
		Map<Integer, String> indexChar = ctext.getIndexChar();
		Map<String, DoubleMatrix> charVector = ctext.getCharVector();
		Map<String, DoubleMatrix> acts = new HashMap<>();
		String s = "";
		s += token;
		for (int t = 0; t < len; t++) {
			DoubleMatrix xt = charVector.get(String.valueOf(token.charAt(0)));
			acts.put("x" + t, xt);
			state.forward(t, acts);

			DoubleMatrix predcitYt = state.decode(acts.get("h" + t), 0.3);
			int index = RandomSample(predcitYt);
			
			token = indexChar.get(index);
			System.out.print(token);
			s += token;
			if (token.equals(stopSymbol)) {
				break;
			}
		}
		System.out.println();
		return s;
	}
	public String feedForward(CharData ctext, int len, String token) {
		return feedForward(ctext, len, token, null);
	}

	public static void main(String[] args) {
		//int hiddenSize = 100;
		//double lr = 1;
		//CharData ct = new CharData("TextFiles\\slinky.txt");
		//LSTM lstm = new LSTM(ct.getCharIndex().size(), hiddenSize, new MatIniter(MatIniter.Type.Uniform, 0.1, 0, 0));
		//int[] d = { ct.getCharIndex().size(), hiddenSize };
		//LstmNetwork l = new LstmNetwork(d, ct.getCharIndex().size(), new
		// MatIniter(MatIniter.Type.Uniform, 0.1, 0, 0));
		 //l.train(ct, 100, 1);
		// lstm.train(ct, lr);
	}
}
