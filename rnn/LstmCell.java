package rnn;

import java.util.HashMap;
import java.util.Map;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import ann.Activer;

public class LstmCell {
	
	private int inSize, hiddenSize, denseSize;

	// input gate
	private DoubleMatrix inputInputWeights, inputPrevHiddenWeights, inputPeepholeWeights, inputBias; 

	// forget gate
	private DoubleMatrix forgetInputWeights,forgetPrevHiddenWeights, forgetPeepholeWeights, forgetBias;

	// input activation
	private DoubleMatrix actInputWeights, actPrevHiddenWeights, actBias;

	// output gate
	private DoubleMatrix outputInputWeights, outputPrevHiddenWeights, outputPeepholeWeights, outputBias;

	// output layer 
	private DoubleMatrix denseLayerWeights, denseLayerBias;

	public LstmCell(int inSize, int hiddenSize, MatIniter initer) {
		this.inSize = inSize;
		this.hiddenSize = hiddenSize;

		if (initer.getType() == MatIniter.Type.Uniform) {
			this.inputInputWeights = initer.uniform(inSize, hiddenSize);
			this.inputPrevHiddenWeights = initer.uniform(hiddenSize, hiddenSize);
			this.inputPeepholeWeights = initer.uniform(hiddenSize, hiddenSize);
			this.inputBias = new DoubleMatrix(1, hiddenSize);

			this.forgetInputWeights = initer.uniform(inSize, hiddenSize);
			this.forgetPrevHiddenWeights = initer.uniform(hiddenSize, hiddenSize);
			this.forgetPeepholeWeights = initer.uniform(hiddenSize, hiddenSize);
			this.forgetBias = new DoubleMatrix(1, hiddenSize);

			this.actInputWeights = initer.uniform(inSize, hiddenSize);
			this.actPrevHiddenWeights = initer.uniform(hiddenSize, hiddenSize);
			this.actBias = new DoubleMatrix(1, hiddenSize);

			this.outputInputWeights = initer.uniform(inSize, hiddenSize);
			this.outputPrevHiddenWeights = initer.uniform(hiddenSize, hiddenSize);
			this.outputPeepholeWeights = initer.uniform(hiddenSize, hiddenSize);
			this.outputBias = new DoubleMatrix(1, hiddenSize);

			this.denseLayerWeights = initer.uniform(hiddenSize, inSize);
			this.denseLayerBias = new DoubleMatrix(1, inSize);
		} else if (initer.getType() == MatIniter.Type.Gaussian) {
			this.inputInputWeights = initer.gaussian(inSize, hiddenSize);
			this.inputPrevHiddenWeights = initer.gaussian(hiddenSize, hiddenSize);
			this.inputPeepholeWeights = initer.gaussian(hiddenSize, hiddenSize);
			this.inputBias = new DoubleMatrix(1, hiddenSize);

			this.forgetInputWeights = initer.gaussian(inSize, hiddenSize);
			this.forgetPrevHiddenWeights = initer.gaussian(hiddenSize, hiddenSize);
			this.forgetPeepholeWeights = initer.gaussian(hiddenSize, hiddenSize);
			this.forgetBias = new DoubleMatrix(1, hiddenSize);

			this.actInputWeights = initer.gaussian(inSize, hiddenSize);
			this.actPrevHiddenWeights = initer.gaussian(hiddenSize, hiddenSize);
			this.actBias = new DoubleMatrix(1, hiddenSize);

			this.outputInputWeights = initer.gaussian(inSize, hiddenSize);
			this.outputPrevHiddenWeights = initer.gaussian(hiddenSize, hiddenSize);
			this.outputPeepholeWeights = initer.gaussian(hiddenSize, hiddenSize);
			this.outputBias = new DoubleMatrix(1, hiddenSize);

			this.denseLayerWeights = initer.gaussian(hiddenSize, inSize);
			this.denseLayerBias = new DoubleMatrix(1, inSize);
		}
	}

	public LstmCell(int inSize, int hiddenSize, MatIniter initer, int deSize) {
		this(inSize, hiddenSize, initer);
		this.denseSize = deSize;
		this.denseLayerWeights = new DoubleMatrix(hiddenSize, deSize);
		this.denseLayerBias = new DoubleMatrix(1, deSize);
	}

	public int getInSize() {
		return inSize;
	}

	public int getHiddenSize() {
		return hiddenSize;
	}

	public int getDenseSize() {
		return denseSize;
	}

	// a forward pass through the cell
	public void forward(int t, Map<String, DoubleMatrix> acts) {
		DoubleMatrix x = acts.get("x" + t);
		DoubleMatrix preH = null, preC = null;
		if (t == 0) {
			preH = new DoubleMatrix(1, getDenseSize());
			preC = preH.dup();
		} else {
			preH = acts.get("h" + (t - 1));
			preC = acts.get("c" + (t - 1));
		}

		DoubleMatrix inputGate = Activer.logistic(x.mmul(inputInputWeights).add(preH.mmul(inputPrevHiddenWeights)).add(preC.mmul(inputPeepholeWeights)).add(inputBias));
		DoubleMatrix forgetGate = Activer.logistic(x.mmul(forgetInputWeights).add(preH.mmul(forgetPrevHiddenWeights)).add(preC.mmul(forgetPeepholeWeights)).add(forgetBias));
		DoubleMatrix inputAct = Activer.tanh(x.mmul(actInputWeights).add(preH.mmul(actPrevHiddenWeights)).add(actBias));
		DoubleMatrix cellMemory = forgetGate.mul(preC).add(inputGate.mul(inputAct));
		DoubleMatrix outputGate = Activer.logistic(x.mmul(outputInputWeights).add(preH.mmul(outputPrevHiddenWeights)).add(cellMemory.mmul(outputPeepholeWeights)).add(outputBias));
		DoubleMatrix gh = Activer.tanh(cellMemory);
		DoubleMatrix hidden = outputGate.mul(gh);

		acts.put("i" + t, inputGate);
		acts.put("f" + t, forgetGate);
		acts.put("ac" + t, inputAct);
		acts.put("c" + t, cellMemory);
		acts.put("o" + t, outputGate);
		acts.put("gh" + t, gh);
		acts.put("h" + t, hidden);
	}

	// back propagation through time
	public Map<String, DoubleMatrix> bptt(Map<String, DoubleMatrix> acts, int lastT, double lr, boolean stacked, Map<String, DoubleMatrix> dxt) {
		Map<String, DoubleMatrix> pass = new HashMap<>();
		for (int t = lastT; t > -1; t--) {
			DoubleMatrix deltaY;
			DoubleMatrix h = acts.get("h" + t);
			DoubleMatrix deltaH = null;
			if (!stacked) {
				DoubleMatrix py = acts.get("py" + t);
				DoubleMatrix y = acts.get("y" + t);
				deltaY = py.sub(y);
				acts.put("dy" + t, deltaY);

				// cell output errors
				if (t == lastT) {
					deltaH = denseLayerWeights.mmul(deltaY.transpose()).transpose();
				} else {
					DoubleMatrix lateDac = acts.get("dac" + (t + 1));
					DoubleMatrix lateDf = acts.get("df" + (t + 1));
					DoubleMatrix lateDo = acts.get("do" + (t + 1));
					DoubleMatrix lateDi = acts.get("di" + (t + 1));
					deltaH = denseLayerWeights.mmul(deltaY.transpose()).transpose().add(actPrevHiddenWeights.mmul(lateDac.transpose()).transpose())
							.add(inputPrevHiddenWeights.mmul(lateDi.transpose()).transpose()).add(outputPrevHiddenWeights.mmul(lateDo.transpose()).transpose());
				}
			} else {
				// top down to stacked lstm
				deltaY = dxt.get("dx" + t);
				if (t == lastT) {
					deltaH = DoubleMatrix.zeros(h.rows, h.columns);
				} else {
					DoubleMatrix lateDac = acts.get("dac" + (t + 1));
					DoubleMatrix lateDf = acts.get("df" + (t + 1));
					DoubleMatrix lateDo = acts.get("do" + (t + 1));
					DoubleMatrix lateDi = acts.get("di" + (t + 1));
					deltaH = actPrevHiddenWeights.mmul(lateDac.transpose()).transpose().add(inputPrevHiddenWeights.mmul(lateDi.transpose()).transpose())
							.add(outputPrevHiddenWeights.mmul(lateDo.transpose()).transpose())
							.add(forgetPrevHiddenWeights.mmul(lateDf.transpose()).transpose());
				}
			}

			acts.put("dh" + t, deltaH);

			// output gate
			DoubleMatrix gh = acts.get("gh" + t);
			DoubleMatrix o = acts.get("o" + t);
			DoubleMatrix deltaO = deltaH.mul(gh).mul(deriveExp(o));
			acts.put("do" + t, deltaO);

			// status
			DoubleMatrix deltaC = null;
			if (t == lastT) {
				deltaC = deltaH.mul(o).mul(deriveTanh(gh)).add(outputPeepholeWeights.mmul(deltaO.transpose()).transpose());
			} else {
				DoubleMatrix lateDc = acts.get("dc" + (t + 1));
				DoubleMatrix lateDf = acts.get("df" + (t + 1));
				DoubleMatrix lateF = acts.get("f" + (t + 1));
				DoubleMatrix lateDi = acts.get("di" + (t + 1));
				deltaC = deltaH.mul(o).mul(deriveTanh(gh)).add(outputPeepholeWeights.mmul(deltaO.transpose()).transpose())
						.add(lateF.mul(lateDc)).add(forgetPeepholeWeights.mmul(lateDf.transpose()).transpose())
						.add(inputPeepholeWeights.mmul(lateDi.transpose()).transpose());
			}
			acts.put("dc" + t, deltaC);

			// cells
			DoubleMatrix ac = acts.get("ac" + t);
			DoubleMatrix i = acts.get("i" + t);
			DoubleMatrix deltaAc = deltaC.mul(i).mul(deriveTanh(ac));
			acts.put("dac" + t, deltaAc);

			DoubleMatrix preC = null;
			if (t > 0) {
				preC = acts.get("c" + (t - 1));
			} else {
				preC = DoubleMatrix.zeros(1, h.length);
			}
			// forget gates
			DoubleMatrix f = acts.get("f" + t);
			DoubleMatrix deltaF = deltaC.mul(preC).mul(deriveExp(f));
			acts.put("df" + t, deltaF);
			// input gates
			DoubleMatrix deltaI = deltaC.mul(ac).mul(deriveExp(i));
			acts.put("di" + t, deltaI);
			
			// stacked lstm
			DoubleMatrix di = deltaI.dup();
			DoubleMatrix df = deltaF.dup();
			DoubleMatrix dO = deltaO.dup();
			DoubleMatrix dac = deltaAc.dup();
			DoubleMatrix deltaX = actInputWeights.mmul(dac.transpose()).transpose().add(inputInputWeights.mmul(di.transpose()).transpose())
					.add(outputInputWeights.mmul(dO.transpose()).transpose())
					.add(forgetInputWeights.mmul(df.transpose()).transpose());
			pass.put("dx" + t, deltaX);
			
		}
		updateParameters(acts, lastT, lr, stacked);
		return pass;
	}

	public void bptt(Map<String, DoubleMatrix> acts, int lastT, double lr) {
		bptt(acts, lastT, lr, false, null);
	}

	private void updateParameters(Map<String, DoubleMatrix> acts, int lastT, double lr, boolean stacked) {
		// initialize deletas for the weight matrices
		DoubleMatrix gWxi = new DoubleMatrix(inputInputWeights.rows, inputInputWeights.columns);
		DoubleMatrix gWhi = new DoubleMatrix(inputPrevHiddenWeights.rows, inputPrevHiddenWeights.columns);
		DoubleMatrix gWci = new DoubleMatrix(inputPeepholeWeights.rows, inputPeepholeWeights.columns);
		DoubleMatrix gbi = new DoubleMatrix(inputBias.rows, inputBias.columns);

		DoubleMatrix gWxf = new DoubleMatrix(forgetInputWeights.rows, forgetInputWeights.columns);
		DoubleMatrix gWhf = new DoubleMatrix(forgetPrevHiddenWeights.rows, forgetPrevHiddenWeights.columns);
		DoubleMatrix gWcf = new DoubleMatrix(forgetPeepholeWeights.rows, forgetPeepholeWeights.columns);
		DoubleMatrix gbf = new DoubleMatrix(forgetBias.rows, forgetBias.columns);

		DoubleMatrix gWxc = new DoubleMatrix(actInputWeights.rows, actInputWeights.columns);
		DoubleMatrix gWhc = new DoubleMatrix(actPrevHiddenWeights.rows, actPrevHiddenWeights.columns);
		DoubleMatrix gbc = new DoubleMatrix(actBias.rows, actBias.columns);

		DoubleMatrix gWxo = new DoubleMatrix(outputInputWeights.rows, outputInputWeights.columns);
		DoubleMatrix gWho = new DoubleMatrix(outputPrevHiddenWeights.rows, outputPrevHiddenWeights.columns);
		DoubleMatrix gWco = new DoubleMatrix(outputPeepholeWeights.rows, outputPeepholeWeights.columns);
		DoubleMatrix gbo = new DoubleMatrix(outputBias.rows, outputBias.columns);

		DoubleMatrix gWhy = new DoubleMatrix(denseLayerWeights.rows, denseLayerWeights.columns);
		DoubleMatrix gby = new DoubleMatrix(denseLayerBias.rows, denseLayerBias.columns);

		// calculate the accumalated weight changes
		for (int t = 0; t < lastT + 1; t++) {
			DoubleMatrix x = acts.get("x" + t).transpose();
			gWxi = gWxi.add(x.mmul(acts.get("di" + t)));
			gWxf = gWxf.add(x.mmul(acts.get("df" + t)));
			gWxc = gWxc.add(x.mmul(acts.get("dac" + t)));
			gWxo = gWxo.add(x.mmul(acts.get("do" + t)));

			if (t > 0) {
				DoubleMatrix preH = acts.get("h" + (t - 1)).transpose();
				DoubleMatrix preC = acts.get("c" + (t - 1)).transpose();
				gWhi = gWhi.add(preH.mmul(acts.get("di" + t)));
				gWhf = gWhf.add(preH.mmul(acts.get("df" + t)));
				gWhc = gWhc.add(preH.mmul(acts.get("dac" + t)));
				gWho = gWho.add(preH.mmul(acts.get("do" + t)));
				gWci = gWci.add(preC.mmul(acts.get("di" + t)));
				gWcf = gWcf.add(preC.mmul(acts.get("df" + t)));
			}
			gWco = gWco.add(acts.get("c" + t).transpose().mmul(acts.get("do" + t)));

			gbi = gbi.add(acts.get("di" + t));
			gbf = gbf.add(acts.get("df" + t));
			gbc = gbc.add(acts.get("dac" + t));
			gbo = gbo.add(acts.get("do" + t));

			if (!stacked) {
				gWhy = gWhy.add(acts.get("h" + t).transpose().mmul(acts.get("dy" + t)));
				gby = gby.add(acts.get("dy" + t));
			}
		}
		// clipping to avoid expanding gradient problem
		inputInputWeights = inputInputWeights.sub(clip(gWxi.div(lastT)).mul(lr));
		inputPrevHiddenWeights = inputPrevHiddenWeights.sub(clip(gWhi.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
		inputPeepholeWeights = inputPeepholeWeights.sub(clip(gWci.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
		inputBias = inputBias.sub(clip(gbi.div(lastT)).mul(lr));

		forgetInputWeights = forgetInputWeights.sub(clip(gWxf.div(lastT)).mul(lr));
		forgetPrevHiddenWeights = forgetPrevHiddenWeights.sub(clip(gWhf.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
		forgetPeepholeWeights = forgetPeepholeWeights.sub(clip(gWcf.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
		forgetBias = forgetBias.sub(clip(gbf.div(lastT)).mul(lr));

		actInputWeights = actInputWeights.sub(clip(gWxc.div(lastT)).mul(lr));
		actPrevHiddenWeights = actPrevHiddenWeights.sub(clip(gWhc.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
		actBias = actBias.sub(clip(gbc.div(lastT)).mul(lr));

		outputInputWeights = outputInputWeights.sub(clip(gWxo.div(lastT)).mul(lr));
		outputPrevHiddenWeights = outputPrevHiddenWeights.sub(clip(gWho.div(lastT < 2 ? 1 : (lastT - 1))).mul(lr));
		outputPeepholeWeights = outputPeepholeWeights.sub(clip(gWco.div(lastT)).mul(lr));
		outputBias = outputBias.sub(clip(gbo.div(lastT)).mul(lr));

		if (!stacked) {
			denseLayerWeights = denseLayerWeights.sub(clip(gWhy.div(lastT)).mul(lr));
			denseLayerBias = denseLayerBias.sub(clip(gby.div(lastT)).mul(lr));
		}
	}

	private DoubleMatrix deriveExp(DoubleMatrix f) {
		return f.mul(DoubleMatrix.ones(1, f.length).sub(f));
	}

	private DoubleMatrix deriveTanh(DoubleMatrix f) {
		return DoubleMatrix.ones(1, f.length).sub(MatrixFunctions.pow(f, 2));
	}

	private DoubleMatrix clip(DoubleMatrix x) {
		double v = 10;
		return x.mul(x.ge(-v).mul(x.le(v)));
		// return x;
	}

	public DoubleMatrix decode(DoubleMatrix ht) {
		return Activer.softmax(ht.mmul(denseLayerWeights).add(denseLayerBias));
	}
	
	public DoubleMatrix decode(DoubleMatrix ht, double temp) {
		return Activer.softmax(ht.mmul(denseLayerWeights).add(denseLayerBias), temp);
	}

}
