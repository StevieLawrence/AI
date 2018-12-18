package swarmIntelligence;

import java.util.function.ToDoubleFunction;

public class PSOMain {

	public static void main(String[] args) {
		int SIZE = 20;
		double w = 0.7;
		double c1 = 2;
		double c2 = 2;
		int TEAMSIZE = 2;
		double[] LOWER_BOUNDS = { -5, -5 };
		double[] UPPER_BOUNDS = { 5, 5 };

		

		/*PSO pso = new PSO(SIZE, w, c1, c2, TEAMSIZE, LOWER_BOUNDS, UPPER_BOUNDS);
		pso.writeParticles("C:\\Users\\dmx\\Documents\\rastrigan0.txt");
		for (int i = 1; i < 4; i++) {
			pso.run(Func.rastrigan, Func.minimum, 10 * i);
			pso.writeParticles("C:\\Users\\dmx\\Documents\\rastrigan" + i + ".txt");
		}*/
		
		PSO pso = new PSO(SIZE, w , c1, c2, TEAMSIZE, LOWER_BOUNDS, UPPER_BOUNDS);
		pso.run(Func.himmelblau, Func.minimum, 100);
		pso.writeParticles("C:\\Users\\dmx\\Documents\\globalonly.txt");

		// double[] evaluate = {-4.4789735983, -4.6774339461447};
		double[] global = { 0.1316017921728463, 0.37753762356560916 };
		System.out.println(tester(Func.himmelblau, global));

	}

	public static double tester(ToDoubleFunction<double[]> tester, double[] param) {
		return tester.applyAsDouble(param);
	}
}
