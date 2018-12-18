package swarmIntelligence;

import java.util.function.BiPredicate;
import java.util.function.ToDoubleFunction;

public class Func {

	/**
	 * Ackley function, global minimum at the origin, recommended bounds -5 <= xi <= 5
	 */
	public static ToDoubleFunction<double[]> ackley = X -> {
		double sum1 = 0;
		double sum2 = 0;
		for (int i = 0 ; i < X.length ; i ++) {
	        sum1 += Math.pow(X[i], 2);
	        sum2 += (Math.cos(2*Math.PI*X[i]));
		}
		
		return -20.0*Math.exp(-0.2*Math.sqrt(sum1 / ((double)X.length))) + 20
                - Math.exp(sum2 /((double )X.length)) + Math.exp(1.0);
	};
	
	/**
	 * Rastrigin function has one global minimum at f(0,...,0), n-dimensional, recommended bounds -5.12 <= xi <= 5.12
	 */
	public static ToDoubleFunction<double[]> rastrigin = X -> {
		double A = 10.0;
		double n = X.length;
		double summation = 0;
		for (int i = 0; i < X.length; i++) {
			summation += Math.pow(X[i], 2) - (10 * Math.cos(2 * Math.PI * X[i]));
		}
		return (A * n) + summation;
	};
	
	/**
	 * Beale function, global minimum f(3,0.5), recommended bounds -4.5 <= x,y <= 4.5
	 */
	public static ToDoubleFunction<double[]> beale = cord -> Math.pow((1.5 - cord[0] + (cord[0] * cord[1])), 2)
			+ Math.pow((2.25 - cord[0] + (cord[0] * Math.pow(cord[1], 2))), 2) 
			+ Math.pow((2.625 - cord[0] +(cord[0] * Math.pow(cord[1], 3))), 2);
	
	/**
	 * Booth function, global minimum f(1,3), recommended bounds -10 <= x,y <= 10
	 */
	public static ToDoubleFunction<double[]> booth = cord -> Math.pow((cord[0] + 2 * cord[1] - 7), 2)
			+ Math.pow((2 * cord[0] + cord[1] - 5), 2);
	
	/**
	 * himmelblau function, global minimum at f(3,2), f(-2.805118,3.131312), f(-3.779310,-3.283186), 
	 * and f(3.584428, -1.848126) recommended bounds -5 <= x,y <= 5
	 */
	public static ToDoubleFunction<double[]> himmelblau = cord -> Math.pow((Math.pow(cord[0], 2) + cord[1] - 11), 2)
			+ Math.pow((cord[0] + Math.pow(cord[1], 2) - 7), 2);
	
	
	public static BiPredicate<Double, Double> minimum = (a, b) -> a < b;
	public static BiPredicate<Double, Double> maximum = (a,b) -> a > b;
}
