package swarmIntelligence;

import java.util.Arrays;
import java.util.Random;

public class Particle {
	private String label;
	private int dimen;
	private double[] personalBest;
	private double[] teamsBest;
	private double[] velocity;
	private double[] position;
	private double w, c1, c2, pBestFitness;

	/**
	 * A potential solution in a particle swarm.
	 * 
	 * @param label
	 *            string for naming the particle
	 * @param position
	 *            double array holding all of the coordinates (multi-dimensional)
	 * @param velocity
	 *            velocity vector for updating position. Influenced by current
	 *            position, personal best, and team's best
	 * @param w
	 *            inertia constant
	 * @param c1
	 *            personal best constant
	 * @param c2
	 *            global best constant
	 */
	Particle(String label, double[] position, double[] velocity, double w, double c1, double c2) {
		this.label = label;
		this.position = position;
		dimen = position.length;
		personalBest = Arrays.copyOf(position, dimen);
		this.w = w;
		this.c1 = c1;
		this.c2 = c2;
		this.velocity = velocity;
	}

	public double[] getTeamsBest() {
		return teamsBest;
	}

	public void setTeamsBest(double[] teamsBest) {
		this.teamsBest = teamsBest;
	}

	public double[] getPersonalBest() {
		return personalBest;
	}

	public void setPersonalBest(double[] personalBest) {
		this.personalBest = personalBest;
	}

	public double[] getVelocity() {
		return velocity;
	}

	public String getLabel() {
		return label;
	}

	public double[] getPosition() {
		return position;
	}
	
	public int getDimen() {
		return dimen;
	}
	
	public double getPBestFitness() {
		return pBestFitness;
	}
	
	public void setPBestFitness(double pBestFitness) {
		this.pBestFitness = pBestFitness;
	}

	/**
	 * Vt+1 = w * rand[0,1] * Vt + c1 * rand[0,1] * (personalBest - Xt) + c2 *
	 * rand[0,1] * (teamsBest - Xt)
	 */
	private void updateVelocity() {
		Random rand = new Random();
		double r1, r2, r3;
		r1 = rand.nextDouble();
		r2 = rand.nextDouble();
		r3 = rand.nextDouble();
		for (int i = 0; i < velocity.length; i++) {
			velocity[i] = w * r1 * velocity[i] + c1 * r2 * (personalBest[i] - position[i])
					+ c2 * r3 * (teamsBest[i] - position[i]);
		}
	}

	/**
	 * Xt+1 = Xt + Vt+1
	 */
	public void updatePosition(double[] lowerBounds, double[]upperBounds) {
		updateVelocity();
		for (int i = 0; i < position.length; i++) {
			position[i] = position[i] + velocity[i];
		}
		matchConstraints(lowerBounds, upperBounds);
	}
	
	
	/**
	 * Set bounds to make sure the particle doesn't leave the search space
	 * @param lowerBounds
	 * @param upperBounds
	 */
	public void matchConstraints(double[] lowerBounds, double[]upperBounds) {
		for (int i = 0; i < position.length; i++) {
			position[i] = Math.min(position[i], upperBounds[i]);
			position[i] = Math.max(position[i], lowerBounds[i]);
		}
	}

	@Override
	public String toString() {
		return label + " position:" + Arrays.toString(position) + " personalBest:" + Arrays.toString(personalBest)
				+ " teamBest:" + Arrays.toString(teamsBest);
	}
}
