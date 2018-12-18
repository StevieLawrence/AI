package swarmIntelligence;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.function.BiPredicate;
import java.util.function.ToDoubleFunction;

public class PSO {
	private double[] lowerBounds;
	private double[] upperBounds;
	int size;
	double w, c1, c2;
	private Particle[] swarm;
	double globalBest;
	private ArrayList<ArrayList<Particle>> teams;
	private double[] teamBests;

	/**
	 * Optimizes a problem by iteratively updating candidate particle solutions
	 * based on their own best and their team/global best
	 * 
	 * @param size
	 *            How many particles
	 * @param w
	 *            inertia scaler factor
	 * @param c1
	 *            personal best scaler factor
	 * @param c2
	 *            team best scaler factor
	 * @param teamSize
	 *            Size of a team
	 * @param lowerBounds
	 *            the lower bounds of the problem
	 * @param upperBounds
	 *            the upper bounds of the problem
	 */
	public PSO(int size, double w, double c1, double c2, int teamSize, double[] lowerBounds, double[] upperBounds) {
		this.lowerBounds = lowerBounds;
		this.upperBounds = upperBounds;
		this.size = size;
		this.w = w;
		this.c1 = c1;
		this.c2 = c2;
		swarm = new Particle[size];
		if (teamSize < 1 || teamSize > size) { // check to see the team size is legal
			initSwarm(1);
		} else {
			initSwarm(teamSize);
		}
		System.out.println("Initialized Swarm");
		printSwarm();
		System.out.println();
	}

	/**
	 * Initialize the swarm
	 * 
	 * @param teamSize
	 *            size of a team of particles
	 */
	private void initSwarm(int teamSize) {
		Random rand = new Random();

		// Initialize the particles
		for (int i = 0; i < size; i++) {
			double[] pos = new double[lowerBounds.length];
			double[] vel = new double[lowerBounds.length];
			for (int j = 0; j < lowerBounds.length; j++) {
				// init position and velocity
				pos[j] = lowerBounds[j] + (rand.nextDouble() * (upperBounds[j] - lowerBounds[j]));
				vel[j] = 0;
			}
			Particle p = new Particle("" + i, pos, vel, w, c1, c2);
			p.setPersonalBest(pos);
			swarm[i] = p;
		}

		// Global PSO
		if (teamSize == 1) {
			teams = null;
			teamBests = null;
			double[] tBest = new double[lowerBounds.length];
			// initialize a global best
			for (int t = 0; t < lowerBounds.length; t++) {
				tBest[t] = lowerBounds[t] + (rand.nextDouble() * (upperBounds[t] - lowerBounds[t]));
			}
			updateGlobalBest(tBest);

		} else {
			// Initialize the teams
			teams = new ArrayList<ArrayList<Particle>>();
			int teamCount = size / teamSize;
			teamBests = new double[teamCount];
			for (int k = 0; k < teamCount; k++) {
				teams.add(new ArrayList<Particle>());
			}

			int team = 0;
			for (int i = 0; i < size; i++) {
				teams.get(team).add(swarm[i]);
				team++;
				if (team == teamCount) {
					team = 0;
				}
			}

			for (int group = 0; group < teams.size(); group++) {
				// for each group set a random team best
				double[] tBest = new double[lowerBounds.length];
				for (int t = 0; t < lowerBounds.length; t++) {
					tBest[t] = lowerBounds[t] + (rand.nextDouble() * (upperBounds[t] - lowerBounds[t]));
				}
				updateTeamBest(tBest, teams.get(group));
			}
		}
	}

	/**
	 * Execute the algorithm by updating the positions of each particle, their
	 * personal best, and the team's best
	 * 
	 * @param toDoubleFunction
	 *            Evaluation function that takes in the input coordinates and
	 *            outputs a double value
	 * @param biPredicate
	 *            Comparison function that decides whether to accept a new personal
	 *            best or team best
	 * @param maxIter
	 *            The max number iterations
	 */
	public void run(ToDoubleFunction<double[]> toDoubleFunction, BiPredicate<Double, Double> biPredicate, int maxIter) {
		// initialize the personalBest fitnesses of the particles
		for (int i = 0; i < swarm.length; i++) {
			swarm[i].setPBestFitness(toDoubleFunction.applyAsDouble(swarm[i].getPosition()));
		}

		// global PSO
		if (teams == null) {
			globalPSO(toDoubleFunction, biPredicate, maxIter);
			printSwarm();
			System.out.println("Global Best Solution = " + globalBest);
		} else {
			neighborHoodPSO(toDoubleFunction, biPredicate, maxIter);
			printSwarmTeams(toDoubleFunction);
		}
	}

	/**
	 * PSO where the entire swarm acts as a unit
	 * @param toDoubleFunction - function being optimized
	 * @param biPredicate - how to evaluate the function
	 * @param maxIter - number of iterations the algorithm is performed
	 */
	private void globalPSO(ToDoubleFunction<double[]> toDoubleFunction, BiPredicate<Double, Double> biPredicate,
			int maxIter) {
		globalBest = toDoubleFunction.applyAsDouble(swarm[0].getTeamsBest()); // init the current global best

		for (int iteration = 0; iteration < maxIter; iteration++) {
			for (int i = 0; i < size; i++) {
				swarm[i].updatePosition(lowerBounds, upperBounds);
				double eval = toDoubleFunction.applyAsDouble(swarm[i].getPosition()); // current value
				double bestEval = swarm[i].getPBestFitness(); // personal best value
				if (biPredicate.test(eval, bestEval)) { // if better then update personal best
					swarm[i].setPersonalBest(Arrays.copyOf(swarm[i].getPosition(), swarm[i].getDimen()));
					swarm[i].setPBestFitness(eval);
				}
				if (biPredicate.test(eval, globalBest)) { // if better than global then update global best
					updateGlobalBest(Arrays.copyOf(swarm[i].getPosition(), swarm[i].getDimen()));
					globalBest = eval;
				}
			}
		}
	}

	/**
	 * PSO with the swarm broken up into smaller teams
	 * @param toDoubleFunction - function being evaluated
	 * @param biPredicate - how to evaluate the function
	 * @param maxIter - number of iterations the algorithm is run
	 */
	private void neighborHoodPSO(ToDoubleFunction<double[]> toDoubleFunction, BiPredicate<Double, Double> biPredicate,
			int maxIter) {
		// initialize the team Bests
		for (int t = 0; t < teamBests.length; t++) {
			teamBests[t] = toDoubleFunction.applyAsDouble(teams.get(t).get(0).getPosition());
		}
		for (int iteration = 0; iteration < maxIter; iteration++) {

			// updated the position and variables for each team
			int index = 0;
			for (ArrayList<Particle> team : teams) {  // for every team
				for (int i = 0; i < team.size(); i++) {
					team.get(i).updatePosition(lowerBounds, upperBounds); // update its particles position
					double eval = toDoubleFunction.applyAsDouble(team.get(i).getPosition());
					double bestEval = team.get(i).getPBestFitness();

					if (biPredicate.test(eval, bestEval)) { // if better than personal best then update
						team.get(i).setPersonalBest(Arrays.copyOf(team.get(i).getPosition(), team.get(i).getDimen()));
						team.get(i).setPBestFitness(eval);
					}

					if (biPredicate.test(eval, teamBests[index])) { // if better than team best then update
						updateTeamBest(Arrays.copyOf(team.get(i).getPosition(), team.get(i).getDimen()), team);
						teamBests[index] = eval;
					}
				}
				index++;
			}
		}
	}

	// update all of the particles teams best to position gBest
	public void updateGlobalBest(double[] gBest) {
		for (int i = 0; i < size; i++) {
			swarm[i].setTeamsBest(gBest);
		}
	}

	// update a given teams best solution with tBest
	public void updateTeamBest(double[] tBest, ArrayList<Particle> team) {
		for (int i = 0; i < team.size(); i++) {
			team.get(i).setTeamsBest(tBest);
		}
	}

	public void printSwarm() {
		Arrays.stream(swarm).forEach(System.out::println);
	}

	public void printSwarmTeams(ToDoubleFunction<double[]> toDoubleFunction) {
		int teamNum = 0;
		for (ArrayList<Particle> team : teams) {
			System.out.println("Team " + teamNum);
			team.stream().forEach(System.out::println);
			double tBest = toDoubleFunction.applyAsDouble(team.get(0).getTeamsBest());
			System.out.println("The Teams Best Solution = " + tBest);
			System.out.println();
			teamNum++;
		}
	}
	
	public void writeParticles(String fileName) {
		try {
			File outFile = new File(fileName);
			FileWriter fWriter = new FileWriter(outFile);
			PrintWriter pWriter = new PrintWriter(fWriter);
			
			for (Particle p: swarm) {
				double[] cords = p.getPosition();
				String s = "";
				for (double coordinate : cords) {
					s+= coordinate + ",";
				}
				s = s.substring(0, s.length() - 1);
				pWriter.println(s);
			}
			
			pWriter.close();
			
			
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
