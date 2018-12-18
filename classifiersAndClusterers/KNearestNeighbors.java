package classifiersAndClusterers;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;

public class KNearestNeighbors {
	private int k;
	private HashSet<ArrayList<Point>> knownPoints;
	private ArrayList<Point> allPoints;
	private String[] groups;
	private ArrayList<Point> trainingSet;

	/**
	 * Unsupervised learning that classifies data based on the identity of its
	 * neighbors
	 * 
	 * @param k
	 *            - the number of nearest neighbors
	 * @param knownPoints
	 *            - data points belonging to known classes
	 * @param unknownPoints
	 *            - unknown data
	 */
	public KNearestNeighbors(int k, HashSet<ArrayList<Point>> knownPoints, ArrayList<Point> unknownPoints) {
		this.k = k;
		this.knownPoints = knownPoints;
		trainingSet = unknownPoints;
		groups = new String[knownPoints.size()];
		allPoints = new ArrayList<Point>();
		int i = 0;
		for (ArrayList<Point> group : knownPoints) {
			groups[i] = group.get(0).getAssignment();
			allPoints.addAll(group);
			i++;
		}
		;
		classifyAllPoints(trainingSet);
	}

	private void classifyAllPoints(ArrayList<Point> unknowns) {
		unknowns.stream().forEach(this::classify);
	}

	/**
	 * find the distance between every known point and the unknown point and sort
	 * them in decreasing order
	 * 
	 * @param unknown
	 *            - the unknown point
	 */
	private void classify(Point unknown) {
		for (Point known : allPoints) { // get distance of every known point to the unknown point
			known.goTo(unknown);
		}
		// sort the known points by those distances in decreasing order
		Collections.sort(allPoints, new Comparator<Point>() {
			@Override
			public int compare(Point p1, Point p2) {
				return p1.distanceTo.compareTo(p2.distanceTo);
			}
		});

		// create a sublist of size k to get k nearest neighbors
		List<Point> minDistances = allPoints.subList(0, k);
		String group = majorityRule(minDistances); // figure out what class it belongs too
		unknown.assignPoint(group);
	}

	/**
	 * To decide which group the unknown point belongs to use majority rule of k
	 * nearest neighbors.
	 * 
	 * @param minDistances
	 *            - list containing k nearest neighbors
	 * @return - the classification
	 */
	private String majorityRule(List<Point> minDistances) {
		int[] freqs = new int[knownPoints.size()];
		for (Point p : minDistances) {
			String group = p.getAssignment();
			for (int i = 0; i < groups.length; i++) {
				if (group.equals(groups[i]))
					freqs[i]++;
			}
		}
		int maxIndex = 0;
		int maxCount = freqs[0];
		for (int j = 1; j < freqs.length; j++) {
			if (maxCount < freqs[j]) {
				maxIndex = j;
				maxCount = freqs[j];
			}
		}

		return groups[maxIndex];
	}

	/**
	 * After classification integrate all of the training data into their respective
	 * classes
	 * 
	 * @param trainingSet
	 */
	public void integrateTrainingData() {
		for (Point trainy : trainingSet) {
			for (ArrayList<Point> knowns : knownPoints) {
				if (knowns.get(0).getAssignment().equals(trainy.getAssignment()))
					knowns.add(trainy);
			}
		}
	}

	public void printTrainingSet() {
		int N;
		for (String group : groups) {
			N = 0;
			System.out.println("Class " + group);
			for (Point p : trainingSet) {
				if (p.getAssignment().equals(group)) {
					System.out.println(p);
					N++;
				}
			}
			System.out.println("Amount = " + N + " out of " + trainingSet.size());
			System.out.println();
		}
	}

	public void printClasses() {
		for (ArrayList<Point> group : knownPoints) {
			System.out.println("\nClass: " + group.get(0).getAssignment());
			group.stream().forEach(System.out::println);
		}
	}

	public void writeTraingingData(String filename) {
		try {
			File outFile = new File(filename);
			FileWriter fWriter = new FileWriter(outFile);
			PrintWriter pWriter = new PrintWriter(fWriter);

			for (Point p : trainingSet) {
				String s = p.listCords();
				s += "," + p.getAssignment();
				pWriter.println(s);
			}

			pWriter.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void writeTrainingDataSeparate() {

		try {
			for (int i = 0; i < groups.length; i++) {
				File outFile = new File("C:\\Users\\dmx\\Documents\\" + groups[i] + "Assignment.txt");
				FileWriter fWriter = new FileWriter(outFile);
				PrintWriter pWriter = new PrintWriter(fWriter);

				for (Point p : trainingSet) {
					if (p.getAssignment().equals(groups[i])) {
						pWriter.println(p.listCords());
					}
				}
				pWriter.close();
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}

	public void setTrainingSet(ArrayList<Point> trainingSet) {
		this.trainingSet = trainingSet;
	}

	public void addClassification(ArrayList<Point> group) {
		knownPoints.add(group);
		allPoints.addAll(group);
		groups = new String[knownPoints.size()];
		int i = 0;
		for (ArrayList<Point> g : knownPoints) {
			groups[i] = g.get(0).getAssignment();
			i++;
		}
		;
	}

	public String[] getGroups() {
		return groups;
	}

	public void printGroups() {
		Arrays.stream(groups).forEach(System.out::println);
	}

}
