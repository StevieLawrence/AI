package classifiersAndClusterers;

import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;

public class Point {
	private double[] cords;
	private String classification;
	Double distanceTo;
	private Point goToPoint;

	/**
	 * Point of any dimension
	 * @param cords the cordinates
	 */
	public Point(double ... cords) {
		this.cords = cords;
		classification = "unknown";
	}

	/**
	 * sqrt(sum(qi - pi)^2)
	 * 
	 * @param p
	 *            the point
	 * @return the distance
	 */
	public double euclideanDist(Point p) {
		double accumulator = 0.0;
		for (int index = 0; index < cords.length; index++) {
			accumulator += Math.pow(cords[index] - p.cords[index], 2);
		}
		
		return Math.sqrt(accumulator);
	}
	
	public Point midPoint(Point p) {
		double[] newCords = new double[cords.length];
		for (int index = 0; index < cords.length; index++) {
			newCords[index] = (cords[index] + p.cords[index]) / 2.0;
		}
		return new Point(newCords);
	}

	private void setDistanceTo(Point p) {
		distanceTo = this.euclideanDist(p);
	}

	public void goTo(Point p) {
		goToPoint = p;
		setDistanceTo(p);
	}

	public void assignPoint(String classification) {
		this.classification = classification;
	}

	public String getAssignment() {
		return classification;
	}

	public Point clone() {
		Point p = new Point(cords);
		return p;
	}
	
	/**
	 * Reads in the data points
	 * 
	 * @param filename
	 *            - the name of the file
	 * @return - array list of all of the data points
	 */
	public static ArrayList<Point> readInPoints(String filename, String delimiter) {
		ArrayList<Point> allPoints = new ArrayList<Point>();
		try {
			Scanner sc = new Scanner(new File(filename));
			sc.nextLine();
			while (sc.hasNextLine()) {
				String line = sc.nextLine().trim();
				String[] s = line.split(delimiter);
				double[] theCords = new double[s.length];
				for (int i = 0; i < s.length; i++) {
					theCords[i] = Double.parseDouble(s[i].trim());
				}
				allPoints.add(new Point(theCords));
			}
			sc.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return allPoints;
	}

	public double[] getCords() {
		return cords;
	}
	
	public void setCords(double[] cords) {
		this.cords = cords;
	}

	public boolean notAssigned() {
		if (classification.equals("unknown"))
			return true;
		else
			return false;
	}

	public void printGoTo() {
		if (goToPoint != null)
			System.out.println(this + " goes " + distanceTo + " to " + goToPoint);
		else
			System.out.println(this + " goes nowhere");

	}
	
	public String listCords() {
		String s = "";
		for (int i = 0; i < cords.length - 1; i++) {
			s += cords[i] + ",";
		}
		s += cords[cords.length - 1];
		return s;
	}

	@Override
	public String toString() {
		return  listCords();
	}

}
