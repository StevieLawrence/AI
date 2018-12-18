package classifiersAndClusterers;

import java.util.HashSet;

public class Cluster {
	private int label;
	Point centroid;
	private HashSet<Point> points;

	/**
	 * grouping of points around a centroid
	 * 
	 * @param label
	 * @param centroid
	 */
	public Cluster(int label, Point centroid) {
		this.label = label;
		this.centroid = centroid;
		points = new HashSet<Point>();
		this.centroid.assignPoint("Centroid");
	}

	public void addPoint(Point p) {
		points.add(p);
	}

	public void removePoint(Point p) {
		points.remove(p);
	}

	/**
	 * @return Average of all of the points in the cluster
	 */
	public double calcCentroid() {
		double[] cordSums = new double[centroid.getCords().length];

		for (Point p : points) {
			for (int index = 0; index < p.getCords().length; index++) {
				cordSums[index] += p.getCords()[index];
			}
		}
		double[] newCords = new double[cordSums.length];
		for (int i = 0; i < newCords.length; i++) {
			newCords[i] = cordSums[i] / (double) points.size();
		}
		
		Point newCentroid = new Point(newCords);
		double deltaDist = newCentroid.euclideanDist(centroid);
		newCentroid.assignPoint("Centroid");
		centroid = newCentroid;
		return deltaDist;
	}

	public int getLabel() {
		return label;
	}

	public HashSet<Point> getPoints() {
		return points;
	}

	public void printCluster() {
		System.out.println("Cluster = " + label);
		System.out.println("Centroid = " + centroid);
		points.stream().forEach(System.out::println);
	}

}
