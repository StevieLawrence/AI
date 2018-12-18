package classifiersAndClusterers;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class KMeansClustering {
	Cluster[] clusters;
	ArrayList<Point> allPoints;
	int maxIterations;

	/**
	 * K-Means Clustering is an un-supervised learning algorithm that applies
	 * clustering of data
	 * 
	 * @param k
	 *            - the number of clusters
	 * @param allPoints
	 *            - all of the points in the data
	 * @param maxIterations
	 *            - the max number of iterations that stops the algorithm if the
	 *            clusters still haven't settled
	 */
	public KMeansClustering(int k, ArrayList<Point> allPoints, int maxIterations) {
		this.allPoints = allPoints;
		this.maxIterations = maxIterations;
		initClusters(k);
	}

	/**
	 * K-Means clustering constructor where the data points are read in.
	 * 
	 * @param k
	 *            - the number of clusters
	 * @param filename
	 *            - name of the files with all of the data points
	 * @param maxIterations
	 *            - max number of iterations
	 */
	public KMeansClustering(int k, String filename, int maxIterations) {
		allPoints = Point.readInPoints(filename,"\t");
		this.maxIterations = maxIterations;
		initClusters(k);
	}

	/**
	 * Initialize the clusters in the algorithm
	 * 
	 * @param k
	 *            - the number of clusters
	 */
	private void initClusters(int k) {
		clusters = new Cluster[k];
		for (int i = 0; i < k; i++) {
			Point p = getRandomPoint(allPoints).clone(); // initialize the centroid to a random point
			clusters[i] = new Cluster(i, p);
			System.out.println(clusters[i].centroid);
		}
	}

	/**
	 * Select a random data point from all of the data points
	 * 
	 * @param points
	 *            - array list of all
	 * @return a random point
	 */
	private Point getRandomPoint(ArrayList<Point> points) {
		Random rand = new Random();
		int i = rand.nextInt(points.size());
		int j = rand.nextInt(points.size());
		while (j == i)
			j = rand.nextInt(points.size());
		Point p1 = points.get(i);
		Point p2 = points.get(j);
		return p1.midPoint(p2);
	}

	/**
	 * Loop through all of the points and calculate their distances to the
	 * centroids. Assign the point to the nearest cluster. Recalculate the centroid.
	 * If a point changes assignment or a centroid moves then go through the process
	 * again.
	 */
	public void run() {
		int iters = 0;
		boolean changedAssignment = true;
		boolean centroidsMoved = true;
		while (centroidsMoved || changedAssignment || iters > maxIterations) {
			changedAssignment = false;
			// loop through all of the points and calculate their distance to the centroids
			for (int i = 0; i < allPoints.size(); i++) {
				double minDistance = allPoints.get(i).euclideanDist(clusters[0].centroid);
				int minIndex = 0;
				for (int k = 1; k < clusters.length; k++) {
					double distance = allPoints.get(i).euclideanDist(clusters[k].centroid);
					// keep track of the minimum distance
					if (distance < minDistance) {
						minDistance = distance;
						minIndex = k;
					}
				}

				String assignment = allPoints.get(i).getAssignment();
				// check to see if the point has not been assigned to a cluster yet
				if (allPoints.get(i).notAssigned()) {
					allPoints.get(i).assignPoint(clusters[minIndex].getLabel() + "");
					clusters[minIndex].addPoint(allPoints.get(i));
					changedAssignment = true;
					// Check to see if the point changed assignment
				} else if (Integer.parseInt(assignment) != minIndex) {
					clusters[Integer.parseInt(assignment)].removePoint(allPoints.get(i));
					allPoints.get(i).assignPoint(clusters[minIndex].getLabel() + "");
					clusters[minIndex].addPoint(allPoints.get(i));
					changedAssignment = true;
				}
			}
			centroidsMoved = updateCentroids();
			iters++;
		}
		System.out.println("finished with " + iters + " iterations");
	}

	/**
	 * Re-calculate the centroids by averaging all of the points in the cluster
	 * 
	 * @return boolean of if just one of the centroids moved
	 */
	private boolean updateCentroids() {
		boolean centroidsMoved = false;
		for (int k = 0; k < clusters.length; k++) {
			double distance = clusters[k].calcCentroid();
			System.out.println("Cluster " + clusters[k].getLabel() + " moved " + distance);
			if (distance != 0.0)
				centroidsMoved = true;
		}
		return centroidsMoved;
	}

	/**
	 * Write the clusters to a file
	 * 
	 * @param fileName
	 *            - name of the file being written to
	 */
	public void writeClustersToFile(String fileName) {
		try {
			File outFile = new File(fileName);
			FileWriter fWriter = new FileWriter(outFile);
			PrintWriter pWriter = new PrintWriter(fWriter);

			for (int k = 0; k < clusters.length; k++) {
				HashSet<Point> points = clusters[k].getPoints();
				pWriter.println("Cluster" + clusters[k].getLabel());
				for (Point p : points) {
					pWriter.println(p.listCords());
				}
				pWriter.println(clusters[k].centroid.listCords());
				pWriter.println();
			}
			pWriter.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Print the clusters
	 */
	public void printClusters() {
		for (int k = 0; k < clusters.length; k++) {
			clusters[k].printCluster();
			System.out.println();
		}
	}

}
