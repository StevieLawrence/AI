package classifiersAndClusterers;

import java.util.ArrayList;
import java.util.HashSet;

public class KTestMain {

	public static void main(String[] args) {

		for (int i = 3; i <= 5; i++) {
			KMeansClustering kMeans = new KMeansClustering(i, "TextFiles\\xyz.txt", 1000);
			kMeans.run();
			kMeans.printClusters();
			kMeans.writeClustersToFile("TextFiles\\kMeans" + i + ".csv");
		}
		/*
		ArrayList<Point> tube = Point.readInPoints("TextFiles\\tube.txt",",");
		tube.stream().forEach(p -> p.assignPoint("tube"));
		ArrayList<Point> torus = Point.readInPoints("TextFiles\\torus.txt",",");
		torus.stream().forEach(p -> p.assignPoint("torus"));
		ArrayList<Point> unk = Point.readInPoints("TextFiles\\unknownPoints.txt",",");
		HashSet<ArrayList<Point>> categs = new HashSet<ArrayList<Point>>();
		categs.add(tube); categs.add(torus);
		KNearestNeighbors knn = new KNearestNeighbors(65, categs, unk);
		knn.printTrainingSet();
		knn.writeTraingingData("TextFiles\\classifications.txt");*/
		/*
		ArrayList<Point> meshgrid = new ArrayList<Point>();
		double step = 0.01;
		double x_min,y_min,z_min,x_max,y_max,z_max;
		x_min = y_min = -0.5;
		x_max = y_max = 0.55;
		z_min = 0; z_max = 6.8;
		for (double x_step = x_min; x_step < x_max; x_step += step) {
			for (double y_step = y_min; y_step < y_max; y_step += step) {
				for (double z_step = z_min; z_step < z_max; z_step += step) {
					meshgrid.add(new Point(x_step, y_step, z_step));
				}
			}
		}
		KNearestNeighbors knn2 = new KNearestNeighbors(65, categs, meshgrid);
		knn2.writeTrainingDataSeparate();*/
	}

}        
