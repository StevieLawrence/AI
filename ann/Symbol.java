package ann;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Symbol {
	private BufferedImage image;
	private int width, height;
	private double[] pixels;
	private String filename;

	public Symbol(String filename) {
		this.filename = filename;
		try {
			image = ImageIO.read(new File("Resources\\" +filename));

		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException("Could not open file: " + filename);
		}
		width = image.getWidth();
		height = image.getHeight();
		pixels = new double[width * height];
		int i = 0;
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				pixels[i] = image.getRGB(x, y);
				i++;
			}
		}
	}

	public Symbol(int[][] shapeData, String filename) {
		this.filename = filename;
		image = new BufferedImage(shapeData.length, shapeData[0].length, BufferedImage.TYPE_INT_RGB);
		width = image.getWidth();
		height = image.getHeight();
		pixels = new double[width * height];
		int i = 0;
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				Color c = new Color(shapeData[y][x],shapeData[y][x],shapeData[y][x]);
				image.setRGB(x, y, c.getRGB());
				pixels[i] = c.getRGB();
				i++;
			}
		}
	}

	public void saveImage() {
		File imgFile = new File("Resources\\"+filename);
		try {
			ImageIO.write(image, "png", imgFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void saveImage(String filename) {
		this.filename = filename;
		File imgFile = new File("Resources\\"+filename);
		try {
			ImageIO.write(image, "png", imgFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public Symbol copy(String filename) {
		Symbol c = new Symbol(this.filename);
		c.filename = filename;
		return c;
	}
	
	public double[] getPixels() {
		return pixels;
	}
	
	public void printPixels() {
		System.out.println(filename);
		for (double pix : pixels) {
			System.out.println(pix);
		}
	}
	
	public void getArrayRep() {
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				System.out.print(image.getRGB(y, x) + " ");
			}
			System.out.println();
		}
	}
	
	public String getFilname() {
		return filename;
	}
	

	public void addNoise(double prob) {
		int i = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if ( Math.random() < prob ) {
					//int c = rand.nextInt(256);
					Color newColor = new Color(255, 255 ,255);
					image.setRGB(x, y, newColor.getRGB());
					pixels[i] = newColor.getRGB();
				}
				i++;
			}
		}
	}

	public static void main(String[] args) {
		int[][] plus = {{1,1,255,1,1},
						{1,1,255,1,1},
						{255,255,255,255,255},
						{1,1,255,1,1},
						{1,1,255,1,1}};
		
		int[][] minus = {{1,1,1,1,1},
						 {1,1,1,1,1},
						 {255,255,255,255,255},
						 {1,1,1,1,1},
						 {1,1,1,1,1}};
		
		int[][] backslash = {{255,1,1,1,1},
						   	 {1,255,1,1,1},
							 {1,1,255,1,1},
							 {1,1,1,255,1},
							 {1,1,1,1,255}};
		
		int[][] forwardslash = {{1,1,1,1,255},
								{1,1,1,255,1},
								{1,1,255,1,1},
								{1,255,1,1,1},
								{255,1,1,1,1}};
		
		int[][] X = {{255,1,1,1,255},
					 {1,255,1,255,1},
					 {1,1,255,1,1},
					 {1,255,1,255,1},
					 {255,1,1,1,255}};
		
		int[][] pike = {{1,1,255,1,1},
						{1,1,255,1,1},
						{1,1,255,1,1},
						{1,1,255,1,1},
						{1,1,255,1,1}};
		
		
		Symbol pl = new Symbol(plus, "plus.png");
		pl.saveImage();
		Symbol min = new Symbol(minus, "minus.png");
		min.saveImage();
		Symbol bs = new Symbol(backslash, "backslash.png");
		bs.saveImage();
		Symbol fw = new Symbol(forwardslash, "forwardslash.png");
		fw.saveImage();
		Symbol x = new Symbol(X,"X.png");
		x.saveImage();
		Symbol pi = new Symbol(pike, "pike.png");
		pi.saveImage();
		
	}

}
