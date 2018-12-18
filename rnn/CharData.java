package rnn;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import org.jblas.DoubleMatrix;

// Text data for text generation with RNNs, one-hot encoding for the characters
public class CharData {
	private Map<String, Integer> charIndex = new HashMap<>();
	private Map<Integer, String> indexChar = new HashMap<>();
	private Map<String, DoubleMatrix> charVector = new HashMap<>();
	private List<String> sequences = new ArrayList<>();

	// Read in text from a file and map the characters to one-hot encoding 
	public CharData(String filename) {
		readInFile(filename);
		for (String c : charIndex.keySet()) {
			DoubleMatrix xt = DoubleMatrix.zeros(1, charIndex.size());
			xt.put(charIndex.get(c), 1);
			charVector.put(c, xt);
		}
	}
	
	// Read in text from an arraylist and map the characters to one-hot encoding
	public CharData(ArrayList<String> data) {
		readInArrayList(data);
		for (String c : charIndex.keySet()) {
			DoubleMatrix xt = DoubleMatrix.zeros(1, charIndex.size());
			xt.put(charIndex.get(c), 1);
			charVector.put(c, xt);
		}
	}

	// read in the text from a file
	public void readInFile(String filename) {
		try {
			Scanner reader = new Scanner(new File(filename));
			String line = "";
			while (reader.hasNextLine()) {
				line = reader.nextLine().toLowerCase();
				if (line.length() > 3) {
					sequences.add(line);
					for (char c : line.toLowerCase().toCharArray()) {
						String key = String.valueOf(c);
						if (!charIndex.containsKey(key)) {
							charIndex.put(key, charIndex.size());
							indexChar.put(charIndex.get(key), key);
						}
					}
				}
			}
			reader.close();
		} catch (Exception e) {

		}
	}
	
	// read in the text from an arraylist
	public void readInArrayList(ArrayList<String> data) {
		sequences = data;
		int size = sequences.size();
		for (int i = 0; i < size; i ++) {
			for (char c : sequences.get(i).toCharArray()) {
				String key = String.valueOf(c);
				if (!charIndex.containsKey(key)) {
					charIndex.put(key, charIndex.size());
					indexChar.put(charIndex.get(key), key);
				}
			}
		}
	}
	
	  public Map<String, Integer> getCharIndex() {
	        return charIndex;
	    }

	    public Map<String, DoubleMatrix> getCharVector() {
	        return charVector;
	    }

	    public List<String> getSequences() {
	        return sequences;
	    }
	    
	    public Map<Integer, String> getIndexChar() {
	        return indexChar;
	    }


	public static void main(String[] args) {
		

	}

}
