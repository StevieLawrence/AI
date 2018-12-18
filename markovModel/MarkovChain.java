package markovModel;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MarkovChain {
	private HashMap<String, Integer> transitions;
	private String[] uniqueChars;
	private double[][] probTable;
	HashSet<String> uniqueCharSet;

	public MarkovChain(File textFile) {
		String text = getText(textFile);
		setMarkovChain(text);

	}
	
	public MarkovChain(ArrayList<String> words) {
		String text = words.stream().collect(Collectors.joining(" ", " ", " "));
		setMarkovChain(text);
	}
	
	private void setMarkovChain(String text) {
		transitions = new HashMap<String, Integer>();
		String gram = "";
		String left = "";
		String right = "";
		uniqueCharSet = new HashSet<String>();
		for (int from = 0, to = 1; to < text.length(); from++, to++) {
			left = text.substring(from, to);
			right = text.substring(to, to + 1);
			uniqueCharSet.add(left);
			gram = left + right;
			if (!transitions.containsKey(gram)) {
				transitions.put(gram, 1);
			} else {
				transitions.put(gram, transitions.get(gram) + 1);
			}
		}

		uniqueChars = uniqueCharSet.stream().toArray(String[]::new);

		probTable = new double[uniqueChars.length][uniqueChars.length];
		for (int i = 0; i < uniqueChars.length; i++) {
			int count = 0;
			for (int j = 0; j < uniqueChars.length; j++) {
				String tran = uniqueChars[i] + uniqueChars[j];
				if (transitions.containsKey(tran)) {
					probTable[i][j] = transitions.get(tran);
					count += transitions.get(tran);
				} else {
					probTable[i][j] = 0;
				}
			}
			for (int k = 0; k < uniqueChars.length; k++) {
				probTable[i][k] = probTable[i][k] / ((double) count);
			}
		}
	}

	private String getText(File textFile) {
		Scanner sc;
		try {
			sc = new Scanner(textFile);
			String text = "";
			while (sc.hasNextLine()) {
				text = text + sc.nextLine();
			}
			sc.close();
			return Stream.of(text.split("\\s+")).map(str -> str.toLowerCase())
					.filter(str -> str.chars().allMatch(Character::isLetter))
					.collect(Collectors.joining(" ", " ", " "));

		} catch (FileNotFoundException e) {
			System.out.println("Counld not find file: using John Milton instead");
			String dummyText = " for who would lose, though full of pain, this intellectual being, "
					+ "those thoughts that wander through eternity, " + "to perish, rather, swallowed up and lost "
					+ "in the wide womb of uncreated night " + "devoid of sense and motion ";
			return dummyText;
		}
	}

	public String createWord() {
		String word = " ";
		int from = 0;
		for (int i = 0; i < uniqueChars.length; i++) {
			if ( word.equals(uniqueChars[i])) {
				from = i;
				break;
			}
		}
		ArrayList<Integer> selection = new ArrayList<Integer>();
		Random rand = new Random();
		while (word.length() <= 1 || !word.endsWith(" ")) {
			selection.clear();
			// System.out.println("while: " + word);
			for (int i = 0; i < uniqueChars.length; i++) {
				if (probTable[from][i] == 0.0)
					continue;
				else {
					selection.add(i);
				}
			}
			Collections.shuffle(selection);
			double runningSum = 0;
			double threshold = rand.nextDouble();
			for (int to : selection) {
				if (probTable[from][to] + runningSum > threshold) {
					word += uniqueChars[to];
					from = to;
					break;
				}
				runningSum += probTable[from][to];
			}

		}

		return word;
	}

	public double getProbability(String word) {
		char[] tokens = word.toCharArray();
		int[] indexes = new int[tokens.length];

		for (int i = 0; i < tokens.length; i++) {
			for (int index = 0; index < uniqueChars.length; index++) {
				String character = String.valueOf(tokens[i]);
				
				if (!uniqueCharSet.contains(character)) {
					return 0;
				}
				
				if (character.equals(uniqueChars[index]))  {
					indexes[i] = index;
				}
			}
		}
		
		double probability = 1.0;
		for (int from = 0, to = 1; to < indexes.length; from++, to++) {
			//System.out.println(probTable[indexes[from]][indexes[to]]);
			probability *= probTable[indexes[from]][indexes[to]];
		}
		return probability;

	}

	public void printProbTable() {
		System.out.print(" ");
		System.out.println(Arrays.toString(uniqueChars));
		for (int i = 0; i < uniqueChars.length; i++) {
			System.out.print(uniqueChars[i]);
			System.out.println(Arrays.toString(probTable[i]));
		}
	}
	
	public double[][] getProbTable(){
		return probTable;
	}

}
