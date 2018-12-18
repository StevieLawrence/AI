package markovModel;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MarkovMain {

	public static void main(String[] args) {
		// MarkovChain mc = new MarkovChain(new File("TextFiles\\ABraveNewWorld.txt"));

		String[] words = { "spare", "spear", "pares", "peers", "reaps", "peaks", "speaker", "keeper", "pester", "paste",
				"tapas", "pasta", "past", "straps", "tears", "terse", "steer", "street", "stare", "rates", "streak",
				"taste", "tapa", "peat", "eat", "ate", "tea", "seat" };
		ArrayList<String> wordsList = new ArrayList<String>(Arrays.asList(words));
		MarkovChain mc = new MarkovChain(wordsList);
		
		ArrayList<String> genWords = new ArrayList<String>();
		for (int j = 0; j < 10; j++) {
			for (int i = 0; i < 10; i++) {
				String word = mc.createWord();
				//System.out.print(word);
				genWords.add(word.trim());
			}
			//System.out.println();
		}
		
		//original words
		System.out.println("\nWords from original");
	genWords.stream().filter(str -> wordsList.contains(str)).forEach(System.out::println);
	mc.printProbTable();


		// repeated words
		System.out.println("\nRepeated words");
		Map<String, Long> freqs = genWords.stream().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
		for (String key : freqs.keySet()) {
			Long val = freqs.get(key);
			if (val > 1)
				System.out.println(key + ": " + val);
		}
		
		// probability of each word
		System.out.println("\nProbabilities");
		for(String wrd : freqs.keySet()) {
			double prob = mc.getProbability(" " + wrd + " ");
			System.out.println(wrd + "| " + prob);
		}
		
		// longest and shortest words
		ArrayList<String> shortestWords = new ArrayList<String>();
		ArrayList<String> longestWords = new ArrayList<String>();
		int minLength = wordsList.get(0).length();
		int maxLength = minLength;
		
		for (String strg : freqs.keySet()) {
			if (strg.length() < minLength) {
				shortestWords.clear();
				shortestWords.add(strg);
				minLength = strg.length();
			}
			else if (strg.length() == minLength) {
				shortestWords.add(strg);
			}
			else if (strg.length() > maxLength) {
				longestWords.clear();
				longestWords.add(strg);
				maxLength = strg.length();
			}
			else if (strg.length() == maxLength) {
				longestWords.add(strg);
			}
		}
		
		
		
		System.out.println("\nShortest and Longest Words");
		System.out.println("Shortest: " + shortestWords.toString());
		System.out.println("Longest: " + longestWords.toString());
		System.out.println(mc.getProbability(" peapeseapeatersteakeparsteapeapateatreererstereeereapearestreasteaspetatrearsteapa "));
		
		MarkovChain mc2 = new MarkovChain(genWords);
		for (int i = 0; i < 10; i++) {
			for (int k = 0; k < 10; k++){
				System.out.print(mc2.createWord());
			}
			System.out.println();
		}
	
		// Other Langs
		System.out.println("other writing text");
		MarkovChain mc3 = new MarkovChain(new File("TextFiles\\slinky.txt"));
		MarkovChain mc4 = new MarkovChain(new File("TextFiles\\frenchwriting.txt"));
		
		for (int c = 0; c < 10; c++) {
			for (int d = 0; d < 10; d++) {
				System.out.print(mc3.createWord());
			}System.out.println();
		}
		System.out.println("french");
		for (int c = 0; c < 10; c++) {
			for (int d = 0; d < 10; d++) {
				System.out.print(mc4.createWord());
			}System.out.println();
		}
	
	}

}
