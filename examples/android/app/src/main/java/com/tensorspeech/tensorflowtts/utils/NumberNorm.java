package com.tensorspeech.tensorflowtts.utils;

import java.util.HashMap;
import java.util.Map;

// Borrowed from https://rosettacode.org/wiki/Spelling_of_ordinal_numbers
public class NumberNorm {

    private static Map<String,String> ordinalMap = new HashMap<>();
    static {
        ordinalMap.put("one", "first");
        ordinalMap.put("two", "second");
        ordinalMap.put("three", "third");
        ordinalMap.put("five", "fifth");
        ordinalMap.put("eight", "eighth");
        ordinalMap.put("nine", "ninth");
        ordinalMap.put("twelve", "twelfth");
    }

    public static String toOrdinal(long n) {
        String spelling = numToString(n);
        String[] split = spelling.split(" ");
        String last = split[split.length - 1];
        String replace;
        if ( last.contains("-") ) {
            String[] lastSplit = last.split("-");
            String lastWithDash = lastSplit[1];
            String lastReplace;
            if ( ordinalMap.containsKey(lastWithDash) ) {
                lastReplace = ordinalMap.get(lastWithDash);
            }
            else if ( lastWithDash.endsWith("y") ) {
                lastReplace = lastWithDash.substring(0, lastWithDash.length() - 1) + "ieth";
            }
            else {
                lastReplace = lastWithDash + "th";
            }
            replace = lastSplit[0] + "-" + lastReplace;
        }
        else {
            if ( ordinalMap.containsKey(last) ) {
                replace = ordinalMap.get(last);
            }
            else if ( last.endsWith("y") ) {
                replace = last.substring(0, last.length() - 1) + "ieth";
            }
            else {
                replace = last + "th";
            }
        }
        split[split.length - 1] = replace;
        return String.join(" ", split);
    }

    private static final String[] nums = new String[] {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
    };

    private static final String[] tens = new String[] {"zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

    public static final String numToString(long n) {
        return numToStringHelper(n);
    }

    private static final String numToStringHelper(long n) {
        if ( n < 0 ) {
            return "negative " + numToStringHelper(-n);
        }
        int index = (int) n;
        if ( n <= 19 ) {
            return nums[index];
        }
        if ( n <= 99 ) {
            return tens[index/10] + (n % 10 > 0 ? "-" + numToStringHelper(n % 10) : "");
        }
        String label = null;
        long factor = 0;
        if ( n <= 999 ) {
            label = "hundred";
            factor = 100;
        }
        else if ( n <= 999999) {
            label = "thousand";
            factor = 1000;
        }
        else if ( n <= 999999999) {
            label = "million";
            factor = 1000000;
        }
        else if ( n <= 999999999999L) {
            label = "billion";
            factor = 1000000000;
        }
        else if ( n <= 999999999999999L) {
            label = "trillion";
            factor = 1000000000000L;
        }
        else if ( n <= 999999999999999999L) {
            label = "quadrillion";
            factor = 1000000000000000L;
        }
        else {
            label = "quintillion";
            factor = 1000000000000000000L;
        }
        return numToStringHelper(n / factor) + " " + label + (n % factor > 0 ? " " + numToStringHelper(n % factor ) : "");
    }
}
