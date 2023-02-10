import java.util.Map;
import java.util.TreeMap;

public class JavaTruthTableMaker {
    public static void main(String[] args) {
        String expr = "(xor (xor (and (and  LN7 LN34 )(and  LN7 LN34 ) ) LN214 ) LN213 )";
        String expr1 = "(xor (and  LN40(xor (xor  k7 LN17 ) LN34 ) ) LN56 )";
//        String expr = "(xor  LN29(xor  LN3(and  LN19 LN20 ) ) )";
        truthTableExpr(expr1);
    }

    /**
     * Construct a truth table for a given expression (according to SyGuS format)
     * Accepts xor, and, or, and not (although I haven't seen not in any benchmarks)
     * The bit patterns produced are based on the variable numbers, so the first bit
     * is for the smallest variable, and they increase. Should enumerate all
     * possible bit patterns.
     * @param expr
     */
    static void truthTableExpr(String expr) {
        TreeMap<Integer, Boolean> vars = new TreeMap<>();
        for(int i=0; i<expr.length()-2; i++) {
            if(expr.substring(i, i+2).equals("LN")) {
                int val = -1, size = 1;
                while (true) {
                    try {
                        val=Integer.parseInt(expr.substring(i+2, i+2+size));
                    } catch(Exception e) {
                        break;
                    }
                    size++;
                }
                vars.put(val, false);
            }
        }

        expr=expr.replaceAll("\\(", " (");
        expr=expr.replaceAll("\\)", " )");

        while(!expr.equals(expr.replaceAll("  ", " "))) {
            expr=expr.replaceAll("  ", " ");
        }

        for(int i=0; i<Math.pow(2, vars.size()); i++) {
            int j=0;
            for(Map.Entry<Integer, Boolean > e : vars.entrySet()) {
                vars.put(e.getKey(), (i>>j&1)==1);
//                System.out.print(e.getKey()+": "+(i>>j&1)+" ");
                System.out.print((i>>j&1));
                j++;
            }
            System.out.println(" "+evalExpr(expr, vars));
        }
    }
    static boolean evalExpr(String expr, TreeMap<Integer, Boolean> vars) {
//        System.out.println(expr);
        int numberExprs=0;
        String operator="";
        boolean[] input=new boolean[2];
        for(int i=0; i<expr.length(); i++) {
            String currExpr=expr.substring(i);
            if(currExpr.startsWith("LN")) {
                int val = -1, size = 1;
                while (true) {
                    try {
                        val=Integer.parseInt(expr.substring(i+2, i+2+size));
                    } catch(Exception e) {
                        break;
                    }
                    size++;
                }
//                System.out.println(val);
                input[numberExprs-1]=vars.get(val);
                numberExprs++;
                i+=1+size;
            }
            else if(currExpr.startsWith("xor")) {
                operator = "xor";
                i+=2;
                numberExprs++;
            }
            else if(currExpr.startsWith("and")) {
                operator = "and";
                i+=2;
                numberExprs++;
            }
            else if(currExpr.startsWith("or")) {
                operator = "or";
                i+=1;
                numberExprs++;
            }
            else if(currExpr.startsWith("not")) {
                operator = "not";
                i+=2;
                numberExprs++;
            }
            else if(expr.charAt(i) == '(') {
                boolean parenExpr = evalExpr(expr.substring(i+1, findMatchingParen(expr, i)-1), vars);
                if(numberExprs==0) return parenExpr;
                input[numberExprs - 1] = parenExpr;
                numberExprs++;
                i = findMatchingParen(expr, i) + 1;
            }
            if(operator.equals("not") && numberExprs == 2) {
                return !input[0];
            }
            if(numberExprs == 3) {
                if(operator.equals("xor")) return input[0] ^ input[1];
                if(operator.equals("and")) return input[0] && input[1];
                if(operator.equals("or")) return input[0] || input[1];
            }
        }
        return false;
    }
    static int findMatchingParen(String str, int start) {
        int count=0;
        for(int i=start+1; i<str.length(); i++) {
            if(str.charAt(i)=='(') count++;
            if(str.charAt(i)==')') count--;
            if(count==-1) return i;
        }
        return -1;
    }
}