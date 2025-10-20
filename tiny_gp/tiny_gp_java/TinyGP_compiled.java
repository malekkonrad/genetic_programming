import java.nio.charset.StandardCharsets;
import java.text.DecimalFormat;
import java.util.*;
import java.io.*;

public class TinyGP {
    // operations
    public static final int
            ADD = 110,
            SUB = 111,
            MUL = 112,
            DIV = 113,
            EXP = 114,
            SIN = 115,
            COS = 116,
            FSET_START = 110,
            FSET_2ARG_END = 113,  //last 2 argument function, start of 1 argument functions
            FSET_END = 113;  // last operation available

    // settings
    public static final double
            DIVISION_CUT_OUT = 0.001,
            EXPONENT_CUT_OUT = 100.0;

    // parameter variables
    public static final int
            MAX_LEN = 10000,
            POPSIZE = 100000,
            DEPTH   = 5,
            GENERATIONS = 10,
            TSIZE = 2;
    public static final double
            minrandom = 5.0,
            maxrandom = -5.0,
            goal_fitness = 0.01;
    public static final int
            varnumber = 2,
            fitnesscases = 10000,
            randomnumber = 100;
    public static final double
            PMUT_PER_NODE  = 0.05,
            CROSSOVER_PROB = 0.9;

    // population variables
    static double [] fitness;
    static char [][] population;
    static double fbestpop = 0.0, favgpop = 0.0;

    // helper variables
    static long seed = -1;
    static Random rd = new Random();
    static double [] x = new double[FSET_START];
    static double avg_len;
    // public static final double [][] targets = None;
    
    public static double[][] targets;

    // public static String file_name = None;

    public static String file = "targets.dat";

    // cache variables
    static double[] numbers;
    static char[] operations;
    static int PC;
    static double[] variables;
    static int length;



    static void loadTargets() {
        try (BufferedReader in = new BufferedReader(new FileReader(file))) {
            // String line = in.readLine();
            targets = new double[fitnesscases][varnumber + 1];
            String line;
            for (int i = 0; i < fitnesscases; i++) {
                line = in.readLine();
                if (line == null) {
                    throw new IOException("Unexpected end of file at line " + (i + 2));
                }
                
                StringTokenizer tokens = new StringTokenizer(line);
                
                for (int j = 0; j <= varnumber; j++) {
                    if (!tokens.hasMoreTokens()) {
                        throw new IOException("Not enough values in line " + (i + 2));
                    }
                    targets[i][j] = Double.parseDouble(tokens.nextToken().trim());
                }
            }
        } 
        catch (FileNotFoundException e) {
            System.err.println("ERROR: Data file not found: " + file);
            System.exit(1);
        } 
        catch (NumberFormatException e) {
            System.err.println("ERROR: Invalid number format in data file");
            System.exit(1);
        } 
        catch (IOException e) {
            System.err.println("ERROR: " + e.getMessage());
            System.exit(1);
        } 
        catch (Exception e) {
            System.err.println("ERROR: Unexpected error while reading data file: " + e.getMessage());
            System.exit(1);
        }
    }

    void simplify(char[] prog) {
        // simplify the individual
        // evaluate operations between constants
        // put operations and evaluation result into separate arrays (operations and numbers)
        // 0 in operations means it's a number
        // ADD, SUB, MUL, DIV - operations
        // 1 + var_id - it's a variable
        int ptr = 0;
        operations = new char[prog.length];
        Arrays.fill(operations, '\0');
        numbers = new double[prog.length];
        for (char primitive : prog) {
            if (primitive < varnumber) {  // it's the variable
                // variables start from 1 to differentiate them operations
                operations[ptr++] = ++primitive; // add that 1
            } else if (primitive < FSET_START) {  // it's a number
                numbers[ptr] = x[primitive];
                while (ptr > 1) {
                    // for 2 argument functions
                    if (operations[ptr - 2] >= FSET_START  // before must be an operation
                            && operations[ptr - 2] <= FSET_2ARG_END // before must be an 2 argument operation (skip variables)
                            && operations[ptr - 1] == 0) {  // number is following the operation
                        int opp = operations[ptr - 2];
                        operations[ptr - 2] = 0;
                        double num1 = numbers[ptr - 1];
                        double num2 = numbers[ptr];
                        double result;
                        if (opp == ADD) {
                            result = num1 + num2;
                        } else if (opp == SUB) {
                            result = num1 - num2;
                        } else if (opp == MUL) {
                            result = num1 * num2;
                        } else if (opp == DIV) {
                            if (Math.abs(num2) <= DIVISION_CUT_OUT)
                                result = num1;
                            else
                                result = num1 / num2;
                        } else {
                            throw new IllegalArgumentException();
                        }
                        ptr -= 2;
                        numbers[ptr] = result;
                        // for 1 argument functions
                    } else if (operations[ptr - 1] > FSET_2ARG_END  // before must be a 1 argument operation
                            && operations[ptr - 1] <= FSET_END) {  // before must be an 1 argument operation (skip variables)
                        int opp = operations[ptr - 1];
                        operations[ptr - 1] = 0;
                        double num = numbers[ptr];
                        double result;
                        if (opp == EXP) {
                            result = num <= EXPONENT_CUT_OUT ? Math.exp(num) : num;
                        } else if (opp == SIN) {
                            result = Math.sin(Math.toRadians(num));
                        } else if (opp == COS) {
                            result = Math.cos(Math.toRadians(num));
                    } else {
                            throw new IllegalArgumentException();
                        }
                        ptr -= 1;
                        numbers[ptr] = result;
                    } else {
                        break;
                    }
                }
                ptr++;
            } else {  // it's an operation
                operations[ptr++] = primitive;
            }
        }
        length = ptr;
    }

    double runIterative(double[] stack) {
        int sp = 0;
        int pc = length - 1;

        while (pc >= 0) {
            int primitive = operations[pc--];

            if (primitive < FSET_START) {
                stack[sp++] = (primitive < varnumber) ? numbers[pc + 1] : variables[primitive - 1];
            } else {
                double result;
                switch (primitive) {
                    case ADD -> result = stack[--sp] + stack[--sp];
                    case SUB -> result = stack[--sp] - stack[--sp];
                    case MUL -> result = stack[--sp] * stack[--sp];
                    case DIV -> {
                        double num = stack[--sp];
                        double den = stack[--sp];
                        if (Math.abs(den) <= DIVISION_CUT_OUT)
                            result = num;
                        else result = num / den;
                    }
                    case EXP -> {
                        double num = stack[--sp];
                        result = num <= EXPONENT_CUT_OUT ? Math.exp(num) : num;
                    }
                    case SIN -> result = Math.sin(Math.toRadians(stack[--sp]));
                    case COS -> result = Math.cos(Math.toRadians(stack[--sp]));
                    default -> throw new IllegalStateException("Unknown op: " + primitive);
                }
                stack[sp++] = result;
            }
        }

        return stack[--sp];
    }

    double fitness_function_seq_iter(char [] prog) {
        double result, actual, fit = 0.0;
        simplify(prog);
        double[] stack = new double[length];
        for (int i = 0; i < fitnesscases; ++i ) {
            for (int j = 0; j < varnumber; ++j) {
                variables[j] = targets[i][j];
            }
            PC = 0;
            result = runIterative(stack);
            actual = targets[i][varnumber];
            fit += Math.abs(result - actual);
        }
        return -fit;
    }

    int grow(char [] buffer, int pos, int max, int depth) {
        char prim = (char) rd.nextInt(2);
        int one_child;

        if ( pos >= max )
            return( -1 );

        if ( pos == 0 )
            prim = 1;

        if ( prim == 0 || depth == 0 ) {
            prim = (char) rd.nextInt(varnumber + randomnumber);
            buffer[pos] = prim;
            return(pos+1);
        }
        else  {
            prim = (char) (rd.nextInt(FSET_END - FSET_START + 1) + FSET_START);
            if (prim <= FSET_2ARG_END) {  // 2 argument functions
                buffer[pos] = prim;
                one_child = grow(buffer, pos + 1, max, depth - 1);
                if (one_child < 0)
                    return -1 ;
                return grow(buffer, one_child, max, depth - 1);
            } else if (prim <= FSET_END) {  // 1 argument functions
                buffer[pos] = prim;
                return grow(buffer, pos + 1, max, depth - 1);
            }
        }
        return 0; // should never get here
    }

    int print_individual(char []buffer, int buffercounter ) {
        int a1=0, a2;
        if (buffer[buffercounter] < FSET_START) {
            if (buffer[buffercounter] < varnumber)
                System.out.print("X"+ (buffer[buffercounter] + 1));
            else
                System.out.print(x[buffer[buffercounter]]);
            return ++buffercounter;
        }
        switch(buffer[buffercounter]) {
            case ADD: System.out.print("(");
                a1 = print_individual(buffer, ++buffercounter);
                System.out.print(" + ");
                break;
            case SUB: System.out.print( "(");
                a1 = print_individual(buffer, ++buffercounter);
                System.out.print(" - ");
                break;
            case MUL: System.out.print("(");
                a1 = print_individual(buffer, ++buffercounter);
                System.out.print(" * ");
                break;
            case DIV: System.out.print("(");
                a1 = print_individual(buffer, ++buffercounter);
                System.out.print(" / ");
                break;
            case EXP: System.out.print("EXP(");
                a1 = ++buffercounter;
                break;
            case SIN: System.out.print("SIN(");
                a1 = ++buffercounter;
                break;
            case COS: System.out.print("COS(");
                a1 = ++buffercounter;
                break;
        }
        a2 = print_individual(buffer, a1);
        System.out.print(")");
        return a2;
    }


    static char [] buffer = new char[MAX_LEN];
    char [] create_random_individual(int depth ) {
        char [] ind;
        int len;

        do len = grow(buffer, 0, MAX_LEN, depth);
        while (len < 0);

        ind = new char[len];

        System.arraycopy(buffer, 0, ind, 0, len);
        return ind;
    }

    char [][] create_random_pop(int n, int depth, double [] fitness) {
        char [][]pop = new char[n][];
        int i;

        for ( i = 0; i < n; i ++ ) {
            pop[i] = create_random_individual( depth );
            fitness[i] = fitness_function_seq_iter( pop[i] );
        }
        return( pop );
    }


    void stats(double [] fitness, char [][] pop, int gen) {
        int i, best = rd.nextInt(POPSIZE);
        int node_count = 0;
        fbestpop = fitness[best];
        favgpop = 0.0;

        for ( i = 0; i < POPSIZE; i ++ ) {
            node_count += pop[i].length;
            favgpop += fitness[i];
            if ( fitness[i] > fbestpop ) {
                best = i;
                fbestpop = fitness[i];
            }
        }
        avg_len = (double) node_count / POPSIZE;
        favgpop /= POPSIZE;

        hist.add(new Hist(
                gen,
                -favgpop,
                -fbestpop,
                avg_len,
                pop[best]
        ));

        System.out.print("Generation="+gen+" Avg Fitness="+(-favgpop)+
                " Best Fitness="+(-fbestpop)+" Avg Size="+avg_len+
                "\nBest Individual: ");
        print_individual( pop[best], 0 );
        System.out.print( "\n");
        System.out.flush();
    }

    int tournament( double [] fitness, int tsize ) {  // select the best individual
        int best = rd.nextInt(POPSIZE), i, competitor;
        double  fbest = -1.0e34;

        for ( i = 0; i < tsize; i ++ ) {
            competitor = rd.nextInt(POPSIZE);
            if ( fitness[competitor] > fbest ) {
                fbest = fitness[competitor];
                best = competitor;
            }
        }
        return( best );
    }

    int negative_tournament( double [] fitness, int tsize ) {  // select the worst individual
        int worst = rd.nextInt(POPSIZE), i, competitor;
        double fworst = 1e34;

        for ( i = 0; i < tsize; i ++ ) {
            competitor = rd.nextInt(POPSIZE);
            if ( fitness[competitor] < fworst ) {
                fworst = fitness[competitor];
                worst = competitor;
            }
        }
        return( worst );
    }

    char [] crossover( char []parent1, char [] parent2 ) {
        int xo1start, xo1end, xo2start, xo2end;
        char [] offspring;
        int len1 = parent1.length;
        int len2 = parent2.length;
        int lenoff;

        xo1start =  rd.nextInt(len1);
        int opp_count = 0;
        int num_count = 0;
        for (xo1end = xo1start; xo1end < parent1.length; xo1end++ ) {  // calculate length
            if (parent1[xo1end] < FSET_START) {
                num_count++;
            } else if (parent1[xo1end] <= FSET_2ARG_END) {  // only count 2 argument functions
                opp_count++;
            }
            if (opp_count == num_count - 1) {
                xo1end++;
                break;
            }
        }

        xo2start =  rd.nextInt(len2);
        opp_count = 0;
        num_count = 0;
        for (xo2end = xo2start; xo2end < parent2.length; xo2end++ ) {  // calculate length
            if (parent2[xo2end] < FSET_START) {
                num_count++;
            } else if (parent2[xo2end] <= FSET_2ARG_END) {  // only count 2 argument functions
                opp_count++;
            }
            if (opp_count == num_count - 1) {
                xo2end++;
                break;
            }
        }

        lenoff = xo1start + (xo2end - xo2start) + (len1-xo1end);

        offspring = new char[lenoff];

        System.arraycopy( parent1, 0, offspring, 0, xo1start );
        System.arraycopy( parent2, xo2start, offspring, xo1start,
                (xo2end - xo2start) );
        System.arraycopy( parent1, xo1end, offspring,
                xo1start + (xo2end - xo2start),
                (len1-xo1end) );

        return( offspring );
    }

    char [] mutation( char [] parent, double pmut ) {
        int len = parent.length, i;
        int mutsite;
        char [] parentcopy = new char [len];

        System.arraycopy( parent, 0, parentcopy, 0, len );
        for (i = 0; i < len; i ++ ) {
            if ( rd.nextDouble() < pmut ) {
                mutsite =  i;
                if ( parentcopy[mutsite] < FSET_START )
                    parentcopy[mutsite] = (char) rd.nextInt(varnumber+randomnumber);
                else {
                    if (parentcopy[mutsite] <= FSET_2ARG_END) {  // 2 argument functions
                        parentcopy[mutsite] =
                                (char) (rd.nextInt(FSET_2ARG_END - FSET_START + 1)
                                        + FSET_START);
                    } else if (parent[mutsite] <= FSET_END) {  // 1 argument functions
                        parentcopy[mutsite] =
                                (char) (rd.nextInt(FSET_END - FSET_2ARG_END)
                                        + FSET_2ARG_END + 1);
                    }
                }
            }
        }
        return( parentcopy );
    }

    void print_params() {
        System.out.print("-- TINY GP (Java version) --\n");
        System.out.print("SEED="+seed+"\nMAX_LEN="+MAX_LEN+
                "\nPOPSIZE="+POPSIZE+"\nDEPTH="+DEPTH+
                "\nCROSSOVER_PROB="+CROSSOVER_PROB+
                "\nPMUT_PER_NODE="+PMUT_PER_NODE+
                "\nMIN_RANDOM="+minrandom+
                "\nMAX_RANDOM="+maxrandom+
                "\nGENERATIONS="+GENERATIONS+
                "\nTSIZE="+TSIZE+
                "\n----------------------------------\n");
    }

    public static List<Hist> hist = new ArrayList<>();

    static final DecimalFormat df = new DecimalFormat("0.00");
    void evolve() {
        setup();
        int gen, indivs, offspring, parent1, parent2, parent;
        double newfit;
        char []newind;
        print_params();
        stats( fitness, population, 0 );
        long firstStartTime = System.nanoTime();
        long startTime;
        long time;
        for ( gen = 1; gen < GENERATIONS; gen ++ ) {
            if (  fbestpop > -goal_fitness ) {
                time = System.nanoTime() - firstStartTime;
                int sec = (int) (time / 1_000_000_000.0);
                int min = sec / 60;
                sec = sec % 60;
                System.out.println("Took: " + min + "min " + sec + "s");
                System.out.print("PROBLEM SOLVED\n");
                return;
            }
            startTime = System.nanoTime();
            for ( indivs = 0; indivs < POPSIZE; indivs ++ ) {
                if ( rd.nextDouble() < CROSSOVER_PROB  ) {
                    parent1 = tournament( fitness, TSIZE );
                    parent2 = tournament( fitness, TSIZE );
                    newind = crossover( population[parent1], population[parent2] );
                }
                else {
                    parent = tournament( fitness, TSIZE );
                    newind = mutation( population[parent], PMUT_PER_NODE );
                }
                newfit = fitness_function_seq_iter(newind);
                offspring = negative_tournament( fitness, TSIZE );
                population[offspring] = newind;
                fitness[offspring] = newfit;
                int progress = (indivs / 1000) + 1;
                if (indivs % 1000 == 0) {
                    time = System.nanoTime() - startTime;
                    System.out.println("\\r["+"■".repeat(progress)+" ".repeat(100 - progress)+"] "
                            +progress+"%  "
                            + df.format(time/1_000_000_000.0) +"s");
                }
            }
            System.out.println();
            stats( fitness, population, gen );
        }
        time = System.nanoTime() - firstStartTime;
        int sec = (int) (time / 1_000_000_000.0);
        int min = sec / 60;
        sec = sec % 60;
        System.out.println("Took: " + min + "min " + sec + "s");
        System.out.print("PROBLEM *NOT* SOLVED\n");
    }

    public void setup() {
        fitness =  new double[POPSIZE];
        if (seed >= 0)
            rd.setSeed(seed);

        if (varnumber + randomnumber >= FSET_START )
            System.out.println("too many variables and constants");

        for (int i = 0; i < FSET_START; i ++)
            x[i]= (maxrandom-minrandom)*rd.nextDouble()+minrandom;
        variables = new double[varnumber];
        population = create_random_pop(POPSIZE, DEPTH, fitness);
    }

    public static double[] getX() {
        return TinyGP.x;
    }

    public static List<Hist> getHist() {
        return TinyGP.hist;
    }

    public TinyGP() {
//         GatewayServer server = new GatewayServer(this);
//         server.start();
        System.out.println("Server started");

        loadTargets();
    
        this.evolve();
        System.out.println("TOKEN");
        for (double v : TinyGP.x) {
            System.out.print(Double.toString(v) + " ");
        }
        System.out.println();
        for (Hist h : TinyGP.hist) {
            System.out.println(h);
        }
    }

    public static void main(String[] args) {
        // app is now the gateway.entry_point
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
        new TinyGP();
    }
}

    class Hist {
        public int gen;
        public double avg_fitness;
        public double best_fitness;
        public double avg_size;
        public char[] best_individual;

        public Hist(int gen, double avg_fitness, double best_fitness, double avg_size, char[] best_individual) {
            this.gen = gen;
            this.avg_fitness = avg_fitness;
            this.best_fitness = best_fitness;
            this.avg_size = avg_size;
            this.best_individual = new char[best_individual.length];
            System.arraycopy(best_individual, 0, this.best_individual, 0, best_individual.length);
        }

        public char[] getBest_individual() {
            return best_individual;
        }

        public double getAvg_size() {
            return avg_size;
        }

        public double getBest_fitness() {
            return best_fitness;
        }

        public double getAvg_fitness() {
            return avg_fitness;
        }

        public int getGen() {
            return gen;
        }

        @Override
        public String toString() {
        return gen + " " +
               avg_fitness + " " +
               best_fitness + " " +
               avg_size + " " +
               Arrays.toString(new String(best_individual).chars().toArray()).replaceAll("[\\[\\],]", "");
        }
    }
