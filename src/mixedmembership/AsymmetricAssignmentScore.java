package mixedmembership;

import data.EmailCorpus;
import util.Probability;

import java.io.*;
import java.util.Arrays;
import java.util.zip.GZIPOutputStream;

/**
 * Created by dilip on 12/2/16.
 */
public class AsymmetricAssignmentScore extends AssignmentScore {

    private final double[] asymmetricPrior;

    public AsymmetricAssignmentScore(int numDocuments, int numTopics, double[] alpha, double[] alphaPrime,
                           JointStructure modelStructure) {
        super(numDocuments, numTopics, alpha, modelStructure);
        if(alphaPrime == null) {
            this.asymmetricPrior = setAsymmetricPrior(numTopics);
        }
        else{
            this.asymmetricPrior = null;
        }
    }

    public double[] softmax(double[] x){
        double[] ret = new double[x.length];
        double norm = 0.0;
        for(int i=0;i < x.length;i++) {
            double exp = Math.exp(x[i]);
            ret[i] = exp;
            norm += exp;
        }

        for(int i=0;i < ret.length;i++){
            ret[i] /= norm;
        }

        return ret;
    }

    public double[] setAsymmetricPrior(int numTopics){
        double[] ret = new double[numTopics];
        Arrays.fill(ret, 0.0);
        ret[ret.length-1] = 1.0;
        return this.softmax(ret);
    }

    @Override
    public double getLogValue(int doc, int topic){
        return Math.log(componentElementCounts[doc][topic] + alpha[0]*this.asymmetricPrior[topic]);
    }

    public void print(EmailCorpus emails, String fileName) {
        print(emails, 0.0, -1, fileName);
    }

    public void print(EmailCorpus emails, double threshold, int numTopics,
                      String fileName) {

        try {

            PrintStream pw = new PrintStream(new GZIPOutputStream(
                    new BufferedOutputStream(new FileOutputStream(new File(
                            fileName)))));

            pw.println("#doc source topic proportion ...");

            Probability[] probs = new Probability[numElements];

            for (int d = 0; d < numComponents; d++) {

                pw.print(d);
                pw.print(" ");
                if (numComponents == 1) {
                    pw.print("corpus");
                } else {
                    pw.print(emails.getEmail(d).getSource());
                }
                pw.print(" ");

                for (int t = 0; t < numElements; t++)
                    probs[t] = new Probability(t, Math.exp(getLogScore(d, t)));

                Arrays.sort(probs);

                if ((numTopics > numElements) || (numTopics < 0))
                    numTopics = numElements;

                for (int i = 0; i < numTopics; i++) {

                    // break if there are no more topics whose proportion is
                    // greater than zero or threshold...

                    if ((probs[i].prob == 0) || (probs[i].prob < threshold))
                        break;

                    pw.print(probs[i].index);
                    pw.print(" ");
                    pw.print(probs[i].prob);
                    pw.print(" ");
                }

                pw.println();
            }

            pw.close();
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    @Override
    public double logProb() {
        return modelStructure.logPriorProb();
    }

}
