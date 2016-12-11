package mixedmembership;

import cc.mallet.util.ArrayUtils;
import data.EmailCorpus;
import util.Probability;

import java.io.*;
import java.util.Arrays;
import java.util.zip.GZIPOutputStream;

/**
 * Created by dilip on 12/2/16.
 */
public class AsymmetricAssignmentScore extends AssignmentScore {

    private final double[] alphaPrime;
    private final double[] asymmetricPrior;
    private final AssignmentScore integratedM;

    public AsymmetricAssignmentScore(int numDocuments, int numTopics, double[] alpha, double[] alphaPrime,
                           JointStructure modelStructure) {
        super(numDocuments, numTopics, alpha, modelStructure);
        this.alphaPrime = alphaPrime;
        if(alphaPrime == null) {
            this.asymmetricPrior = setAsymmetricPrior(numTopics);
            this.integratedM = null;
        }
        else{
            this.asymmetricPrior = null;
            this.integratedM = new AssignmentScore(numDocuments, numTopics, alphaPrime, modelStructure);
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

    // for keeping data structures updated
    @Override
    public void incrementCounts(int c, int i) {
        componentElementCounts[c][i]++;
        componentCountsNorm[c]++;
        if(this.integratedM != null){
            this.integratedM.incrementCounts(c, i);
        }
    }

    // for keeping data structures updated
    @Override
    public void decrementCounts(int c, int i) {
        componentElementCounts[c][i]--;
        componentCountsNorm[c]--;
        assert componentElementCounts[c][i] >= 0;
        if(this.integratedM != null){
            this.integratedM.decrementCounts(c, i);
        }
    }

    // clear data structures
    @Override
    public void resetCounts() {

        for (int c = 0; c < numComponents; c++)
            Arrays.fill(componentElementCounts[c], 0);

        Arrays.fill(componentCountsNorm, 0);
        if(this.integratedM != null){
            this.integratedM.resetCounts();
        }
    }

    @Override
    public double getLogValue(int doc, int topic){
        if(this.integratedM == null) {
            return Math.log(componentElementCounts[doc][topic] + alpha[0] * this.asymmetricPrior[topic]);
        }
        else{
            return Math.log(componentElementCounts[doc][topic] + alpha[0] * this.integratedM.getLogScore(doc, topic));
        }
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

    @Override
    public double[] getSliceSamplableParameters() {
        if(this.alphaPrime == null){
            return alpha;
        }
        else{
            return ArrayUtils.append(alpha, alphaPrime);
        }
    }

    @Override
    public void setSliceSamplableParameters(double[] newValues) {
        System.arraycopy(newValues, 0, alpha, 0, alpha.length);
        if(this.alphaPrime != null) {
            System.arraycopy(newValues, alpha.length+1, alphaPrime, 0, alphaPrime.length);
        }
    }

}
