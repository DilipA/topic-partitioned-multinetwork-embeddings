package experiments;

import java.io.IOException;

import mixedmembership.JointTextNetworkModel;
import util.LogRandoms;

public class ConditionalStructureAsymmetricBernoulliExperimentTest extends ExperimentTest {

	@Override
	public JointTextNetworkModel initializeModel(LogRandoms rng) {

		JointTextNetworkModel model = new JointTextNetworkModel(emails, T, -1,
				rng, true, true, false, true, false, false, false, false, true, false, false, null);

		model.initialize(null, null, null, null, alpha, beta, gamma, true, 0,
				1, null, wordStateFileName + ".true", edgeStateFileName
						+ ".true", null, null, null, null, null, null, null,
				null, null, missingEdgesFileName + ".true", 0, null, null,
				null, null, null, null, null, null);

		return model;
	}

	@Override
	public String[] getFirstExperimentArgs(int firstNumIter) {
		return new String[] { "-wf=" + wordMatrixFile, "-vf=" + vocabFile,
				"-ef=" + edgeMatrixFile, "-a=" + emails.getNumAuthors() + "",
				"-t=" + T + "", "-n=" + firstNumIter + "", "-p=" + "0",
				"-s=" + "1", "-v", "--alpha=" + alphaBase / T + "",
				"--beta=" + betaBase / emails.getWordDict().size() + "",
				"--gamma=" + gammaBase / 2 + "", "-h", "-e", "-of=" + "./" };
	}

	@Override
	public String[] getNextExperimentArgs(int nextNumIter, int itersDone) {
		return new String[] { "-wf=" + wordMatrixFile, "-vf=" + vocabFile,
				"-ef=" + edgeMatrixFile, "-a=" + emails.getNumAuthors() + "",
				"-t=" + T + "", "-n=" + nextNumIter + "", "-p=" + "0",
				"-s=" + "1", "-v", "--alpha=" + alphaBase / T + "",
				"--beta=" + betaBase / emails.getWordDict().size() + "",
				"--gamma=" + gammaBase / 2 + "", "-r ", "-rf=" + "./",
				"-i=" + itersDone + "", "-h", "-e", "-of=" + "./" };
	}

	@Override
	public JointTextNetworkModel runExperiment(String[] args, LogRandoms rng)
			throws IOException {
		return ConditionalStructureAsymmetricBernoulliExperiment.main(args, rng);
	}

	@Override
	public void otherAssertions(JointTextNetworkModel model,
			JointTextNetworkModel mainModel) {

		emailAssertion(mainModel);
		wordModelAssertions(model, mainModel);
		edgeModelAssertions(model, mainModel);
	}

	@Override
	public boolean usingEdgeModel() {
		return true;
	}
}
