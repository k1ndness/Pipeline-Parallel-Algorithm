package org.example;

public class SigmoidActivationFunction implements ActivationFunction {

    /**
     * Slope parameter
     */
    private double slope = 1d;


    /**
     * Creates a Sigmoid function with default slope value.
     */
    public SigmoidActivationFunction() {
    }

    /**
     * Creates a Sigmoid function with a slope parameter.
     *
     * @param slope
     *            slope parameter to be set
     */
    public SigmoidActivationFunction(double slope) {
        this.slope = slope;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double calculateOutput(double summedInput) {
        double denominator = 1 + Math.exp(-slope * summedInput);

        return (1d / denominator);
    }

    @Override
    public double calculateDerivative(double input) {
        return calculateOutput(input) * (1 - calculateOutput(input));
    }

}
