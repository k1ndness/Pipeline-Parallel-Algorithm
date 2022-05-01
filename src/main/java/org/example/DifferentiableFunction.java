package org.example;

/**
 * Hello world!
 *
 */public interface DifferentiableFunction {

    /**
     * Performs calculation of function's derivative.
     *
     * @param totalInput
     *            neuron's total input
     *
     * @return function's derivative calculated based on the total input
     */
    double calculateDerivative(double totalInput);

}
