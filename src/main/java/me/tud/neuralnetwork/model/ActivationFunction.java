package me.tud.neuralnetwork.model;

import java.util.function.Function;

public enum ActivationFunction {

    SIGMOID(x -> 1d / (1 + Math.exp(-x)), x -> x * (1 - x)),
    TANH(x -> {
        double a = Math.exp(x), b = Math.exp(-x);
        return (a - b) / (a + b);
    }, x -> 1 - (x * x)),
    RELU(x -> Math.max(0, x), x -> x > 0 ? 1d : 0d);

    private final Function<Double, Double> function;
    private final Function<Double, Double> derivative;

    ActivationFunction(Function<Double, Double> function, Function<Double, Double> derivative) {
        this.function = function;
        this.derivative = derivative;
    }

    public double apply(double x) {
        return function.apply(x);
    }

    public double applyDerivative(double x) {
        return derivative.apply(x);
    }

}
