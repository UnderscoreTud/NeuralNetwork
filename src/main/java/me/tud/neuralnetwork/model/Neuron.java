package me.tud.neuralnetwork.model;

import java.io.Serial;
import java.io.Serializable;
import java.util.concurrent.ThreadLocalRandom;

public class Neuron implements Serializable {

    @Serial
    private static final long serialVersionUID = 1093856207263835942L;

    private final NeuralNetwork network;
    private final Layer layer;
    private final double[] weights;
    private double bias, output;

    public Neuron(Layer layer, int inputs) {
        this.network = layer.getNetwork();
        this.layer = layer;
        this.weights = new double[inputs];
        for (int i = 0; i < inputs; i++)
            this.weights[i] = ThreadLocalRandom.current().nextDouble() * 2 - 1;
        this.bias = ThreadLocalRandom.current().nextDouble() * 2 - 1;
    }

    public double feedForward(double[] inputs) {
        checkInputsSize(inputs.length);
        double sum = bias;
        for (int i = 0; i < weights.length; i++)
            sum += inputs[i] * weights[i];
        return output = network.getActivationFunction().apply(sum);
    }

    public double getError(Layer nextLayer, int position) {
        double sum = 0;
        for (int i = 0; i < nextLayer.size(); i++)
            sum += nextLayer.getErrors()[i] * nextLayer.getNeurons()[i].weights[position];
        return sum * network.getActivationFunction().applyDerivative(output);
    }

    public void updateWeights(double[] inputs, double error, double learningRate) {
        checkInputsSize(inputs.length);
        double delta = error * network.getActivationFunction().applyDerivative(feedForward(inputs));
        bias += learningRate * delta;
        for (int i = 0; i < weights.length; i++)
            weights[i] += inputs[i] * delta * learningRate;
    }

    public NeuralNetwork getNetwork() {
        return network;
    }

    public Layer getLayer() {
        return layer;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        System.arraycopy(this.weights, 0, weights, 0, this.weights.length);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getOutput() {
        return output;
    }

    private void checkInputsSize(int inputSize) {
        if (weights.length != inputSize)
            throw new IllegalArgumentException("The input amount must be the same as the input size of the neuron");
    }
}
