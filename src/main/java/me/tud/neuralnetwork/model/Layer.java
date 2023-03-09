package me.tud.neuralnetwork.model;

import java.io.Serializable;

public class Layer implements Serializable {

    private static final long serialVersionUID = 3879050080783162556L;

    private final NeuralNetwork network;
    private final Neuron[] neurons;
    private final double[] lastErrors;

    public Layer(NeuralNetwork network, int neurons, int neuronInputs) {
        this.network = network;
        this.neurons = new Neuron[neurons];
        for (int i = 0; i < neurons; i++)
            this.neurons[i] = new Neuron(this, neuronInputs);
        this.lastErrors = new double[neurons];
    }

    public int size() {
        return neurons.length;
    }

    public double[] feedForward(double[] inputs) {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++)
            outputs[i] = neurons[i].feedForward(inputs);
        return outputs;
    }

    public double[] getOutputs() {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++)
            outputs[i] = neurons[i].getOutput();
        return outputs;
    }

    public double[] getErrors(Layer nextLayer) {
        for (int i = 0; i < neurons.length; i++)
            lastErrors[i] = neurons[i].getError(nextLayer, i);
        return lastErrors;
    }

    public double[] getErrors() {
        return lastErrors;
    }

    public void setErrors(double[] errors) {
        System.arraycopy(errors, 0, lastErrors, 0, lastErrors.length);
    }

    public void updateWeights(double[] inputs, double[] errors, double learningRate) {
        for (int i = 0; i < neurons.length; i++)
            neurons[i].updateWeights(inputs, errors[i], learningRate);
    }

    public NeuralNetwork getNetwork() {
        return network;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

}
