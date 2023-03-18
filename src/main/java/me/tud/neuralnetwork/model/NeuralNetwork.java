package me.tud.neuralnetwork.model;

import me.tud.neuralnetwork.util.DataPair;
import me.tud.neuralnetwork.util.TrainSet;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.nio.file.Files;

public class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 1914115072288484963L;

    private final int inputSize;
    private final Layer[] hiddenLayers;
    private final Layer outputLayer;
    private ActivationFunction activationFunction = ActivationFunction.SIGMOID;

    public NeuralNetwork(int... layerSizes) {
        if (layerSizes.length < 2)
            throw new IllegalArgumentException("Layer sizes must be at least 2");
        this.inputSize = layerSizes[0];
        hiddenLayers = new Layer[layerSizes.length - 2];
        for (int i = 1; i < layerSizes.length - 1; i++)
            hiddenLayers[i - 1] = new Layer(this, layerSizes[i], layerSizes[i - 1]);
        outputLayer = new Layer(this, layerSizes[layerSizes.length - 1], layerSizes[layerSizes.length - 2]);
    }

    public double[] predict(double[] inputs) {
        if (inputs.length != inputSize)
            throw new IllegalArgumentException("The inputs must be of the input layer size");
        double[] layerOutputs = inputs;
        for (Layer hiddenLayer : hiddenLayers)
            layerOutputs = hiddenLayer.feedForward(layerOutputs);
        return outputLayer.feedForward(layerOutputs);
    }

    public void train(TrainSet data, int iterations, double learningRate) {
        if (data.getInputSize() != inputSize)
            throw new IllegalArgumentException("The input size of the given data doesn't match the input size of the network");
        if (data.getOutputSize() != outputLayer.size())
            throw new IllegalArgumentException("The output size of the given data doesn't match the output size of the network");
        for (int i = 0; i < iterations; i++) {
            for (DataPair pair : data)
                train(pair.getInput(), pair.getOutput(), learningRate);
        }
    }

    public void train(double[] inputs, double[] expectedOutputs, double learningRate) {
        if (inputs.length != inputSize)
            throw new IllegalArgumentException("The inputs must be of the input layer size");
        if (expectedOutputs.length != outputLayer.size())
            throw new IllegalArgumentException("The expected outputs must be of the output layer size");
        // Calculate errors
        double[] actualOutputs = predict(inputs);
        double[] outputErrors = new double[expectedOutputs.length];
        for (int i = 0; i < expectedOutputs.length; i++)
            outputErrors[i] = (expectedOutputs[i] - actualOutputs[i]) * activationFunction.applyDerivative(actualOutputs[i]);
        outputLayer.setErrors(outputErrors);

        for (int i = hiddenLayers.length - 1; i >= 0; i--) {
            Layer nextLayer = i == hiddenLayers.length - 1 ? outputLayer : hiddenLayers[i + 1];
            Layer hiddenLayer = hiddenLayers[i];
            hiddenLayer.getErrors(nextLayer);
        }

        // Update weights according to errors
        for (int i = 0; i < hiddenLayers.length; i++)
            hiddenLayers[i].updateWeights(i == 0 ? inputs : hiddenLayers[i - 1].getOutputs(), hiddenLayers[i].getErrors(), learningRate);
        outputLayer.updateWeights(hiddenLayers.length == 0 ? inputs : hiddenLayers[hiddenLayers.length - 1].getOutputs(), outputErrors, learningRate);
    }

    public int getInputSize() {
        return inputSize;
    }

    public Layer[] getHiddenLayers() {
        return hiddenLayers;
    }

    public Layer getOutputLayer() {
        return outputLayer;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public void save(@NotNull File targetFile) throws IOException {
        if (!targetFile.exists() && !targetFile.createNewFile())
            throw new IOException("Could not create file at '" + targetFile.getPath() + '\'');
        write(Files.newOutputStream(targetFile.toPath()));
    }

    public void write(OutputStream stream) throws IOException {
        try (ObjectOutputStream outputStream = new ObjectOutputStream(stream)) {
            outputStream.writeObject(this);
        }
    }

    public static NeuralNetwork loadNetwork(@NotNull File targetFile) throws IOException, IllegalArgumentException, ClassNotFoundException {
        if (!targetFile.exists())
            throw new IOException("Could not find file at '" + targetFile.getPath() + '\'');
        if (!targetFile.exists())
            throw new IOException("Could not find file at '" + targetFile.getPath() + '\'');
        try {
            return read(Files.newInputStream(targetFile.toPath()));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("File '" + targetFile.getPath() + "' does not contain a NeuralNetwork object");
        }
    }

    public static NeuralNetwork read(InputStream stream) throws IOException, IllegalArgumentException, ClassNotFoundException {
        try (ObjectInputStream inputStream = new ObjectInputStream(stream)) {
            Object object = inputStream.readObject();
            if (object instanceof NeuralNetwork)
                return (NeuralNetwork) object;
            throw new IllegalArgumentException("The given input stream does not contain a NeuralNetwork object");
        }
    }

}
