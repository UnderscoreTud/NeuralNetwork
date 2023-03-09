package me.tud.neuralnetwork.util;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class TrainSet implements Iterable<DataPair>, Serializable {

    @Serial
    private static final long serialVersionUID = 6969851092691793249L;

    private final List<DataPair> data = new ArrayList<>();
    private final int inputSize, outputSize;

    public TrainSet(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public void addData(double[] input, double[] output) {
        if (input.length != inputSize)
            throw new IllegalArgumentException("'input' does not match 'inputSize'");
        if (output.length != outputSize)
            throw new IllegalArgumentException("'output' does not match 'outputSize'");
        data.add(new DataPair(input, output));
    }

    public double[] getInput(int index) {
        return data.get(index).getInput();
    }

    public double[] getOutput(int index) {
        return data.get(index).getOutput();
    }

    public DataPair get(int index) {
        return data.get(index);
    }

    @Override
    public Iterator<DataPair> iterator() {
        return List.copyOf(data).iterator();
    }

    public List<DataPair> getData() {
        return data;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public int size() {
        return data.size();
    }

}
