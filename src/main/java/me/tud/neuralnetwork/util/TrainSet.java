package me.tud.neuralnetwork.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class TrainSet implements Iterable<DataPair>, Serializable {

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
        return Collections.unmodifiableList(data).iterator();
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

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("[\n");
        int i = 0;
        for (DataPair pair : data)
            builder.append('\t').append(++i).append(". ").append(pair).append('\n');
        builder.append(']');
        return builder.toString();
    }

    public static TrainSet of(double[][] data) {
        TrainSet trainSet = new TrainSet(data[0].length, data[1].length);
        for (int i = 0; i < data.length; i += 2)
            trainSet.addData(data[i], data[i + 1]);
        return trainSet;
    }

}
