package me.tud.neuralnetwork.util;

import java.io.Serializable;
import java.util.Arrays;

public class DataPair implements Serializable {

    private static final long serialVersionUID = 7359384787995021864L;

    private final double[] input, output;

    public DataPair(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }

    public double[] getInput() {
        return input;
    }

    public double[] getOutput() {
        return output;
    }

    public int getInputSize() {
        return input.length;
    }

    public int getOutputSize() {
        return output.length;
    }

    @Override
    public String toString() {
        return "Input: " + Arrays.toString(input) + ", Output: " + Arrays.toString(output);
    }

}
