package me.tud.neuralnetwork.util;

import java.io.Serializable;

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

}
