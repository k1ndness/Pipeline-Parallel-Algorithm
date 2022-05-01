package org.example;
import java.util.List;
public interface ProcessingUnit {
    String getId();

    double getOutput();

    void setOutput(double output);

    void calculateOutput();

    Connection getConnection(String neuronId);

    List<Connection> getInputConnections();
}
