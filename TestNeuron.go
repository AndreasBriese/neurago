// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import "log"

// TestNeuron is the implementation of a McCulloch-Pitts artificial neuron for
// test purposes.
type TestNeuron struct {
	value       float64
	outputFn    OutputFunction
	inputFn     InputFunction
	connections map[Neuron]float64
}

// See Neuron#Value
func (n TestNeuron) Value() float64 {
	if outFn := n.OutputFunction(); outFn != nil {
		return outFn.Output(n.value)*2 - 1
	}
	log.Panicln("TestNeuron#Value -> Cannot return the neuron value, no output function defined.")
	return 0
}

// See Neuron#SetValue
func (n *TestNeuron) SetValue(value float64) {
	n.value = value
}

// See Neuron#OutputFunction
func (n TestNeuron) OutputFunction() OutputFunction {
	return n.outputFn
}

// See Neuron#SetOutputFunction
func (n *TestNeuron) SetOutputFunction(outFn OutputFunction) {
	n.outputFn = outFn
}

// See Neuron#InputFunction
func (n TestNeuron) InputFunction() InputFunction {
	return n.inputFn
}

// See Neuron#SetInputFunction
func (n *TestNeuron) SetInputFunction(inFn InputFunction) {
	n.inputFn = inFn
}

// See Neuron#Connections
func (n TestNeuron) Connections() map[Neuron]float64 {
	return n.connections
}

// See Neuron#SetConnections
func (n *TestNeuron) SetConnections(connections map[Neuron]float64) {
	n.connections = connections
}

// See Neuron#SetConnection
func (n *TestNeuron) SetConnection(neuron Neuron, weight float64) {
	n.connections[neuron] = weight
}

// See Neuron#Update
func (n *TestNeuron) Update() {
	if inFn := n.InputFunction(); inFn != nil {
		n.SetValue(inFn.Integrate(n.Connections()))
	} else {
		log.Panicln("TestNeuron#Update -> Cannot update the neuron value, no input function defined.")
	}
}

// NewTestNeuron returns a new, initialised, McCulloch-Pitts Neuron
// Dependencies: StepFunction.go and WeightedSumFunction.go
func NewTestNeuron(value float64, outputThreshold float64) *TestNeuron {
	neuron := &TestNeuron{
		value,
		NewTestOutputFunction(outputThreshold),
		NewTestInputFunction(),
		make(map[Neuron]float64),
	}

	return neuron
}
