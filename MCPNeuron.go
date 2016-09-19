// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"local/utils/math"
	"log"
)

// MCPNeuron is the implementation of a McCulloch-Pitts artificial neuron.
type MCPNeuron struct {
	value       float64
	outputFn    OutputFunction
	inputFn     InputFunction
	connections map[Neuron]float64
}

// See Neuron#Value
func (n MCPNeuron) Value() float64 {
	if outFn := n.OutputFunction(); outFn != nil {
		return outFn.Output(n.value)*2 - 1
	}
	log.Panicln("MCPNeuron#Value -> Cannot return the neuron value, no output function defined.")
	return 0
}

func (n MCPNeuron) Val() float64 {
	return n.value
}

// See Neuron#SetValue
func (n *MCPNeuron) SetValue(value float64) {
	n.value = value
}

// See Neuron#OutputFunction
func (n MCPNeuron) OutputFunction() OutputFunction {
	return n.outputFn
}

// See Neuron#SetOutputFunction
func (n *MCPNeuron) SetOutputFunction(outFn OutputFunction) {
	n.outputFn = outFn
}

// See Neuron#InputFunction
func (n MCPNeuron) InputFunction() InputFunction {
	return n.inputFn
}

// See Neuron#SetInputFunction
func (n *MCPNeuron) SetInputFunction(inFn InputFunction) {
	n.inputFn = inFn
}

// See Neuron#Connections
func (n MCPNeuron) Connections() map[Neuron]float64 {
	return n.connections
}

// See Neuron#SetConnections
func (n *MCPNeuron) SetConnections(connections map[Neuron]float64) {
	n.connections = connections
}

// See Neuron#SetConnection
func (n *MCPNeuron) SetConnection(neuron Neuron, weight float64) {
	n.connections[neuron] = weight
}

// See Neuron#Update
func (n *MCPNeuron) Update() bool {
	if inFn := n.InputFunction(); inFn != nil {
		val := n.value
		n.SetValue(math.ToFixed(inFn.Integrate(n.Connections()), 0))
		if val != n.value {
			return true
		}
	} else {
		log.Panicln("MCPNeuron#Update -> Cannot update the neuron value, no input function defined.")
	}
	return false
}

// NewMCPNeuron returns a new, initialised, McCulloch-Pitts Neuron
// Dependencies: StepFunction.go and WeightedSumFunction.go
func NewMCPNeuron(value float64, outputThreshold float64) *MCPNeuron {
	neuron := &MCPNeuron{
		value,
		NewStepFunction(outputThreshold),
		NewWeightedSumFunction(),
		make(map[Neuron]float64),
	}

	return neuron
}
