// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago

import (
	"math/rand"
	"time"
)

// TestANN is an Hopfield network used for test purposes
type TestANN struct {
	neurons []Neuron
}

// See Neuron#Neurons
func (hn *TestANN) Neurons() []Neuron {
	return hn.neurons
}

// See Neuron#SetNeurons
func (hn *TestANN) SetNeurons(neurons []Neuron) {
	hn.neurons = neurons
}

// See Neuron#Output
func (hn *TestANN) Output() []float64 {
	var updatedNeuron Neuron
	var weightedSum float64
	var stabilized bool
	var rIndex int
	neurons := hn.Neurons()
	nbOfNeurons := len(neurons)
	output := make([]float64, nbOfNeurons)
	rand.Seed(time.Now().UTC().UnixNano())

	for !stabilized {
		rIndex = rand.Intn(nbOfNeurons)
		updatedNeuron = neurons[rIndex]
		weightedSum = 0
		for connectedNeuron, weight := range updatedNeuron.Connections() {
			weightedSum += connectedNeuron.Value() * weight
		}
		if weightedSum >= 0 {
			updatedNeuron.SetValue(1)
		} else {
			updatedNeuron.SetValue(-1)
		}
		stabilized = isStable(hn)
	}
	for i, neuron := range hn.Neurons() {
		output[i] = neuron.Value()
	}
	return output
}

// See Neuron#SetInput
func (hn *TestANN) SetInput(inputPattern []float64) {
	for i, neuron := range hn.Neurons() {
		neuron.SetValue(inputPattern[i])
	}
}

// NewTestANN returns a pointer to a Hopfield Network initialised with
// the array of neurons given in parameter
func NewTestANN(neurons []Neuron) *TestANN {
	hn := &TestANN{neurons}

	for i, neuronA := range neurons {
		for j, neuronB := range neurons {
			if i != j {
				neuronA.SetConnection(neuronB, 0.0)
			}
		}
	}
	return hn
}
