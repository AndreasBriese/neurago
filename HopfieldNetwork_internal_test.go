// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"reflect"
	"testing"
)

// TestNeurons tests the method "Neurons" by checking that it correctly returns the
// Hopfield Network set of neurons.
func TestNeurons(t *testing.T) {
	neurons := []Neuron{
		NewTestNeuron(1), NewTestNeuron(1),
		NewTestNeuron(1), NewTestNeuron(1),
		NewTestNeuron(1), NewTestNeuron(1),
		NewTestNeuron(1), NewTestNeuron(1),
		NewTestNeuron(1), NewTestNeuron(1),
	}
	hopNet, err := NewHopfieldNetwork(neurons)
	if err != nil {
		t.Error("HopfieldNetwork#Neurons failed: Error returned by NewHopfieldNetwork")
	}
	if !reflect.DeepEqual(neurons, hopNet.Neurons()) {
		t.Error("HopfieldNetwork#Neurons failed")
	}
}

// TestSetNeurons tests the method "SetNeurons" by checking that it correctly sets
// the given array of neurons as the new set of neurons.
func TestSetNeurons(t *testing.T) {
	hopNet, err := NewHopfieldNetwork([]Neuron{
		NewTestNeuron(1), NewTestNeuron(1),
		NewTestNeuron(1), NewTestNeuron(1),
		NewTestNeuron(1), NewTestNeuron(1),
		NewTestNeuron(1), NewTestNeuron(1),
		NewTestNeuron(1), NewTestNeuron(1),
	})
	neurons := []Neuron{}

	if err != nil {
		t.Error("HopfieldNetwork#Neurons failed: Error returned by NewHopfieldNetwork")
	}
	for i := 0; i < 10; i++ {
		neurons = append(neurons, NewTestNeuron(0))
	}
	hopNet.SetNeurons(neurons)
	if !reflect.DeepEqual(neurons, hopNet.neurons) {
		t.Error("HopfieldNetwork#SetNeurons failed")
	}
}

// TestOutput tests the method "Output" by checking that it correctly computes and returns
// the output of the calling Hopfield Network.
// TODO: make a table driven test version
func TestOutput(t *testing.T) {
	hopNet, err := NewHopfieldNetwork([]Neuron{
		NewTestNeuron(0), NewTestNeuron(0), NewTestNeuron(0),
	})
	if err != nil {
		t.Error("HopfieldNetwork#Neurons failed: Error returned by NewHopfieldNetwork")
	}
	trainer := NewTestTrainer()
	trainingPattern := []float64{-1, 1, -1}

	trainer.Train(hopNet, [][]float64{trainingPattern})
	hopNet.neurons[0].SetValue(-1)
	hopNet.neurons[1].SetValue(-1)
	hopNet.neurons[2].SetValue(-1)
	if !reflect.DeepEqual(hopNet.Output(), trainingPattern) {
		t.Error("HopfieldNetwork#Output failed")
	}
}

// TestSetInput tests the method "SetInput" by checking that it correctly sets
// the input pattern of the calling Hopfield Network.
func TestSetInput(t *testing.T) {
	hopNet, err := NewHopfieldNetwork([]Neuron{
		NewTestNeuron(0), NewTestNeuron(0), NewTestNeuron(0),
	})
	if err != nil {
		t.Error("HopfieldNetwork#Neurons failed: Error returned by NewHopfieldNetwork")
	}
	inputPattern := []float64{-1, -1, -1}

	hopNet.SetInput(inputPattern)
	for i, neuron := range hopNet.neurons {
		if neuron.Value() != inputPattern[i] {
			t.Error("HopfieldNetwork#SetInput")
		}
	}
}
