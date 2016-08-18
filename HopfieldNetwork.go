// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"errors"
	"fmt"
	"math/rand"
	"time"
)

// HopfieldNetwork is the type for Hopfield recurrent neural networks
type HopfieldNetwork struct {
	neurons []Neuron
}

// See Neuron#Neurons
func (hn *HopfieldNetwork) Neurons() []Neuron {
	return hn.neurons
}

// See Neuron#SetNeurons
func (hn *HopfieldNetwork) SetNeurons(neurons []Neuron) {
	hn.neurons = neurons
}

func neuronsToValues(neurons []Neuron) []float64 {
	values := []float64{}

	for _, neuron := range neurons {
		values = append(values, neuron.Value())
	}
	return values
}

func printWeightMatrix(neurons []Neuron) {
	for _, neuron := range neurons {
		for _, weight := range neuron.Connections() {
			fmt.Printf("%f ", weight)
		}
		fmt.Println()
	}
}

// See Neuron#Output
func (hn *HopfieldNetwork) Output() []float64 {
	var stabilized bool
	neurons := make([]Neuron, len(hn.Neurons()))
	nbOfNeurons := len(neurons)
	output := make([]float64, nbOfNeurons)

	copy(neurons, hn.Neurons())
	rand.Seed(time.Now().UTC().UnixNano())
	for !stabilized {
		fmt.Println(neuronsToValues(neurons))
		printWeightMatrix(neurons)
		shuffleNeurons(neurons)
		fmt.Println(neuronsToValues(neurons))
		printWeightMatrix(neurons)
		for _, neuron := range neurons {
			neuron.Update()
		}
		stabilized = isStable(hn)
	}
	for i, neuron := range hn.Neurons() {
		output[i] = neuron.Value()
	}
	return output
}

// See Neuron#SetInput
func (hn *HopfieldNetwork) SetInput(inputPattern []float64) {
	for i, neuron := range hn.Neurons() {
		neuron.SetValue(inputPattern[i])
	}
}

// NewHopfieldNetwork returns a pointer to a Hopfield Network initialised with
// the array of neurons given in parameter.
func NewHopfieldNetwork(neurons []Neuron) (*HopfieldNetwork, error) {
	hn := &HopfieldNetwork{neurons}
	for i, neuronA := range neurons {
		for j, neuronB := range neurons {
			if neuronA == nil || neuronB == nil {
				return nil, errors.New("Uninitialized neuron")
			}
			if i != j {
				neuronA.SetConnection(neuronB, 0.0)
			}
		}
	}
	return hn, nil
}

// isStable is used to test if the given network is stable
func isStable(n ANN) bool {
	var localField float64
	neurons := n.Neurons()
	nbOfNeurons := len(neurons)
	count := 0

	for _, neuron := range neurons {
		localField = computeLocalField(neuron)
		if localField >= 0 && neuron.Value() == 1 ||
			localField < 0 && neuron.Value() == -1 {

			count++
		}
	}

	if count == nbOfNeurons {
		return true
	}
	return false
}

// shuffleNeurons shuffle randomly a slice of neurons
func shuffleNeurons(neurons []Neuron) {
	for i := range neurons {
		j := rand.Intn(i + 1)
		neurons[i], neurons[j] = neurons[j], neurons[i]
	}
}
