// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"errors"
	"math/rand"
	"time"
)

// HopfieldNetwork is the type for Hopfield recurrent neural networks
type HopfieldNetwork struct {
	//	neurons []Neuron
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

// See Neuron#Output
func (hn *HopfieldNetwork) Output() []float64 {
	var stabilized, updated bool
	neurons := make([]Neuron, len(hn.Neurons()))
	nbOfNeurons := len(neurons)
	output := make([]float64, nbOfNeurons)
	copy(neurons, hn.Neurons())
	rand.Seed(time.Now().UTC().UnixNano())
	//start := time.Now()
	for !stabilized {
		shuffleNeurons(neurons)
		updated = false
		for _, neuron := range neurons {
			if neuron.Update() == true {
				updated = true
			}
		}
		stabilized = !updated
		// if time.Since(start) > (time.Minute * 3) {
		// 	fmt.Printf("Network stabilization is taking too long, ")
		// 	fmt.Printf("checking weight symmetry... ")
		// 	if !isSymmetric(hn) {
		// 		fmt.Println("Not symmetric.")
		// 	} else {
		// 		fmt.Println("Symmetric.")
		// 	}
		// }
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

// computeLocalField returns the local field of a neuron
func computeLocalField(n Neuron) float64 {
	var localField float64
	connections := n.Connections()

	for connectedNeuron, weight := range connections {
		localField += weight * connectedNeuron.Value()
	}
	return localField
}

// Checks if the network is symmetric
func isSymmetric(net ANN) bool {
	neurons := net.Neurons()
	for i, nA := range neurons {
		for j, nB := range neurons {
			if i != j {
				if nA.Connections()[nB] != nB.Connections()[nA] {
					return false
				}
			}
		}
	}
	return true
}
