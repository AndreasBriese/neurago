// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import "log"

// HebbTrainer trains network using the hebb learning rule
type HebbTrainer struct{}

// See Trainer#Train
func (t HebbTrainer) Train(net ANN, patterns [][]float64) {
	var connections map[Neuron]float64
	neurons := net.Neurons()
	nbOfNeurons := len(neurons)
	nbOfPatterns := len(patterns)

	if neurons == nil {
		log.Panicln("Runtime Error: called method 'Train' on an uninitialized network")
	}
	for _, pattern := range patterns {
		for i, value := range pattern {
			if i >= nbOfNeurons {
				log.Panicln("Runtime Error: Not enough neurons to represent the pattern")
			}
			neurons[i].SetValue(value)
		}
		for _, neuron := range neurons {
			connections = neuron.Connections()
			for inputNeuron, weight := range connections {
				weight = 1 / float64(nbOfPatterns) * neuron.Value() * inputNeuron.Value()
				neuron.SetConnection(inputNeuron, connections[inputNeuron]+weight)
			}
		}
	}
}

// NewHebbTrainer returns a newly intantiated HebbTrainer
func NewHebbTrainer() *HebbTrainer {
	return new(HebbTrainer)
}
