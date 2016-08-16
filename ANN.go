// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// ANN is the interface Artificial Neural Network types have to implement.
type ANN interface {
	// Neurons returns neurons of the calling artificial neural network.
	Neurons() []Neuron
	// SetNeurons sets the neurons of the calling ANN.
	SetNeurons([]Neuron)
	// Output computes an output from the calling ANN inputs and returns it.
	Output() []float64
	// SetInput sets the given pattern as the input pattern of the calling ANN.
	SetInput([]float64)
}
