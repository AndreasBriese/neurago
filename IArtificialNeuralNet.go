// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// Interface ANNs need to implement
type IArtificialNeuralNet interface {
	Perceptrons() []*Perceptron
	SetPerceptrons([]*Perceptron)
	Output() []float64
	SetInput([]float64)
}
