// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// InputFunction is the interface used to implement neuron input functions.
type InputFunction interface {
	// Integrate does the synaptic integration from inputs and returns a result.
	Integrate(map[Neuron]float64) float64
}
