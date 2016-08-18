// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// OutputFunction is the interface used to implement neuron output functions.
type OutputFunction interface {
	// Output applies a function to the given input and returns the result.
	Output(float64) float64
}
