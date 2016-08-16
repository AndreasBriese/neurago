// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// Trainer is the interface each learning rule and training methods
// must implement.
type Trainer interface {
	// Train takes an artificial neural network and an array of patterns to learn as
	// its arguments and trained the given network to the given patterns.
	Train(ANN, [][]float64)
}
