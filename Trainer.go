// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// Trainer is the interface each learning rule and training methods
// must implement.
type Trainer interface {
	// Train teaches the given patterns to a network
	Train(ANN, [][]float64)
}
