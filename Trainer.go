// Package neurago is a little library providing tools to
// implement artificial neural networks
package neurago

type Trainer interface {
	Train([]*Perceptron, [][]float64)
}
