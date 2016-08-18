// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// TestInputFunction is an implementation of a weighted sum over neuron inputs.
type TestInputFunction struct{}

// See InputFunction#Integrate
func (wsf TestInputFunction) Integrate(inputs map[Neuron]float64) float64 {
	var wSum float64

	for inputNeuron, weight := range inputs {
		wSum += (inputNeuron.Value() * weight)
	}
	return wSum
}

// NewTestInputFunction returns a pointer on a newly instantiated TestInputFunction.
func NewTestInputFunction() *TestInputFunction {
	return new(TestInputFunction)
}
