// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// WeightedSumFunction is an implementation of a weighted sum over neuron inputs.
type WeightedSumFunction struct{}

// See InputFunction#Integrate
func (wsf WeightedSumFunction) Integrate(inputs map[Neuron]float64) float64 {
	var wSum float64

	for inputNeuron, weight := range inputs {
		wSum += (inputNeuron.Value() * weight)
	}
	return wSum
}

// NewWeightedSumFunction returns a pointer on a newly instantiated WeightedSumFunction.
func NewWeightedSumFunction() *WeightedSumFunction {
	return new(WeightedSumFunction)
}
