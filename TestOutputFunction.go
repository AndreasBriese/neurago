// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// TestOutputFunction is the interface used to implement output functions within neurons.
type TestOutputFunction struct {
	theta float64
}

// See OutputFunction#Output
func (fct TestOutputFunction) Output(input float64) float64 {
	if input >= fct.theta {
		return 1
	}
	return 0
}

// SetTheta sets theta to the given parameter.
func (fct *TestOutputFunction) SetTheta(theta float64) {
	fct.theta = theta
}

// NewTestOutputFunction returns a new TestOutputFunction using the given parameter
// as a threshold.
func NewTestOutputFunction(threshold float64) *TestOutputFunction {
	return &TestOutputFunction{threshold}
}
