// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

// StepFunction is an implementation of an output step function.
type StepFunction struct {
	theta float64
}

// See OutputFunction#Output
func (fct StepFunction) Output(input float64) float64 {
	if input >= fct.theta {
		return 1
	}
	return 0
}

// Theta returns the threshold uesed by the StepFunction
func (fct *StepFunction) Theta() float64 {
	return fct.theta
}

// SetTheta sets theta to the given parameter.
func (fct *StepFunction) SetTheta(theta float64) {
	fct.theta = theta
}

// NewStepFunction returns a new StepFunction using the given parameter as a threshold.
func NewStepFunction(threshold float64) *StepFunction {
	return &StepFunction{threshold}
}
