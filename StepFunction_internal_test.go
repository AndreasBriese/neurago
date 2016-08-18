// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import "testing"

func ThetaTest(t *testing.T) {
	fct := &StepFunction{0}

	fct.theta = 12
	if fct.Theta() != 12 {
		t.Error("StepFunction#Theta failed")
	}
}

// SetThetaTest tests the method SetTheta by ensuring that it correctly sets theta
// to the provided argument.
func SetThetaTest(t *testing.T) {
	fct := &StepFunction{0}

	if fct.theta != 0 {
		t.Error("StepFunction#SetTheta failed")
	}
	fct.SetTheta(-1)
	if fct.theta != -1 {
		t.Error("StepFunction#SetTheta failed")
	}
}
