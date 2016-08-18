// Package neurago_test is the package which holds tests for
// github.com/lemourA/neurago
package neurago_test

import (
	"testing"

	"github.com/lemourA/neurago"
)

// OutputTest tests the Output method by checking that it correctly returns 1 or 0
// according to the given threshold.
func OutputTest(t *testing.T) {
	fct := neurago.NewStepFunction(4.2)

	if fct.Output(2.1) != 0 {
		t.Error("StepFunction#Output failed")
	}
	if fct.Output(4.2) != 1 {
		t.Error("StepFunction#Output failed")
	}
	if fct.Output(4.3) != 1 {
		t.Error("StepFunction#Output failed")
	}
}
