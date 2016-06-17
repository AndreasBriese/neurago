// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"fmt"
)

// PrintPattern is used to print out patterns on the standard output
func PrintPattern(pattern []float64, unitsPerLine int) {
	for i, elt := range pattern {
		if elt == -1 {
			fmt.Printf("0")
		} else {
			fmt.Printf("1")
		}
		if (i+1)%unitsPerLine == 0 && i != 0 {
			fmt.Printf("\n")
		}
	}
}
