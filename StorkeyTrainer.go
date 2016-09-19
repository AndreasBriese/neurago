// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import "github.com/gonum/matrix/mat64"

// StorkeyTrainer trains network using the storkey learning rule
type StorkeyTrainer struct{}

func storkeyLocalField(net ANN, nA Neuron, pat []float64, a, b int) float64 {
	var localField float64
	neurons := net.Neurons()

	for i, nB := range neurons {
		if i != a && i != b {
			localField += nA.Connections()[nB] * pat[i]
		}
	}
	return localField
}

func isAlreadyDone(a, b int, slice [][]int) bool {
	for _, elt := range slice {
		if (elt[0] == a && elt[1] == b) || (elt[0] == b && elt[1] == a) {
			return true
		}
	}
	return false
}

// See Trainer#Train
func (t StorkeyTrainer) Train(net ANN, patterns [][]float64) {
	var a, b, c float64
	var localFieldA, localFieldB float64
	neurons := net.Neurons()
	nbOfNeurons := float64(len(neurons))
	weights := mat64.NewDense(int(nbOfNeurons), int(nbOfNeurons),
		make([]float64, int(nbOfNeurons*nbOfNeurons)))

	for _, pat := range patterns {
		for i, nA := range neurons {
			for j, nB := range neurons {
				if i != j {
					localFieldA = storkeyLocalField(net, nA, pat, i, j)
					localFieldB = storkeyLocalField(net, nB, pat, i, j)
					a = (1.0 / nbOfNeurons) * pat[i] * pat[j]
					b = (1.0 / nbOfNeurons) * pat[i] * localFieldA
					c = (1.0 / nbOfNeurons) * pat[j] * localFieldB
					weights.Set(i, j, a-b-c)
				}
			}
		}
		for i, nA := range neurons {
			for j, nB := range neurons {
				nA.SetConnection(nB, nA.Connections()[nB]+weights.At(i, j))
			}
		}
	}
}

// NewStorkeyTrainer returns a newly intantiated StorkeyTrainer
func NewStorkeyTrainer() *StorkeyTrainer {
	return new(StorkeyTrainer)
}
