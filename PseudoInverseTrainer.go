// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import "github.com/gonum/matrix/mat64"

// Pseudoinverse trainer which trains networks using the hebb learning rule
type PseudoInverseTrainer struct{}

func buildWeightMatrix(net ANN) *mat64.Dense {
	var conn map[Neuron]float64
	neurons := net.Neurons()
	nbOfNeurons := len(neurons)
	weightMat := make([]float64, nbOfNeurons*nbOfNeurons)

	for i, neuronA := range neurons {
		conn = neuronA.Connections()
		for j, neuronB := range neurons {
			if i != j {
				weightMat[i*nbOfNeurons+j] = conn[neuronB]
			} else {
				weightMat[i*nbOfNeurons+j] = 0
			}
		}
	}
	return mat64.NewDense(nbOfNeurons, nbOfNeurons, weightMat)
}

// See Trainer#Train
func (t PseudoInverseTrainer) Train(net ANN, patterns [][]float64) {
	neurons := net.Neurons()

	hebbTrainer := NewHebbTrainer()
	hebbTrainer.Train(net, patterns)

	wMat := buildWeightMatrix(net)
	wmr, wmc := wMat.Dims()

	wMatT := wMat.T()

	wMatProduct := mat64.NewDense(wmr, wmc, make([]float64, wmr*wmc))
	wMatProduct.Mul(wMat, wMatT)

	wMatProdInv := mat64.NewDense(wmr, wmc, make([]float64, wmr*wmc))
	wMatProdInv.Inverse(wMatProduct)

	wMat.Mul(wMatT, wMatProdInv)
	for i, neuronA := range neurons {
		for j, neuronB := range neurons {
			if i != j {
				neuronA.SetConnection(neuronB, wMat.At(i, j))
			}
		}
	}
}

// NewPseudoInverseTrainer returns a newly intantiated PseudoInverseTrainer
func NewPseudoInverseTrainer() *PseudoInverseTrainer {
	return new(PseudoInverseTrainer)
}
