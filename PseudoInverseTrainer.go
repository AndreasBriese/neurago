// Package neurago is a little library providing tools to
// implement artificial neural networks.
package neurago

import (
	"local/utils/math"
	"log"

	"github.com/gonum/matrix/mat64"
)

//var pinv_lRate = 0.1

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

func computeWeight(net ANN,
	nA, nB Neuron,
	i, j int,
	patterns [][]float64,
	correlInv *mat64.Dense) {

	var weight float64
	nbOfPatterns := len(patterns)
	nbOfNeurons := float64(len(net.Neurons()))
	//fmt.Printf("correlInv:\n%v\n\n", mat64.Formatted(correlInv))
	for v := 0; v < nbOfPatterns; v++ {
		for u := 0; u < nbOfPatterns; u++ {
			//fmt.Printf("pat[%d][%d](%f) * %f * pat[%d][%d](%f) ", v, i, patterns[v][i], correlInv.At(v, u), u, j, patterns[u][j])
			weight += patterns[v][i] * correlInv.At(v, u) * patterns[u][j]
			//fmt.Println("weight: ", weight)
		}
	}
	weight *= (1.0 / nbOfNeurons)
	nA.SetConnection(nB, math.ToFixed(weight, 11))
}

func (t PseudoInverseTrainer) Train(net ANN, patterns [][]float64) {
	var sum float64
	nbOfPatterns := len(patterns)
	nbOfNeurons := len(net.Neurons())
	correl := mat64.NewDense(nbOfPatterns, nbOfPatterns, make([]float64, nbOfPatterns*nbOfPatterns))
	correlInv := mat64.NewDense(nbOfPatterns, nbOfPatterns, make([]float64, nbOfPatterns*nbOfPatterns))

	for v := 0; v < nbOfPatterns; v++ {
		for u := 0; u < nbOfPatterns; u++ {
			sum = 0
			for k := 0; k < nbOfNeurons; k++ {
				sum += patterns[v][k] * patterns[u][k]
			}
			correl.Set(v, u, sum/float64(nbOfNeurons))
		}
	}
	if err := correlInv.Inverse(correl); err != nil {
		log.Panicln("PseudoInverseTrainer#Train, Mat64.Dense#Inverse error: ", err)
	}
	for i, nA := range net.Neurons() {
		for j, nB := range net.Neurons() {
			if i != j {
				computeWeight(net, nA, nB, i, j, patterns, correlInv)
			}
		}
	}
}

// func teachPattern(net ANN, pattern []float64) {
// 	var sum float64
// 	neurons := net.Neurons()
// 	patternLen := len(pattern)
// 	weights := buildWeightMatrix(net)

// 	for i := 0; i < patternLen; i++ {
// 		for j := 0; j < patternLen; j++ {
// 			if i != j {
// 				sum = 0
// 				for k := 0; k < patternLen; k++ {
// 					sum += weights.At(i, k) * pattern[k]
// 				}
// 				weights.Set(i, j, weights.At(i, j)+pinv_lRate*(pattern[i]-sum)*pattern[j])
// 			}
// 		}
// 	}
// 	for i, neuronA := range neurons {
// 		for j, neuronB := range neurons {
// 			if i != j {
// 				neuronA.SetConnection(neuronB, weights.At(i, j))
// 			}
// 		}
// 	}
// }

// // See Trainer#Train
// func (t PseudoInverseTrainer) Train(net ANN, patterns [][]float64) {
// 	for _, pattern := range patterns {
// 		teachPattern(net, pattern)
// 	}
// }

// See Trainer#Train
// func (t PseudoInverseTrainer) Train(net ANN, patterns [][]float64) {
// 	neurons := net.Neurons()

// 	hebbTrainer := NewHebbTrainer()
// 	hebbTrainer.Train(net, patterns)

// 	wMat := buildWeightMatrix(net)
// 	wmr, wmc := wMat.Dims()

// 	wMatT := wMat.T()

// 	wMatProduct := mat64.NewDense(wmr, wmc, make([]float64, wmr*wmc))
// 	wMatProduct.Mul(wMat, wMatT)

// 	wMatProdInv := mat64.NewDense(wmr, wmc, make([]float64, wmr*wmc))
// 	wMatProdInv.Inverse(wMatProduct)

// 	wMat.Mul(wMatT, wMatProdInv)
// 	wMat.Apply(func(i, j int, v float64) float64 {
// 		return math.ToFixed(v, 10)
// 	}, wMat)
// 	for i, neuronA := range neurons {
// 		for j, neuronB := range neurons {
// 			if i != j {
// 				neuronA.SetConnection(neuronB, wMat.At(i, j))
// 			}
// 		}
// 	}
// }

// NewPseudoInverseTrainer returns a newly intantiated PseudoInverseTrainer
func NewPseudoInverseTrainer() *PseudoInverseTrainer {
	return new(PseudoInverseTrainer)
}
