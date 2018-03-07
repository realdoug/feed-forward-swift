import Foundation

struct Neuron {
  var weights: [Double]
  var output: Double
  var delta: Double

  init(
    weights: [Double] = [Double](),
    output: Double = 0.0,
    delta: Double = 0.0
  ){
    self.weights = weights
    self.output = output
    self.delta = delta
  }
}

func newLayer(numNeurons: Int, numWeights: Int) -> [Neuron] {
  return (0..<numNeurons).map { i in
    Neuron(weights: (0..<numWeights+1).map { i in drand48() })
  }
}

func activate(_ weights: [Double], _ inputs: [Double]) -> Double {
  let upto = weights.count-1
  var activation = weights[upto]
  for i in 0..<upto {
    activation = activation + (weights[i] * inputs[i])
  }
  return activation
}

func sigmoid(_ activation: Double) -> Double {
  return 1.0 / (1.0 + pow(M_E, -activation))
}

func sigmoidDeriv(_ value: Double) -> Double {
  return value * (1.0 - value)
}

func loss(expected: Double, predicted: Double) -> Double {
  return expected-predicted
}

func fwd(network: [[Neuron]], row: [Double]) -> [[Neuron]] {
  var newNet = [[Neuron]]()

  var nextInputs = row
  for layer in network {
    let newLayer = layer.map { neuron -> Neuron in
      let activation = activate(neuron.weights, nextInputs)
      return Neuron(
        weights: neuron.weights,
        output: sigmoid(activation)
      )
    }
    newNet.append(newLayer)
    nextInputs = newLayer.map { $0.output }
  }

  return newNet
}

func backpropError(network: [[Neuron]], errors: [Double]) -> [[Neuron]] {
  var newNet = [[Neuron]]()

  let lastLayerIndex = network.count-1
  for i in stride(from: lastLayerIndex, through: 0, by: -1){
    let oldLayer = network[i]
    let newLayer = oldLayer
      .enumerated()
      .map() { (index, neuron) -> Neuron in
        var err: Double
        if i == lastLayerIndex {
          err = errors[index]
        } else {
          err = newNet[0].reduce(0.0, {
            $0 + $1.weights[index] * $1.delta
          })
        }

        return Neuron(
          weights: neuron.weights,
          output: neuron.output,
          delta: err * sigmoidDeriv(neuron.output)
        )
      }
    newNet.insert(newLayer, at: 0)
  }

  return newNet
}

func updateWeights(network: [[Neuron]], row: [Double], learnRate: Double) -> [[Neuron]]{
  let newNet = network
    .enumerated()
    .map { (index, layer) -> [Neuron] in
      var inputs = Array(row.dropLast())
      if index != 0 {
        inputs = network[index-1].map { $0.output }
      }
      let newLayer = layer.map { neuron -> Neuron in
        var weights = inputs.enumerated().map { (i, inp) in
          neuron.weights[i] + (learnRate * neuron.delta * inp)
        }
        weights.append(neuron.weights.last! + (learnRate * neuron.delta))

        return Neuron(
          weights: weights,
          output: neuron.output,
          delta: neuron.delta
        )
      }
      return newLayer
    }

  return newNet
}

func trainNetwork(
  network: [[Neuron]],
  dataset: [[Double]],
  learnRate: Double,
  numEpoch: Int,
  classes: [Double]
) -> [[Neuron]] {
  var newNet = network

  for epoch in 0..<numEpoch {
    var epochError = 0.0
    for row in dataset {
      let fwdNet = fwd(network: newNet, row: row)
      let predictions = fwdNet[fwdNet.count-1].map { $0.output }
      let oneHotExpected = classes.map { $0 == row.last ? 1.0 : 0.0 }

      var errors = [Double]()
      for (ex, pred) in zip(oneHotExpected, predictions) {
        errors.append( loss(expected: ex, predicted: pred) )
      }
      epochError += errors.reduce(0.0, { $0 + pow($1, 2) })
      let bakNet = backpropError(network: fwdNet, errors: errors)
      newNet = updateWeights(network: bakNet, row: row, learnRate: learnRate)
    }
    print("epoch: \(epoch), lr: \(learnRate), error: \(epochError)")
  }

  return newNet
}

func initNet(numIn: Int, numHid: Int, numOut: Int) -> [[Neuron]] {
  var network = [[Neuron]]()

  let hiddenLayer = newLayer(numNeurons: numHid, numWeights: numIn)
  network.append(hiddenLayer)

  let outputLayer = newLayer(numNeurons: numOut, numWeights: numHid)
  network.append(outputLayer)

  return network
}
