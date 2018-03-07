import Foundation

srand48(1)

let dataset = seedsDataset()
let features = dataset[0].count-1
let classes = Array(Set(dataset.map { $0[features] }))

let network = initNet(numIn: features, numHid: 5, numOut: classes.count)
trainNetwork(
  network: network,
  dataset: dataset,
  learnRate: 0.3,
  numEpoch: 500,
  classes: classes)
