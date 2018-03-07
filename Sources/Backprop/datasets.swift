import CSV
import Foundation

func toydataset() -> [[Double]] {
  return [
    [2.7810836,   2.550537003,  0],
  	[1.465489372, 2.362125076,  0],
  	[3.396561688, 4.400293529,  0],
  	[1.38807019,  1.850220317,  0],
  	[3.06407232,  3.005305973,  0],
  	[7.627531214, 2.759262235,  1],
  	[5.332441248, 2.088626775,  1],
  	[6.922596716, 1.77106367,   1],
  	[8.675418651, -0.242068655, 1],
  	[7.673756466, 3.508563011,  1]
  ]
}

func seedsDataset() -> [[Double]] {
  let stream = InputStream(fileAtPath: "seeds_dataset.csv")
  let csv = try! CSVReader(stream: stream!, delimiter: "\t")
  var seedsData = [[Double]]()
  while let row = csv.next() {
    let fmtRow = row
      .filter { $0 != "" }
      .map { Double($0)! }
    seedsData.append(fmtRow)
  }
  var mins = (0..<7).map { n in 0.0 }
  var maxs = (0..<7).map { n in 0.0 }
  for row in seedsData {
    for (index, cell) in row.enumerated() {
      if index < 7 {
        if cell < mins[index] {
          mins[index] = cell
        }
        if cell > maxs[index] {
          maxs[index] = cell
        }
      }
    }
  }

  let normalized = seedsData.map { row in
    row
      .enumerated()
      .map { (index, cell) in
        return index < 7 ? (cell - mins[index]) / (maxs[index] - mins[index]) : cell
      }
  }

  return normalized
}
