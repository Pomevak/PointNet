import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { loadCategories, trainDataset } from "./data";

async function main() {
  if (await tf.setBackend("webgl")) {
    const pointNet = new PointNet();
    await pointNet.init();
    pointNet.train();
  }
}

class PointNet {
  model?: tf.Sequential;

  async init() {
    const categories = await loadCategories();
    this.model = tf.sequential({
      layers: [
        // mlp1
        tf.layers.conv2d({
          filters: 64,
          kernelSize: [1, 3],
          inputShape: [2048, 3, 1]
        }),
        tf.layers.batchNormalization(),
        tf.layers.activation({ activation: "relu" }),
        tf.layers.conv2d({ filters: 64, kernelSize: 1 }),
        tf.layers.batchNormalization(),
        tf.layers.activation({ activation: "relu" }),
        // mlp2
        tf.layers.conv2d({ filters: 64, kernelSize: 1 }),
        tf.layers.batchNormalization(),
        tf.layers.activation({ activation: "relu" }),
        tf.layers.conv2d({ filters: 128, kernelSize: 1 }),
        tf.layers.batchNormalization(),
        tf.layers.activation({ activation: "relu" }),
        tf.layers.conv2d({ filters: 1024, kernelSize: 1 }),
        tf.layers.batchNormalization(),
        tf.layers.activation({ activation: "relu" }),
        // maxpooling
        tf.layers.maxPool2d({ poolSize: [2048, 1], strides: 1 }),
        tf.layers.flatten(),
        // mlp3
        tf.layers.dense({ units: 512 }),
        tf.layers.batchNormalization(),
        tf.layers.activation({ activation: "relu" }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({ units: 256 }),
        tf.layers.batchNormalization(),
        tf.layers.activation({ activation: "relu" }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({ units: categories.length }),
        tf.layers.batchNormalization(),
        tf.layers.activation({ activation: "softmax" })
      ]
    });
    this.model.summary();

    this.model.compile({
      optimizer: tf.train.adam(),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });
  }

  async train() {
    if (!this.model) return;
    await tfvis.show.modelSummary({ name: "Model Architecture" }, this.model);

    const metrics = ["loss", "val_loss", "acc", "val_acc"];
    const fitCallbacks = tfvis.show.fitCallbacks(
      { name: "Model Training" },
      metrics
    );

    return this.model.fitDataset(trainDataset, {
      epochs: 1,
      callbacks: [
        fitCallbacks,
        {
          async onBatchEnd() {
            const info = tf.memory();
            console.log(`numTensors: ${info.numTensors}`);
          }
        }
      ]
    });
  }
}

main();
