// @ts-ignore
import * as hdf5 from "./thirdparty/jsfive";
import * as tf from "@tensorflow/tfjs";

const MODELNET40_URL = "http://10.6.96.143:2048";

export default class ModelNet {
  categories: string[] = [];
  train = tf.data.generator(
    () => this.generateData("train_files.txt", 32) as any
  );

  static async init() {
    const mn = new ModelNet();
    mn.categories = await mn.loadCategories();
    return mn;
  }

  async *generateData(filename: string, batchSize: number) {
    const paths = await this.getDatasetPaths(filename);
    for (const path of paths.slice(0, 2)) {
      const { data, label } = await this.loadH5(
        [MODELNET40_URL, path].join("/")
      );
      const batchNum = Math.ceil(label.length / batchSize);
      for (let i = 0; i < batchNum; ++i) {
        const start = i * batchSize,
          end = start + batchSize;
        const _d = data.slice(start * 2048 * 3, end * 2048 * 3),
          _l = label.slice(start, end);

        yield tf.tidy(() => {
          const shape = [_d.length / 2048 / 3, 2048, 3, 1];
          const xs = tf.tensor(_d, shape, "float32");
          const depth = this.categories.length;
          const ys = tf.oneHot(tf.tensor1d(_l, "int32"), depth);
          return { xs, ys };
        });
      }
    }
  }

  async loadCategories() {
    const res = await fetch([MODELNET40_URL, "shape_names.txt"].join("/"));
    const text = await res.text();
    const categories = text.split("\n").filter(path => path);
    return categories;
  }

  async getDatasetPaths(filename: string) {
    const res = await fetch([MODELNET40_URL, filename].join("/"));
    const text = await res.text();
    const paths = text.split("\n").filter(path => path);
    return paths;
  }

  async loadH5(url: string) {
    const res = await fetch(url);
    const buffer = await res.arrayBuffer();
    const file = new hdf5.File(buffer, "ply_data_train0.h5");
    const data: number[] = file.get("data").value;
    const label: number[] = file.get("label").value;
    return { data, label };
  }
}
