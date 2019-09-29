// @ts-ignore
import * as hdf5 from "./thirdparty/jsfive";
import * as tf from "@tensorflow/tfjs";

const MODELNET40_URL = "http://10.6.96.143:2048";

export async function loadCategories() {
    return new Promise<string[]>((resolve, reject) => {
        fetch([MODELNET40_URL, "shape_names.txt"].join("/"))
            .then(async res => {
                const text = await res.text();
                const categories = text.split("\n").filter(path => path);
                resolve(categories);
            })
            .catch(reason => reject(reason));
    });
}

function getDatasetPaths(filename: string) {
    return new Promise<string[]>((resolve, reject) => {
        fetch([MODELNET40_URL, filename].join("/"))
            .then(async res => {
                const text = await res.text();
                const paths = text.split("\n").filter(path => path);
                resolve(paths);
            })
            .catch(reason => reject(reason));
    });
}

function loadH5(url: string) {
    return new Promise<{ data: number[]; label: number[] }>((resolve, reject) => {
        fetch(url)
            .then(res => res.arrayBuffer())
            .then(buffer => {
                const file = new hdf5.File(buffer, "ply_data_train0.h5");
                const data = file.get("data").value as number[];
                const label = file.get("label").value as number[];
                resolve({ data, label });
            })
            .catch(reason => reject(reason));
    });
}

async function* generateData(filename: string, batchSize: number) {
    const paths = await getDatasetPaths(filename);
    for (const path of paths) {
        const { data: d, label: l } = await loadH5(
            [MODELNET40_URL, path].join("/")
        );
        const batchNum = Math.ceil(l.length / batchSize);
        for (let i = 0; i < batchNum; ++i) {
            const start = i * batchSize,
                end = start + batchSize;
            const _d = d.slice(start * 2048 * 3, end * 2048 * 3),
                _l = l.slice(start, end);

            yield tf.tidy(() => {
                const xs = tf.tensor(_d, [_d.length / 2048 / 3, 2048, 3, 1], "float32");
                const ys = tf.oneHot(tf.tensor1d(_l, "int32"), 40);
                return { xs, ys };
            })
        }
    }
}

export const trainDataset = tf.data.generator(
    () => generateData("train_files.txt", 32) as any
);