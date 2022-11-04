import os
from pkg_resources import resource_filename
import tensorflow as tf
import numpy as np

try:
    tf.enable_eager_execution()
except AttributeError:
    pass

class CISpliceAI():
    def __init__(self, model_path=None):
        model_path = resource_filename(__name__, os.path.join('data', 'CI-SpliceAI.pb')) if model_path is None else model_path
        self._model = self._load_model(model_path)

    def predict(self, batch):
        '''Predicts on a batch of data. Can be of different sequence length, accepts lists, numpy arrays and tf tensors as input.'''
        # Batch might have different sequence lengths. Pad them to the longest.
        longest_sequence = max([variant.shape[0] for variant in batch])

        paddings = [longest_sequence - variant.shape[0] for variant in batch]

        padded_batch = tf.stack([
            tf.concat([variant, tf.zeros((padding, 4), dtype=tf.float32)], axis=0)
            for variant, padding in zip(batch, paddings)
        ])

        # Run model on the padded batch
        preds = self._model(padded_batch)
        if type(preds) == list:
            preds = preds[0]

        # Return list of unpadded preds
        return [
            preds[i][:preds[i].shape[0]-paddings[i]].numpy()
            for i in range(preds.shape[0])
        ]
    def _load_model(self, model_path):

        with tf.io.gfile.GFile(os.path.join(model_path), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = self._wrap_frozen_graph(graph_def=graph_def,
                                        inputs=[f'{graph_def.node[0].name}:0'],
                                        outputs=[f'{graph_def.node[-1].name}:0']
        )

        return frozen_func

    def _wrap_frozen_graph(self, graph_def, inputs, outputs):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs)
        )


if __name__ == '__main__':
    model = CISpliceAI()
    preds = model.predict([
        np.zeros((15000, 4), dtype=bool),
        np.zeros((15001, 4), dtype=bool),
    ])
    assert preds[0].shape == (5000, 3)
    assert preds[1].shape == (5001, 3)
