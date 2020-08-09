# Lint as: python3
"""Launcher for running MNIST on Torch with GPUs."""
import datetime
import getpass
import os

from absl import app
from absl import flags
import termcolor

import xcloud as xm
from xcloud import hyper

FLAGS = flags.FLAGS
flags.DEFINE_string('acc_type', 'v100', 'Accelerator type`).')
flags.DEFINE_string('exp', 'baseline', 'exp name')


def main(_):
    runtime = xm.CloudRuntime(
        cpu=8,
        memory=32,
        accelerator=xm.GPU('nvidia-tesla-' + FLAGS.acc_type.lower(), 1),
    )

    args = {}
    args['is_gcp'] = True
    args['dataset'] = 'deid-xcloud/data/i2b2_2014'
    args['output_dir'] = 'deid-xcloud/results/i2b2_2014/{}'.format(FLAGS.exp)

    executable = xm.CloudPython(
        name='noisy-ner-{}'.format(FLAGS.exp),
        runtime=runtime,
        project_path=os.path.dirname(os.path.realpath(__file__)),
        module_name='noisy_ner.main',
        args=args,
    )

    ratio_to_epoch = {'1.0': 50, '0.1': 50, '0.03': 100, '0.01': 100}
    parameters_list = []
    for ratio in ratio_to_epoch:
        parameters = hyper.product([
            hyper.fixed('training_ratio', float(ratio)),
            hyper.fixed('epoch', int(ratio_to_epoch[ratio])),
            hyper.sweep('learning_rate', [0.3, 0.1]),
            hyper.sweep('dropout', [0.1, 0.2]),
            hyper.sweep('hidden_size', [128, 256])
        ])

        parameters_list.append(parameters)
    parameters = hyper.chainit(parameters_list)
    
    """
    parameters = hyper.product([
        hyper.sweep('batch_size', [16]),
        hyper.sweep('epoch', [1])
    ])
    """

    exploration = xm.ParameterSweep(
        executable, parameters, max_parallel_work_units=200)
    xm.launch(xm.ExperimentDescription('noisy-ner-{}'.format(FLAGS.exp)), exploration)


if __name__ == '__main__':
    app.run(main)
