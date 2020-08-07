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
flags.DEFINE_string(
    flags.DEFINE_string('acc_type', 'v100', 'Accelerator type`).')


def main(_):
    runtime = xm.CloudRuntime(
        cpu=4,
        memory=15,
        accelerator=xm.GPU('nvidia-tesla-' + FLAGS.acc_type.lower(), 2),
    )

    args = {}
    args['is_gcp'] = True
    args['dataset'] = 'gs://deid-xcloud/data/i2b2-2014/'
    args['output_dir'] = 'gs://deid-xcloud/results/i2b2-2014/'

    if FLAGS.gcs_path:
        args['gcs_path'] = FLAGS.gcs_path

        # Option 2 This will build a docker image for the user.
    executable = xm.CloudPython(
        name='noisy-ner',
        runtime=runtime,
        project_path=os.path.dirname(os.path.realpath(__file__)),
        module_name='noisy_ner.main',
        args=args,
    )

    parameters = hyper.product([
        hyper.sweep('batch_size', [16])
    ])
    """
    parameters = hyper.product([
        hyper.sweep('training_ratio', [0.01, 0.03, 0.1, 1]),
        hyper.sweep('batch_size', [16, 32, 64]),
        hyper.sweep('learning_rate', [1.0, 0.3, 0.1, 0.03])
    ])
    """

    exploration = xm.ParameterSweep(
        executable, parameters, max_parallel_work_units=100)
    xm.launch(xm.ExperimentDescription('noisy-ner'), exploration)


if __name__ == '__main__':
    app.run(main)
