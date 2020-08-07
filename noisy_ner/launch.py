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


def main(_):
    runtime = xm.CloudRuntime(
        cpu=4,
        memory=15,
        accelerator=xm.GPU('nvidia-tesla-' + FLAGS.acc_type.lower(), 1),
    )

    args = {}
    args['is_gcp'] = True
    args['dataset'] = 'gs://deid-xcloud/data/i2b2-2014/'
    args['output_dir'] = 'gs://deid-xcloud/results/i2b2-2014/'

    executable = xm.CloudPython(
        name='noisy-ner',
        runtime=runtime,
        project_path=os.path.dirname(os.path.realpath(__file__)),
        module_name='noisy_ner.main',
        args=args,
    )
    """

    input_dir = 'gs://deid-xcloud/data/i2b2-2014/'
    output_dir = 'gs://deid-xcloud/noisy-ner/outputs/i2b2-2014'

    executable = xm.CloudPython(
        name='noisy_ner',
        runtime=runtime,
        project_path=(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        module_name='unused',
        args=args,
        build_steps=(
            xm.steps.default_build_steps('noisy_ner') +
            xm.steps.install_gsutil() +
            [
                f'pip install -r requirements.txt',
                f'mkdir -p inputs/',
                f'mkdir -p teachers/',
                f'mkdir -p outputs/'
            ]),
        exec_cmds=xm.steps.install_gsutil_creds() + [
            f'python3 /root/gsutil/gsutil -m cp -r gs://deid-xcloud/data/i2b2-2014 ',
            'cd stylized_imagenet',
            'python3 preprocess_imagenet.py "$@"',
            f'python3 /root/gsutil/gsutil -m cp -r ',
        ],
    )
    
    parameters = hyper.product([
        hyper.sweep('training_ratio', [0.01, 0.03, 0.1, 1]),
        hyper.sweep('batch_size', [16, 32, 64]),
        hyper.sweep('learning_rate', [1.0, 0.3, 0.1, 0.03])
    ])
    """
    parameters = hyper.product([
        hyper.sweep('batch_size', [16])
    ])

    exploration = xm.ParameterSweep(
        executable, parameters, max_parallel_work_units=100)
    xm.launch(xm.ExperimentDescription('noisy-ner'), exploration)


if __name__ == '__main__':
    app.run(main)
