# Lint as: python3
"""Launcher for running MNIST on Torch with GPUs."""
import os, json, time

from absl import app
from absl import flags

import xcloud as xm
from xcloud import hyper

FLAGS = flags.FLAGS
flags.DEFINE_string('json', None, 'json store experiment hyper-parameters')


def main(_):
    args = json.load(open(FLAGS.json, 'r'))

    runtime = xm.CloudRuntime(
        cpu=4,
        memory=32,
        accelerator=xm.GPU('nvidia-tesla-' + args['gpu_type'].lower(), args['num_gpus']),
    )

    executable = xm.CloudPython(
        name='noisy-ner-{}'.format(FLAGS.exp),
        runtime=runtime,
        project_path=os.path.dirname(os.path.realpath(__file__)),
        module_name='noisy_ner.main',
    )

    remove_args = ['gpu_type', 'num_gpus']
    args['output_dir'] = os.path.join(args['output_dir'], str(int(time.time())))
    parameters = []
    for key in args:
        if key not in remove_args:
            parameters.append(hyper.sweep(key, args[key]))
    parameters = hyper.product(parameters)

    exploration = xm.ParameterSweep(
        executable, parameters, max_parallel_work_units=200)
    xm.launch(xm.ExperimentDescription('noisy-ner-{}'.format(FLAGS.exp)), exploration)


if __name__ == '__main__':
    app.run(main)
