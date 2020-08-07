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
    'gcs_path', None, 'A GCS directory within a bucket to store output '
    '(in gs://bucket/directory format).')
flags.DEFINE_string('acc_type', 'v100', 'Accelerator type`).')


def main(_):
  runtime = xm.CloudRuntime(
      cpu=4,
      memory=15,
      accelerator=xm.GPU('nvidia-tesla-' + FLAGS.acc_type.lower(), 2),
  )

  if not FLAGS.gcs_path:
    print('--gcs_path was not passed. ' 'The output model will not be saved.')
  elif'gs://' not in FLAGS.gcs_path:
    suggestion = os.path.join(
        'gs://', 'xcloud_public_bucket', getpass.getuser(),
        'mnist-tf-gpu-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    raise app.UsageError(
        '--gcs_path not in gs://bucket/directory format. Suggestion: ' +
        f'--gcs_path={suggestion}')

  args = {}
  if FLAGS.gcs_path:
    args['gcs_path'] = FLAGS.gcs_path


    # Option 2 This will build a docker image for the user.
  executable = xm.CloudPython(
        name='mnist-torch-gpu',
        runtime=runtime,
        project_path=os.path.dirname(os.path.realpath(__file__)),
        module_name='mnist_torch_gpu.main',
        args=args,
  )

  parameters = hyper.product([
      hyper.sweep('batch_size', [8*2**k for k in range(2)]),
      hyper.zipit([
          hyper.loguniform('learning_rate', hyper.interval(0.01, 0.1)),
      ], length=2),
  ])

  exploration = xm.ParameterSweep(
      executable, parameters, max_parallel_work_units=2)
  xm.launch(xm.ExperimentDescription('mnist-torch-gpu'), exploration)

  if FLAGS.gcs_path:
    no_prefix = FLAGS.gcs_path[len('gs://'):]
    print()
    print('When your job completes, you will see artifacts in ' +
          termcolor.colored(
              f'https://pantheon.corp.google.com/storage/browser/{no_prefix}',
              color='blue'))


if __name__ == '__main__':
  app.run(main)
