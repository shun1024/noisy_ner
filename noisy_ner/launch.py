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

    executable = xm.CloudPython(
                    name='stylized-imagenet',
                            runtime=runtime,
                                    project_path=(
                                                    # If you are running this script from your workstation against
                                                                # google3 HEAD, you can also use this string literal:
                                                                            # '/google/src/head/depot/google3/learning/brain/frameworks/xcloud/examples/mnist_torch_gpu'
                                                                                        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                                                                                                module_name='unused',
                                                                                                        args=args,
                                                                                                                #base_image='nvidia/cuda:10.2-cudnn7-devel',
                                                                                                                        base_image='nvidia/cuda:9.0-cudnn7-devel',
                                                                                                                                #base_image='bethgelab/deeplearning:cuda9.0-cudnn7',
                                                                                                                                        build_steps=(
                                                                                                                                                    xm.steps.default_build_steps('stylized_imagenet') +
                                                                                                                                                                xm.steps.install_gsutil() +
                                                                                                                                                                            [
                                                                                                                                                                                             #'apt install -y git',
                                                                                                                                                                                                          #'pip3 install torch numpy',
                                                                                                                                                                                                                       #'git clone https://github.com/NVIDIA/apex',
                                                                                                                                                                                                                                    #'pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex/',
                                                                                                                                                                                                                                                 f'mkdir -p {imagenet_path}/',
                                                                                                                                                                                                                                                              f'mkdir -p {adain_path}/',
                                                                                                                                                                                                                                                                           f'mkdir -p {stylized_image_path}/',
                                                                                                                                                                                                                                                                                        f'mkdir -p {stylized_image_path}/train',
                                                                                                                                                                                                                                                                                                    ]),
                                                                                                                                                                                    exec_cmds=xm.steps.install_gsutil_creds() + [
                                                                                                                                                                                                        f'python3 /root/gsutil/gsutil -m cp -r {FLAGS.adain_preprocessed_paintings_dir}/* {adain_path}/',
                                                                                                                                                                                                                    f'python3 /root/gsutil/gsutil -m cp -r {FLAGS.imagenet_path}/train {imagenet_path}/',
                                                                                                                                                                                                                                f'python3 /root/gsutil/gsutil cp {FLAGS.adain_decoder_path} {adain_decoder_path}',
                                                                                                                                                                                                                                            f'python3 /root/gsutil/gsutil cp {FLAGS.adain_vgg_path} {adain_vgg_path}',
                                                                                                                                                                                                                                                        'cd stylized_imagenet',
                                                                                                                                                                                                                                                                    'python3 preprocess_imagenet.py "$@"',
                                                                                                                                                                                                                                                                                f'python3 /root/gsutil/gsutil -m cp -r {stylized_image_path}/* {FLAGS.gcs_dump_path}/',
                                                                                                                                                                                                                                                                                        ],
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
