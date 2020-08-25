# Lint as: python3
"""Launcher for running MNIST on Torch with GPUs."""
import os, json, time

from absl import app
from absl import flags

import xcloud as xm
from xcloud import hyper

FLAGS = flags.FLAGS
flags.DEFINE_string('json', None, 'json store experiment hyper-parameters')


def get_exp_name(parameter):
    non_add_name = ['teacher_dir']
    exp_name = []
    for key in parameter:
        if key in non_add_name:
            continue
        short_name = key[0] + key[-1]
        exp_name.append('{}={}'.format(short_name, parameter[key]))
    return ','.join(exp_name)


def add_exp_name(parameters):
    for parameter in parameters:
        exp_name = get_exp_name(parameter)
        parameter['exp'] = exp_name
    return parameters


def check_compatibility(parameters):
    result = []
    for parameter in parameters:
        if 'teacher_dir' in parameter:
            teacher_to = parameter['teacher_dir'].split('=')[1].split(',')[0]
            if float(teacher_to) != parameter['training_ratio']:
                continue
        result.append(parameter)
    return result


def main(_):
    json_args = json.load(open(FLAGS.json, 'r'))

    runtime = xm.CloudRuntime(
        cpu=4,
        memory=32,
        accelerator=xm.GPU('nvidia-tesla-' + json_args['gpu_type'].lower(), json_args['num_gpu']),
    )

    executable = xm.CloudPython(
        name='noisy-ner-{}'.format(json_args['exp']),
        runtime=runtime,
        project_path=os.path.dirname(os.path.realpath(__file__)),
        module_name='noisy_ner.main',
        args={
            'is_gcp': True,
            'dataset': json_args['dataset'],
            'output_dir': json_args['output_dir'],
        }
    )

    remove_json_args = ['gpu_type', 'num_gpu', 'exp', 'dataset', 'output_dir']
    json_args['output_dir'] = os.path.join(json_args['output_dir'], str(int(time.time())))
    parameters = []
    for key in json_args:
        if key not in remove_json_args:
            parameters.append(hyper.sweep(key, json_args[key]))
    parameters = hyper.product(parameters)
    parameters = [p for p in parameters]
    parameters = add_exp_name(parameters)
    parameters = check_compatibility(parameters)
    exploration = xm.ParameterSweep(
        executable, parameters, max_parallel_work_units=200)
    xm.launch(xm.ExperimentDescription('{}'.format(json_args['exp'])), exploration)


if __name__ == '__main__':
    app.run(main)
