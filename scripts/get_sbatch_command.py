import os, sys
import json, itertools

memory = 32
gpu = 16

json_file = sys.argv[1]
is_local = sys.argv[2]


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def get_short_name(exp_name):
    result = []
    for exp in exp_name:
        result.append("%s,%s" % (exp[0][:2], exp[1]))
    return '_'.join(result)


def convert_json_to_command():
    non_special_parameter = ['output_dir', 'dataset', 'gpu_type', 'num_gpu', 'exp']
    commands = []
    with open(json_file, 'r') as f:
        exps = json.load(f)
        for exp in exps:
            data = exps[exp]
            base_command = 'python -m noisy_ner.main --dataset %s --output_dir %s' % (
            data['dataset'], data['output_dir'])

            for key in list(data):
                if key in non_special_parameter:
                    del data[key]

            parameters = dict_product(data)
            for parameter in parameters:
                post_command = []
                exp_name = []
                for key in parameter:
                    if key not in non_special_parameter:
                        post_command.append('--%s %s' % (key, str(parameter[key])))
                        exp_name.append([key, str(parameter[key])])

                post_command.append('--exp %s' % get_short_name(exp_name))
                post_command = ' '.join(post_command)
                command = '%s %s' % (base_command, post_command)
                commands.append('"%s"' % command)
    return commands


lines = convert_json_to_command()
if int(is_local):
    commands = []
    for i in range(len(lines)):
        commands.append('CUDA_VISIBLE_DEVICES={} {} &'.format(i % gpu, lines[i].strip('"')))
    print('\n'.join(commands))

else:
    scripts = ('\n').join(lines)
    print("#!/bin/bash\n#SBATCH --partition=p100\n#SBATCH --mem=%dG\n#SBATCH --gres=gpu:1" % memory)
    print("#SBATCH --array=0-%d%%%d" % (len(lines) - 1, gpu))
    print("#SBATCH --output=./logs/tune-\%A_\%a.log")
    print("list=(\n%s\n)" % scripts)
    print("${list[SLURM_ARRAY_TASK_ID]}")
