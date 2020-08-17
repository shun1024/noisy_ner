import os, sys
import json, itertools

memory = 32
gpu = 8 

json_file = sys.argv[1]
is_local = sys.argv[2]

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def get_short_key(name):
    tmp = name.split('_')
    result = []
    for t in tmp:
        result.append(t[0])
    return ''.join(result)


def convert_json_to_command():
    non_special_parameter = ['command', 'output_dir', 'data_dir']
    commands = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        parameters = dict_product(data)
        for parameter in parameters:
            taskname = []
            post_command = []
            for key in parameter:
                if key not in non_special_parameter:
                    short_key = get_short_key(key)
                    taskname.append('%s%s' % (short_key, str(parameter[key])))
                    post_command.append('--%s %s' % (key, str(parameter[key])))

            taskname = '_'.join(taskname)
            post_command = ' '.join(post_command)
            command = '%s --output_dir %s %s' % (parameter['command'], parameter['output_dir'], post_command)

            commands.append('"%s"' % command)
    return commands

lines = convert_json_to_command()
if int(is_local):
    commands = []
    for i in range(len(lines)):
        commands.append('CUDA_VISIBLE_DEVICES={} {} &'.format(i%gpu, lines[i].strip('"')))
    print('\n'.join(commands))

else:
    scripts = ('\n').join(lines)
    print("#!/bin/bash\n#SBATCH --partition=p100\n#SBATCH --mem=%dG\n#SBATCH --gres=gpu:1" % memory)
    print("#SBATCH --array=0-%d%%%d" % (len(lines) - 1, gpu))
    print("#SBATCH --output=./logs/tune-\%A_\%a.log")
    print("list=(\n%s\n)" % scripts)
    print("${list[SLURM_ARRAY_TASK_ID]}")
