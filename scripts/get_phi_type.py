import glob, os


def get_phi(indir):
    phi = []
    files = glob.glob(os.path.join(indir, '*.txt'))
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                tmp = line.strip().split()
                if len(tmp) > 0:
                    phi.append(tmp[-1])

    phi = set(phi)
    phi = list(phi)
    phi = sorted(phi)
    print(phi)


def change_phi_name(file, rules):
    results = []
    with open(file, 'r') as f:
        for line in f:
            tmp = line.strip().split()
            if len(tmp) == 0:
                results.append('')
            else:
                if tmp[-1] in rules:
                    tmp[-1] = rules[tmp[-1]]
                results.append(' '.join(tmp))

    with open(file, 'w') as f:
        for line in results:
            f.write('%s\n' % line)


if __name__ == '__main__':
    dirs = glob.glob('./data/*')
    for indir in dirs:
        print(indir)
        get_phi(indir)
        print()

    rules = {
        'B-ID': 'B-IDNUM',
        'I-ID': 'I-IDNUM',
        'B-LOCATION': 'B-LOCATION_OTHER',
        'I-LOCATION': 'I-LOCATION_OTHER',
    }
    for file in glob.glob('./data/i2b2_2006/*'):
        change_phi_name(file, rules)
