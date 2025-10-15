import csv

def gt_to_numeric(gt):
    if gt in ['0/0', '0|0']:
        return 0
    elif gt in ['0/1', '1/0', '0|1', '1|0']:
        return 1
    elif gt in ['1/1', '1|1']:
        return 2
    else:
        return -1  


def vcf_to_csv_numbered_samples(input_path, output_path):
    with open(input_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            if not line[0].startswith("#"):
                header = line
                break
        sample_count = len(header[9:])
        rows = [row for row in reader if len(row) >= 10][:20]

    variant_ids = [f"{row[0]}:{row[1]}" for row in rows]

    matrix = [[] for _ in range(sample_count)]
    for row in rows:
        for i, sample_field in enumerate(row[9:]):
            gt = sample_field.split(':')[0] if sample_field else '.'
            matrix[i].append(gt_to_numeric(gt))

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample'] + variant_ids)
        for idx, gts in enumerate(matrix, start=1):
            writer.writerow([idx] + gts)
