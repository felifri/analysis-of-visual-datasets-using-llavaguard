import json

from util.file_utils import get_file_paths

TIME_DURATION_UNITS = (
    ('w', 60*60*24*7),
    ('d', 60*60*24),
    ('h', 60*60),
    ('m', 60),
    ('s', 1)
)


def human_time_duration(seconds):
    if seconds == 0:
        return 'inf'
    parts = []
    for unit, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(int(seconds), div)
        if amount > 0:
            parts.append('{}{}'.format(amount, unit))
    return ' '.join(parts)

def inspect_download_stats(dir_path: str):
    num_images_attempted = 0
    num_success = 0
    num_failed_to_download = 0
    num_failed_to_resize = 0
    total_duration = 0
    num_attempts_per_hour = 0

    stats_paths, _ = get_file_paths(dir_path, '.json', recursive=False)

    for path in stats_paths:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            num_images_attempted += data.get('count', 0)
            num_success += data.get('successes', 0)
            num_failed_to_download += data.get('failed_to_download', 0)
            num_failed_to_resize += data.get('failed_to_resize', 0)
            total_duration += data.get('duration', 0)
            num_attempts_per_hour = 0 if total_duration == 0 else "{:.2f}".format(60 * 60 / total_duration * num_images_attempted)

    print('{0:<16} {1:<7} {2:<13} {3:<11} {4:<8} {5:<15}'.format(
        'Images_Attempted',
        'Success',
        'Fail_Download',
        'Fail_Resize',
        'Duration',
        'Attempts_/_Hour'
    ))
    print('{0:>16} {1:>7} {2:>13} {3:>11} {4:>8} {5:>15}'.format(
        num_images_attempted,
        num_success,
        num_failed_to_download,
        num_failed_to_resize,
        human_time_duration(total_duration),
        num_attempts_per_hour
    ))

if __name__ == '__main__':
    datasets = [
        ('laion2B-en (256)', '/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/data'),
        ('img2dataset-test_10000 (256)', '/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/test_10000/data_256'),
        ('img2dataset-test_10000 (384)', '/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/test_10000/data_384'),
        ('img2dataset-test_10000 (512)', '/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/test_10000/data_512'),
    ]

    for name, dir_path in datasets:
        print(f"Statistics for '{name}':")
        inspect_download_stats(dir_path)
