import os
import shutil

dataset = 'dataset1/'

mine_dataset = {
    "zone 1": {
        "depth": 1,
        "dates": {}
    },
    "zone 2": {
        "depth": 1,
        "dates": {}
    },
    "zone 3": {
        "depth": 1,
        "dates": {}
    },
    "zone 4": {
        "depth": 0,
        "dates": {}
    },
    "zone 5": {
        "depth": 0,
        "dates": {}
    },
    "zone 6": {
        "depth": -1,
        "dates": {}
    },
    "zone 7": {
        "depth": 5,
        "dates": {}
    },
    "zone 8": {
        "depth": 5,
        "dates": {}
    },
    "zone 9": {
        "depth": 10,
        "dates": {}
    },
}

def traverse_zone(date_subdir, type='JPG'):
    for f in os.listdir(dataset + date_subdir + f'/{type}'):
        name_split = f.lower().split(' ', 2)

        try:
            zone_name = name_split[0] + ' ' + name_split[1]
        except IndexError:
            continue

        try:
            mine_dataset[zone_name]['dates'][date_subdir][type.lower()] = []
        except KeyError:
            mine_dataset[zone_name]['dates'][date_subdir] = {}
            mine_dataset[zone_name]['dates'][date_subdir][type.lower()] = []

        for img in os.listdir(dataset + date_subdir + f'/{type}/' + f):
            img_path = dataset + date_subdir + f'/{type}/' + f + '/' + img
            mine_dataset[zone_name]['dates'][date_subdir][type.lower()].append(img_path)

def traverse_by_date(date_subdir):
    traverse_zone(date_subdir, 'JPG')
    traverse_zone(date_subdir, 'R_JPG')

def copy_to_output(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for zone in mine_dataset:
        os.mkdir(output_dir + zone)
        for date in mine_dataset[zone]['dates']:
            os.mkdir(output_dir + zone + '/' + date)
            os.mkdir(output_dir + zone + '/' + date + '/jpg')
            os.mkdir(output_dir + zone + '/' + date + '/r_jpg')
            for img in mine_dataset[zone]['dates'][date]['jpg']:
                img_name = img.split('/')[-1]
                shutil.copy(img, output_dir + '/' + zone + '/' + date + '/jpg/' + img_name)

            for img in mine_dataset[zone]['dates'][date]['r_jpg']:
                img_name = img.split('/')[-1]
                shutil.copy(img, output_dir + '/' + zone + '/' + date + '/r_jpg/' + img_name)


if __name__ == '__main__':
    for date_subdir in os.listdir(dataset):
        traverse_by_date(date_subdir)


    copy_to_output('new_output/')
    print(mine_dataset)