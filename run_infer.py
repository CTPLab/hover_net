"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name,
                              and expected overlaid color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC,
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size. [default: 128]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""

import torch
from pathlib import Path
from random import shuffle
import xml.etree.ElementTree as ET
from xml.dom import minidom
import xmltodict
from docopt import docopt
import copy
import os
import csv
import logging
tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map]

options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
"""

wsi_cli = """
Arguments for processing wsi

usage:
    wsi (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]

options:
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks.
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
"""


def header_lookup(headers):
    """The header lookup table. Assign the index for each candidate as follow,

    var_id[patient id] = 0
    var_id[survival rate] = 1

    Args:
        headers: the name list of candidate causal variables,
                    outcome, patien id, etc.
    """

    var_id = dict()

    for idx, head in enumerate(headers):
        var_id[head] = idx

    return var_id


def proc_summ(cell_info,
              summ_file,
              tma_num=2):
    summ_dict = dict()
    with open(str(summ_file), 'r') as sfile:
        summ_reader = csv.reader(sfile, delimiter=',')

        for idx, tma_summ in enumerate(summ_reader):
            if idx == 0:
                summ_id = header_lookup(tma_summ)
                continue

            pat_id = tma_summ[summ_id['Spot Id 1']]
            pat_valid = tma_summ[summ_id['Spot Valid']]
            if not (pat_valid == 'true' and pat_id.isdigit()):
                print('invalid tma of the patient {}, ignore.'.
                      format(pat_id))
                continue

            col = 'C' + tma_summ[summ_id['TMA Column']].zfill(2)
            row = 'R' + tma_summ[summ_id['TMA Row']].zfill(2)
            left = tma_summ[summ_id['Left(pixels)']]
            top = tma_summ[summ_id['Top (pixels)']]
            wid = tma_summ[summ_id['Width (pixels)']]
            hei = tma_summ[summ_id['Height (pixels)']]

            summ_dict[col+row] = [int(left), int(top),
                                  int(wid), int(hei), False]

        summ_keys = list(summ_dict.keys())
        shuffle(summ_keys)
        for num in range(tma_num):
            summ_dict[summ_keys[num]][-1] = True

    annotations = ET.Element('Annotations')
    # create the layer info
    anno = ET.SubElement(annotations, 'Annotation',
                         LineColor='65535',
                         Name="Layer 1",
                         Visible='True')
    regions = ET.SubElement(anno, 'Regions')
    for num in range(tma_num):
        region = ET.SubElement(regions, 'Region',
                               Type='Ellipse',
                               HasEndcaps='0',
                               NegativeROA='0')
        vertices = ET.SubElement(region, 'Vertices')
        left, top, hei, wid, _ = summ_dict[summ_keys[num]]

        ET.SubElement(vertices, 'V',
                      X=str(left),
                      Y=str(top))
        ET.SubElement(vertices, 'V',
                      X=str(left + wid),
                      Y=str(top + hei))
        ET.SubElement(region, 'Comments')
    # create the skeleton for cell type
    ann_list = [None] * len(cell_info.keys())
    for cell_nm, cell_val in cell_info.items():
        anno = ET.SubElement(annotations, 'Annotation',
                             LineColor=cell_val[2],
                             Name=cell_nm,
                             Visible='True')
        ann_list[cell_val[0] - 1] = ET.SubElement(anno, 'Regions')

    return summ_dict, annotations


torch.multiprocessing.set_sharing_strategy('file_system')
# -------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    cell_info = {
        'Inflammation': [1, [0, 0, 255], '16711680'],  # blue, 16711680
        'Epithelium': [2, [255, 0, 0], '255'],  # red, 255
        'Miscellaneous': [3, [255, 0, 255], '16711935'],  # magenta, 16711935
        'Stroma': [4, [0, 128, 0], '32768'],  # dark green, 32768
        'Mucin': [5, [0, 255, 255], '16776960'],  # cyan, 16776960
    }

    sub_cli_dict = {'tile': tile_cli, 'wsi': wsi_cli}
    args = docopt(__doc__, help=False, options_first=True,
                  version='HoVer-Net Pytorch Inference v1.0')
    sub_cmd = args.pop('<command>')
    sub_cmd_args = args.pop('<args>')

    if args['--help'] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict:
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if args['--help'] or sub_cmd is None:
        print(__doc__)
        exit()

    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)

    args.pop('--version')
    gpu_list = args.pop('--gpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    args = {k.replace('--', ''): v for k, v in args.items()}
    sub_args = {k.replace('--', ''): v for k, v in sub_args.items()}
    if args['model_path'] == None:
        raise Exception(
            'A model path must be supplied as an argument with --model_path.')

    nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    method_args = {
        'method': {
            'model_args': {
                'nr_types': nr_types,
                'mode': args['model_mode'],
            },
            'model_path': args['model_path'],
        },
        'type_info_path': None if args['type_info_path'] == ''
        else args['type_info_path'],
    }

    # ***
    run_args = {
        'batch_size': int(args['batch_size']),

        'nr_inference_workers': int(args['nr_inference_workers']),
        'nr_post_proc_workers': int(args['nr_post_proc_workers']),
    }

    if args['model_mode'] == 'fast':
        run_args['patch_input_shape'] = 256
        run_args['patch_output_shape'] = 164
    else:
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({
            'input_dir': sub_args['input_dir'],
            'output_dir': sub_args['output_dir'],

            'draw_dot': sub_args['draw_dot'],
            'save_qupath': sub_args['save_qupath'],
            'save_raw_map': sub_args['save_raw_map'],
        })

    if sub_cmd == 'wsi':
        run_args.update({
            'input_dir': sub_args['input_dir'],
            'output_dir': sub_args['output_dir'],
            'input_mask_dir': sub_args['input_mask_dir'],
            'cache_path': sub_args['cache_path'],

            'proc_mag': int(sub_args['proc_mag']),
            'ambiguous_size': int(sub_args['ambiguous_size']),
            'chunk_shape': int(sub_args['chunk_shape']),
            'tile_shape': int(sub_args['tile_shape']),
            'save_thumb': sub_args['save_thumb'],
            'save_mask': sub_args['save_mask'],
        })
    # ***

    # ! TODO: where to save logging
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s', datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    if sub_cmd == 'tile':
        from infer.tile import InferManager
        in_dir = Path(run_args['input_dir'])
        for tma_dir in in_dir.glob('*'):
            if tma_dir.is_dir():
                print(str(tma_dir))
                summary_list = list(tma_dir.glob('**/*summary_results.csv*'))
                summ_dict, annotations = proc_summ(cell_info, summary_list[0])
                run_args['summ_dict'] = summ_dict
                run_args['anno'] = annotations
                run_args['cell_info'] = cell_info
                run_args['input_dir'] = str(tma_dir)
                infer = InferManager(**method_args)
                infer.process_file_list(run_args)
                print('process {} done!'.format(str(tma_dir)))
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)
